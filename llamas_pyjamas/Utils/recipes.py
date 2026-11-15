"""Processing recipes for each frame type, and the provenance stamps that record them.

Each frame type gets a fixed set of processing steps. Calibration frames must not inherit
science processing: cosmic-ray cleaning of a lamp flat, for example, lowers bright fibre
cores and leaves false dips in the flat that turn into fake emission lines in the science
(``Flat/README.md``).

Products record the recipe they were made with in their headers:

- ``RECIPE``: the recipe name;
- ``RCPHASH``: a hash of the recipe's fields and product format;
- ``PIPEVER``: the package version;
- ``FIBIDS``: the fibre-id convention, which is ``physical`` (fibremap position).

A consumer checks the stamp with :func:`check_recipe`, and a resumed run refuses products
whose ``RCPHASH`` no longer matches. Bump a recipe's ``fmt`` whenever a change to the code
makes its products different, so that older products are rebuilt rather than reused.
"""

import hashlib
import logging
import sys
from dataclasses import astuple, dataclass

logger = logging.getLogger(__name__)


class CalibrationRecipeError(RuntimeError):
    """A product was made with a different recipe than its consumer expects."""


@dataclass(frozen=True)
class Recipe:
    """The processing one frame type gets.

    ``cosmic_rays``: L.A.Cosmic may run (science-like frames, per the configured channels).
    ``pixel_flat``: the 2D pixel flat is divided in before extraction.
    ``dark``: a matched master dark is subtracted at bias-first.
    ``sky_subtract``: a sky model is subtracted from the extracted spectra.
    ``fmt``: product format version, part of the hash.
    """
    name: str
    cosmic_rays: bool
    pixel_flat: bool
    dark: bool
    sky_subtract: bool
    fibre_ids: str = 'physical'
    fmt: int = 1


RECIPES = {
    # Traces, pixel flat and per-fibre lamp model come from the lamp flat. fmt 2: no
    # cosmic-ray cleaning, physical fibre ids, combined exposures.
    'lamp_flat': Recipe('lamp_flat', cosmic_rays=False, pixel_flat=False, dark=False,
                        sky_subtract=False, fmt=2),
    # Twilight: fibre throughput. fmt 2: physical fibre ids.
    'twilight': Recipe('twilight', cosmic_rays=False, pixel_flat=True, dark=False,
                       sky_subtract=False, fmt=2),
    'arc': Recipe('arc', cosmic_rays=False, pixel_flat=False, dark=False, sky_subtract=False),
    'science': Recipe('science', cosmic_rays=True, pixel_flat=True, dark=True,
                      sky_subtract=True),
    'standard': Recipe('standard', cosmic_rays=True, pixel_flat=True, dark=True,
                       sky_subtract=True),
    'sky_frame': Recipe('sky_frame', cosmic_rays=True, pixel_flat=True, dark=True,
                        sky_subtract=False),
}

CALIBRATION_RECIPES = ('lamp_flat', 'twilight', 'arc')


def get_recipe(name):
    """The recipe called ``name``; raises ``KeyError`` naming the known ones."""
    try:
        return RECIPES[name]
    except KeyError:
        raise KeyError(f"unknown recipe {name!r}; known: {', '.join(RECIPES)}") from None


def recipe_hash(recipe):
    """Short, stable hash of a recipe's fields (12 hex characters)."""
    if isinstance(recipe, str):
        recipe = get_recipe(recipe)
    text = '|'.join(str(v) for v in astuple(recipe))
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def pipeline_version():
    """The package version, or 'unknown' before the package has finished importing."""
    return str(getattr(sys.modules.get('llamas_pyjamas'), '__version__', 'unknown'))


def stamp_recipe(header, recipe):
    """Write RECIPE, RCPHASH, PIPEVER and FIBIDS into a FITS header (or dict); returns it."""
    if isinstance(recipe, str):
        recipe = get_recipe(recipe)
    header['RECIPE'] = (recipe.name, 'Processing recipe (Utils/recipes.py)')
    header['RCPHASH'] = (recipe_hash(recipe), 'Hash of the recipe and product format')
    header['PIPEVER'] = (pipeline_version(), 'llamas_pyjamas version')
    header['FIBIDS'] = (recipe.fibre_ids, 'Fibre-id convention of per-fibre tables')
    return header


def _value(header, key):
    """Header value, unwrapping the (value, comment) tuples a dict header may hold."""
    value = header.get(key) if header is not None else None
    return value[0] if isinstance(value, tuple) else value


def recipe_problem(header, expected):
    """Why ``header`` does not carry recipe ``expected``, or None when it does."""
    recipe = get_recipe(expected)
    name = _value(header, 'RECIPE')
    if name is None:
        return f"no recipe stamp (made before recipes were recorded; expected {expected!r})"
    if name != recipe.name:
        return f"made with recipe {name!r}, expected {expected!r}"
    found = _value(header, 'RCPHASH')
    if found != recipe_hash(recipe):
        return (f"recipe {expected!r} has changed since it was made "
                f"(RCPHASH {found}, now {recipe_hash(recipe)})")
    if not recipe.cosmic_rays and _value(header, 'CRCLEAN'):
        return f"cosmic-ray cleaned (CRCLEAN=True), which recipe {expected!r} forbids"
    return None


def check_recipe(header, expected, what='product', strict=True):
    """Check that ``header`` carries recipe ``expected``.

    Returns True when it does. Otherwise raises :class:`CalibrationRecipeError` when
    ``strict`` (batch runs), or logs a warning and returns False (the GUI).
    """
    problem = recipe_problem(header, expected)
    if problem is None:
        return True
    message = f"{what}: {problem}. Rebuild it (re-run with clobber = true)."
    if strict:
        raise CalibrationRecipeError(message)
    logger.warning(message)
    return False
