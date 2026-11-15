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
import json
import logging
import os
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
    # Traces, pixel flat and per-fibre lamp model come from the configured lamp flat
    # alone. fmt 2: no cosmic-ray cleaning, physical fibre ids. fmt 3: lamp-flat exposures
    # are no longer combined (products from the short-lived combine are refused).
    'lamp_flat': Recipe('lamp_flat', cosmic_rays=False, pixel_flat=False, dark=False,
                        sky_subtract=False, fmt=3),
    # Twilight: fibre throughput. fmt 2: physical fibre ids.
    'twilight': Recipe('twilight', cosmic_rays=False, pixel_flat=True, dark=False,
                       sky_subtract=False, fmt=2),
    'arc': Recipe('arc', cosmic_rays=False, pixel_flat=False, dark=False, sky_subtract=False),
    # A lamp flat extracted through the pixel flat for a fibre-flat RSS (no twilight).
    'dome_flat': Recipe('dome_flat', cosmic_rays=False, pixel_flat=True, dark=False,
                        sky_subtract=False),
    # Science-like frames. fmt 2: cosmic-ray pixels are missing data in the extraction
    # (NaN + MASK_CR_MISSING), not L.A.Cosmic's replacement values.
    'science': Recipe('science', cosmic_rays=True, pixel_flat=True, dark=True,
                      sky_subtract=True, fmt=2),
    'standard': Recipe('standard', cosmic_rays=True, pixel_flat=True, dark=True,
                       sky_subtract=True, fmt=2),
    'sky_frame': Recipe('sky_frame', cosmic_rays=True, pixel_flat=True, dark=True,
                        sky_subtract=False, fmt=2),
}

CALIBRATION_RECIPES = ('lamp_flat', 'twilight', 'arc', 'dome_flat')


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


#: Extraction methods recorded as EXTMETH ('optimal' is the user-facing alias of 'horne').
EXTRACT_METHODS = ('horne', 'boxcar', 'legacy')
#: Method of products made before EXTMETH was recorded: the pipeline default until 2026-11-15.
LEGACY_EXTMETH = 'boxcar'
#: Header keys that record the extraction method.
EXTRACTION_STAMPS = ('EXTMETH', 'EXTHW')


def normalise_method(method):
    """'horne' for 'optimal' (its user-facing alias), otherwise the lower-case name."""
    m = str(method).strip().lower()
    return 'horne' if m == 'optimal' else m


def run_extraction_method():
    """``(method, boxcar half-width or None)`` of this run, or None outside a pipeline run.

    From ``LLAMAS_EXTRACT_METHOD`` and ``LLAMAS_BOXCAR_HALFWIDTH``, which reduce.py sets for
    every extraction of a run (mixing methods breaks the per-fibre throughput calibration).
    """
    method = os.environ.get('LLAMAS_EXTRACT_METHOD')
    if not method:
        return None
    method = normalise_method(method)
    half = float(os.environ.get('LLAMAS_BOXCAR_HALFWIDTH', '2.5')) if method == 'boxcar' else None
    return method, half


def stamp_extraction_method(header, method=None, halfwidth=None):
    """Write EXTMETH (and EXTHW for boxcar) into a header (or dict); returns it.

    ``method`` None takes the run's (:func:`run_extraction_method`); outside a run nothing is
    written.
    """
    if method is None:
        run = run_extraction_method()
        if run is None:
            return header
        method, halfwidth = run
    method = normalise_method(method)
    header['EXTMETH'] = (method, 'Extraction method (Extract/README.md)')
    if method == 'boxcar':
        if halfwidth is None:
            halfwidth = float(os.environ.get('LLAMAS_BOXCAR_HALFWIDTH', '2.5'))
        header['EXTHW'] = (float(halfwidth), 'Boxcar aperture half-width [px]')
    return header


def extraction_method_problem(header, run=None):
    """Why a product was extracted differently from this run, or None.

    ``run`` is ``(method, boxcar half-width)``; None takes :func:`run_extraction_method`, and
    outside a run nothing is checked. A product without EXTMETH counts as ``LEGACY_EXTMETH``
    (the default it was made with); a boxcar product without EXTHW is not checked for width.
    """
    run = run if run is not None else run_extraction_method()
    if run is None:
        return None
    method, half = run
    method = normalise_method(method)
    stamped = _value(header, 'EXTMETH')
    found = normalise_method(stamped or LEGACY_EXTMETH)
    if found != method:
        note = '' if stamped else ' (no EXTMETH stamp: the old default)'
        return f"extracted with {found}{note}; this run extracts with {method}"
    found_half = _value(header, 'EXTHW')
    if method == 'boxcar' and half is not None and found_half is not None \
            and abs(float(found_half) - float(half)) > 1e-6:
        return f"boxcar half-width {float(found_half):g} px; this run uses {float(half):g} px"
    return None


def copy_extraction_method(source, target):
    """Copy EXTMETH/EXTHW from one header to another (where the source has them)."""
    for key in EXTRACTION_STAMPS:
        if source is not None and key in source:
            target[key] = (source[key], source.comments[key]) if hasattr(source, 'comments') \
                else source[key]


def stamp_recipe(header, recipe, method=None, halfwidth=None):
    """Write RECIPE, RCPHASH, PIPEVER, FIBIDS and the extraction method (EXTMETH, EXTHW;
    :func:`stamp_extraction_method`) into a FITS header (or dict); returns it."""
    if isinstance(recipe, str):
        recipe = get_recipe(recipe)
    header['RECIPE'] = (recipe.name, 'Processing recipe (Utils/recipes.py)')
    header['RCPHASH'] = (recipe_hash(recipe), 'Hash of the recipe and product format')
    header['PIPEVER'] = (pipeline_version(), 'llamas_pyjamas version')
    header['FIBIDS'] = (recipe.fibre_ids, 'Fibre-id convention of per-fibre tables')
    stamp_extraction_method(header, method, halfwidth)
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
    return extraction_method_problem(header)


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


def cosmic_rays_for(recipe, requested=None):
    """Whether an extraction made with ``recipe`` cleans cosmic rays.

    Every extraction names its recipe: there is no science default a calibration frame
    could inherit. ``requested`` (the caller's or the config's switch) can turn cleaning
    off where the recipe allows it, never on where it does not.
    """
    if recipe is None:
        raise TypeError("an extraction needs a recipe (Utils/recipes.py: lamp_flat, twilight, "
                        "arc, dome_flat, science, standard, sky_frame); there is no default")
    allowed = get_recipe(recipe).cosmic_rays
    if requested and not allowed:
        raise ValueError(f"recipe {recipe!r} forbids cosmic-ray cleaning")
    return allowed if requested is None else bool(requested)


def recipe_for_frame(header):
    """The recipe for a raw frame from its primary header (the GUI quick look).

    LDLS flats are ``lamp_flat``, sky flats ``twilight``, frames with an arc lamp on ``arc``,
    and anything else ``science``.
    """
    obj = str(header.get('OBJECT', '')).strip().lower()
    if 'ldls' in obj or obj in ('flat', 'dome flat', 'lamp flat'):
        return 'lamp_flat'
    if 'skyflat' in obj.replace(' ', '') or 'twilight' in obj:
        return 'twilight'
    lamps = [k for k in header if 'LAMPS' in k and k.endswith('STATE')]
    if any(str(header.get(k, '')).strip().upper() == 'ON' for k in lamps) or 'arc' in obj:
        return 'arc'
    return 'science'


#: Keys copied from an extraction's primary header into its recipe sidecar.
SIDECAR_KEYS = ('RECIPE', 'RCPHASH', 'PIPEVER', 'FIBIDS', 'CRCLEAN', 'CRID', 'CRCHAN', 'CRMODE',
                'EXTMETH', 'EXTHW')


def sidecar_path(pkl_path):
    """Path of the small JSON beside an extraction pickle that records its recipe."""
    return pkl_path + '.recipe.json'


def write_recipe_sidecar(pkl_path, header):
    """Record an extraction pickle's recipe stamps beside it, readable without loading it."""
    stamps = {k: _value(header, k) for k in SIDECAR_KEYS if _value(header, k) is not None}
    with open(sidecar_path(pkl_path), 'w') as fh:
        json.dump(stamps, fh)
    return stamps


def read_recipe_sidecar(pkl_path):
    """The recipe stamps of an extraction pickle ({} when it has no sidecar)."""
    path = sidecar_path(pkl_path)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)
