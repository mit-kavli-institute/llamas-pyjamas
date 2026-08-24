# LLAMAS Pyjamas Documentation

Sphinx source for the LLAMAS Pyjamas documentation. The API reference is generated
automatically from the package's Google-style docstrings.

This directory contains **only source files**. Build artefacts (`_build/`) are
git-ignored and must never be committed.

The one exception is [`reference_material/`](reference_material/README.md) — archived material
kept here for tidiness rather than at the repo root. It is excluded from the build via
`exclude_patterns` in `conf.py` and never reaches the published site.

## Building locally

```bash
pip install -r docs/requirements.txt   # one-time
cd docs
make html
```

The site is written to `docs/_build/html/`. Open `docs/_build/html/index.html`.

Use `make clean` to remove the build, and `make help` to list other formats.

> **Note on dependencies:** Heavy/environment-specific pipeline packages (`ray`,
> `pypeit`, `cloudpickle`, `PyQt6`, `lacosmic`, `pyds9`) and the Qt matplotlib backend
> (`matplotlib.backends.backend_qtagg`) are *mocked* during the build via
> `autodoc_mock_imports` in `conf.py`, so you do **not** need them installed to build the
> docs — only the lightweight deps in `requirements.txt`.

## Regenerating the API reference

The per-module pages under `docs/api/` are generated with `sphinx-apidoc`. Regenerate them
after adding or removing modules:

```bash
cd <repo-root>
sphinx-apidoc --force --separate --module-first -o docs/api llamas_pyjamas \
  llamas_pyjamas/GUI llamas_pyjamas/Postprocessing llamas_pyjamas/Scripts \
  llamas_pyjamas/Tests llamas_pyjamas/Tutorials llamas_pyjamas/Sky/diagnosis \
  'llamas_pyjamas/**/*backup*.py' \
  llamas_pyjamas/analyze_spectrum_X90_Y58.py llamas_pyjamas/check_arc_simple.py \
  llamas_pyjamas/check_detector_order.py llamas_pyjamas/check_reference_arc.py \
  llamas_pyjamas/example_pixel_to_fiber.py llamas_pyjamas/extract_cube_spectrum_correct.py \
  llamas_pyjamas/flux_calibration.py llamas_pyjamas/plot_galaxy_spectrum.py \
  llamas_pyjamas/sky_subtract_spectra.py llamas_pyjamas/test_normalized_flat_fix.py \
  llamas_pyjamas/verify_cube_position.py \
  llamas_pyjamas/Arc/arcReidentify.py llamas_pyjamas/Arc/nistArc.py \
  llamas_pyjamas/Flat/lampThroughput.py llamas_pyjamas/Flat/twilightTie.py \
  llamas_pyjamas/Utils/readmode.py llamas_pyjamas/CubeViewer/__main__.py
```

The exclusions fall into four groups.

**Not importable packages.** `GUI`, `Postprocessing` and `Scripts` have no `__init__.py`, so
`sphinx-apidoc` skips them regardless; they are listed to record the intent. Data-only
directories (`LUT`, `Config`, `output`, `mastercalib`, `calib`, `reduced`, `logs`) contain no
`.py` files and need no exclusion.

**Not public API.** `Tests` is the test suite — its modules mutate `sys.path` and set
`QT_QPA_PLATFORM` at import time. `Sky/diagnosis` is a directory of one-off investigation
scripts. `Tutorials` holds notebooks. `CubeViewer/__main__.py` is a `python -m` entry stub.

**Cannot be imported by autodoc.** `Arc/arcReidentify.py` and `Arc/nistArc.py` run top-level
loops at import and import a `LlamasPipeline` module that no longer exists.

**Source not committed.** `Flat/lampThroughput.py`, `Flat/twilightTie.py`, `Utils/readmode.py`
and the loose top-level analysis scripts are untracked, so CI would build empty pages for them.
Drop an entry from this list once its source is committed.

> **After a `--force` run, re-apply two hand edits.** `docs/api/llamas_pyjamas.rst` and
> `docs/api/llamas_pyjamas.Trace.rst` are maintained by hand: their `automodule` directives carry
> **no** `:members:`/`:undoc-members:`/`:show-inheritance:` options — both packages re-export via
> `import *`, and the options would duplicate every submodule's members — and each carries an
> explanatory `.. note::`. `--force` overwrites both; restore them before committing.

## Publishing to GitHub Pages

Deployment is automated by `.github/workflows/docs.yml`: on every push to the
`documentation` branch, the workflow builds the docs and publishes them to GitHub Pages.
To enable it once:

1. Push the `documentation` branch (with the workflow) to GitHub.
2. **Settings → Pages → Build and deployment → Source** = **GitHub Actions**.
3. **Settings → Environments → `github-pages` → Deployment branches** — allow the
   `documentation` branch (the default `github-pages` environment otherwise restricts
   deployments to the default branch and the `deploy` job will fail).

The site will be served at `https://mit-kavli-institute.github.io/llamas-pyjamas/`.

To publish manually instead (without the workflow):

```bash
cd docs && make html
pip install ghp-import
ghp-import -n -p -f docs/_build/html   # pushes to the gh-pages branch
```

## Writing docstrings

The project uses **Google-style** docstrings (parsed by `sphinx.ext.napoleon`):

```python
def example(param1, param2="default"):
    """Brief description.

    Args:
        param1 (str): Description of param1.
        param2 (str, optional): Description. Defaults to "default".

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: When something is invalid.
    """
    return True
```

After adding a new module, regenerate the API reference (above) and rebuild.
