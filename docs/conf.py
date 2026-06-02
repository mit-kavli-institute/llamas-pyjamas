# Configuration file for the Sphinx documentation builder.
#
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Make the llamas_pyjamas package importable for autodoc (repo root is one
# level up from this docs/ directory).
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'LLAMAS Pyjamas'
copyright = '2026, LLAMAS Development Team'
author = 'LLAMAS Development Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Heavy / environment-specific dependencies are mocked so the package can be
# imported on a clean machine (e.g. CI) without installing them. Sphinx's mock
# objects support subclassing and decorator use, so this is safe for the
# pipeline's Ray actors and PypeIt-derived helpers.
autodoc_mock_imports = [
    'ray',
    'cloudpickle',
    'pypeit',
    'pyds9',
    'PyQt6',
    'lacosmic',
]

# -- Napoleon (Google-style docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = False
# Render "Attributes:" sections inline as :ivar: rather than as separate
# attribute directives. This avoids duplicate-object warnings on dataclasses
# whose fields are both auto-documented and listed in the class docstring.
napoleon_use_ivar = True

# Cosmetic autodoc noise from Ray actor classes (decorated with @ray.remote,
# which is mocked at build time) is suppressed; the classes still appear.
suppress_warnings = ['autodoc.mocked_object']

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# -- HTML output -------------------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'LLAMAS Pyjamas'

html_theme_options = {
    'source_repository': 'https://github.com/mit-kavli-institute/llamas-pyjamas/',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'navigation_with_keys': True,
}

# Suppress noise from modules that emit print()/logging at import time.
nitpicky = False


# The Ray actor wrapper classes (decorated with @ray.remote, which is mocked at
# build time) duplicate their base classes' members and cannot be introspected
# cleanly under the mock. Skip documenting them; the base classes they wrap
# (TraceLlamas, ExtractLlamas, the CRR constructors) are fully documented.
_RAY_ACTOR_CLASSES = {'TraceRay', 'ExtractLlamasRay', 'CRRWorker'}


def _skip_ray_actor_classes(app, what, name, obj, skip, options):
    if name in _RAY_ACTOR_CLASSES:
        return True
    return skip


def setup(app):
    app.connect('autodoc-skip-member', _skip_ray_actor_classes)
