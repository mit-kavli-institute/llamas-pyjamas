# LLAMAS Pyjamas Documentation Build Instructions

This document provides complete instructions for building the HTML documentation for the LLAMAS Pyjamas project using Sphinx.

## Prerequisites

### System Requirements

- Python 3.8 or later
- pip package manager
- Git (for cloning the repository)

### Required Python Packages

Install the documentation requirements:

```bash
pip install -r docs/requirements.txt
```

This will install:
- **Sphinx** (>= 5.0.0) - Documentation generator
- **sphinx-rtd-theme** (>= 1.0.0) - Read the Docs theme
- **sphinx-autodoc-typehints** - Type hints support
- **Scientific packages** (numpy, scipy, matplotlib, astropy) - For autodoc to work with the codebase

## Quick Start

### 1. Navigate to Documentation Directory

```bash
cd docs
```

### 2. Build HTML Documentation

```bash
make html
```

### 3. View Documentation

Open `docs/_build/html/index.html` in your web browser:

```bash
# On macOS
open _build/html/index.html

# On Linux
xdg-open _build/html/index.html

# On Windows
start _build/html/index.html
```

## Detailed Build Instructions

### Building Different Output Formats

Sphinx supports multiple output formats:

```bash
# HTML documentation (most common)
make html

# PDF documentation (requires LaTeX installation)
make latexpdf

# EPUB documentation
make epub

# Single HTML file
make singlehtml

# Plain text
make text

# Manual pages
make man

# Check for broken links
make linkcheck

# Spell check (requires sphinxcontrib-spelling)
make spelling
```

### Windows Users

If you're on Windows, use `make.bat` instead of `make`:

```batch
make.bat html
make.bat clean
```

### Alternative Build Method

If `make` is not available, you can use sphinx-build directly:

```bash
# Build HTML documentation
sphinx-build -b html . _build/html

# Clean previous builds
rm -rf _build

# Build with verbose output
sphinx-build -v -b html . _build/html
```

## Configuration

### Sphinx Configuration (`conf.py`)

The documentation is configured in `docs/conf.py`. Key settings include:

- **Theme**: `sphinx_rtd_theme` (Read the Docs theme)
- **Extensions**: 
  - `sphinx.ext.autodoc` - Automatic documentation from docstrings
  - `sphinx.ext.napoleon` - Google/NumPy style docstring support
  - `sphinx.ext.viewcode` - Add source code links
  - `sphinx.ext.intersphinx` - Cross-reference other projects
- **Google-style docstrings**: Enabled via Napoleon extension
- **Type hints**: Displayed in descriptions via `autodoc_typehints = 'description'`

### Theme Customization

The Read the Docs theme can be customized in `conf.py`:

```python
html_theme_options = {
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. ModuleNotFoundError during build

**Problem**: Sphinx can't import your package modules.

**Solutions**:
- Ensure the package is installed: `pip install -e .`
- Check that the package path is in `sys.path` (configured in `conf.py`)
- Add missing dependencies to `autodoc_mock_imports` in `conf.py`

#### 2. Missing Dependencies

**Problem**: Import errors for external packages.

**Solution**: Install missing packages or add them to mock imports:

```python
# In conf.py
autodoc_mock_imports = [
    'pyds9',
    'ray',
    'cloudpickle',
    'pypeit',
]
```

#### 3. Build Errors with Type Hints

**Problem**: Sphinx fails to process type hints.

**Solution**: Install `sphinx-autodoc-typehints`:

```bash
pip install sphinx-autodoc-typehints
```

#### 4. LaTeX/PDF Build Issues

**Problem**: `make latexpdf` fails.

**Solution**: Install LaTeX distribution:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-recommended texlive-latex-extra

# macOS with Homebrew
brew install mactex

# Or use smaller BasicTeX
brew install basictex
```

#### 5. Memory Issues with Large Codebases

**Problem**: Sphinx runs out of memory.

**Solutions**:
- Build modules individually
- Increase system memory
- Use `autodoc_mock_imports` for heavy dependencies

#### 6. Broken Cross-References

**Problem**: Links to other modules don't work.

**Solution**: Check module paths and use proper Sphinx syntax:

```rst
:doc:`llamas_pyjamas.Extract`
:func:`llamas_pyjamas.Trace.traceLlamas.TraceLlamas.traceSingleCamera`
:class:`llamas_pyjamas.Extract.extractLlamas.ExtractLlamas`
```

### Getting Detailed Error Information

For more detailed error information:

```bash
# Verbose output
make html SPHINXOPTS="-v"

# Show all warnings
make html SPHINXOPTS="-W"

# Debug mode
make html SPHINXOPTS="-vvv"
```

## Continuous Integration

### GitHub Actions

For automated documentation building, create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r docs/requirements.txt
        pip install -e .
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

### Read the Docs

To build on Read the Docs:

1. Connect your GitHub repository to Read the Docs
2. Create `.readthedocs.yaml` in the repository root:

```yaml
version: 2

build:
  os: ubuntu-20.04
  tools:
    python: "3.9"

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .

sphinx:
  configuration: docs/conf.py
```

## Advanced Usage

### Custom CSS Styling

Add custom CSS by creating `docs/_static/custom.css`:

```css
/* Custom styling for LLAMAS documentation */
.wy-nav-content {
    max-width: 1200px;
}
```

Then reference it in `conf.py`:

```python
html_static_path = ['_static']
html_css_files = ['custom.css']
```

### Adding Custom Templates

Create custom Sphinx templates in `docs/_templates/`:

```html
<!-- _templates/breadcrumbs.html -->
<div class="custom-breadcrumbs">
  <!-- Custom breadcrumb HTML -->
</div>
```

### Jupyter Notebook Integration

To include Jupyter notebooks in documentation:

1. Install nbsphinx: `pip install nbsphinx`
2. Add to `conf.py` extensions: `'nbsphinx'`
3. Place notebooks in docs directory
4. Reference in `.rst` files

## Performance Optimization

### Parallel Building

Enable parallel building for faster documentation generation:

```bash
make html SPHINXOPTS="-j auto"
```

### Incremental Builds

For development, use incremental builds:

```bash
# Only rebuild changed files
sphinx-build -b html . _build/html

# Force rebuild all files
sphinx-build -a -b html . _build/html
```

## Maintenance

### Regular Tasks

1. **Update dependencies**: Regularly update `docs/requirements.txt`
2. **Check links**: Run `make linkcheck` to find broken links
3. **Review warnings**: Address Sphinx warnings during builds
4. **Update examples**: Keep code examples current with API changes

### Monitoring Build Health

- Set up notifications for failed documentation builds
- Regularly review and update mock imports
- Monitor documentation coverage with `sphinx.ext.coverage`

## Support

If you encounter issues not covered in this guide:

1. Check the [Sphinx documentation](https://www.sphinx-doc.org/)
2. Review the [Read the Docs theme documentation](https://sphinx-rtd-theme.readthedocs.io/)
3. Search existing issues on the project's GitHub repository
4. Create a new issue with detailed error information and build logs

## Summary

The LLAMAS Pyjamas documentation is built using:

- **Sphinx** with **Read the Docs theme**
- **Google-style docstrings** via Napoleon
- **Automatic API documentation** via autodoc
- **Cross-references** to external projects
- **Multiple output formats** (HTML, PDF, EPUB)

The build process is designed to be straightforward and automated, supporting both local development and continuous integration workflows.