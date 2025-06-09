# LLAMAS Pyjamas Documentation

This directory contains the Sphinx documentation for the LLAMAS Pyjamas project.

## Building the Documentation

### Prerequisites

First, install the documentation requirements:

```bash
pip install -r docs/requirements.txt
```

### Building HTML Documentation

To build the HTML documentation locally:

```bash
cd docs
make html
```

The generated HTML documentation will be available in `docs/_build/html/index.html`.

### Building Other Formats

Sphinx supports multiple output formats:

```bash
# PDF documentation (requires LaTeX)
make latexpdf

# EPUB documentation
make epub

# Single HTML file
make singlehtml

# Plain text
make text
```

### Cleaning Build Files

To clean all generated documentation files:

```bash
make clean
```

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation page
- `llamas_pyjamas.*.rst` - Auto-generated module documentation
- `_static/` - Static files (CSS, images, etc.)
- `_templates/` - Custom Sphinx templates
- `_build/` - Generated documentation output

## Writing Documentation

### Docstring Style

This project uses Google-style docstrings. Example:

```python
def example_function(param1, param2="default"):
    """Brief description of the function.

    Longer description with more details about what the function does.

    Args:
        param1 (str): Description of param1.
        param2 (str, optional): Description of param2. Defaults to "default".

    Returns:
        bool: Description of return value.

    Raises:
        ValueError: Description of when this exception is raised.

    Example:
        >>> result = example_function("hello")
        >>> print(result)
        True
    """
    return True
```

### Adding New Pages

1. Create a new `.rst` file in the `docs/` directory
2. Add the file to the appropriate `toctree` in `index.rst`
3. Rebuild the documentation

### Cross-References

Use Sphinx cross-references to link to other parts of the documentation:

```rst
:doc:`llamas_pyjamas.Extract`
:func:`llamas_pyjamas.Trace.traceLlamas.TraceLlamas.traceSingleCamera`
:class:`llamas_pyjamas.Extract.extractLlamas.ExtractLlamas`
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError during build**: Make sure all dependencies are installed
2. **Import errors**: Check that the package is properly installed in development mode
3. **Missing docstrings**: Ensure all public functions and classes have docstrings

### Dependencies

If you encounter import errors during documentation build, you may need to install additional packages or add them to the `autodoc_mock_imports` list in `conf.py`.

## Contributing

When adding new modules or functions:

1. Write comprehensive Google-style docstrings
2. Add the module to the appropriate `.rst` file
3. Update the main `index.rst` if needed
4. Test the documentation build locally
5. Include examples where appropriate