# GUI Module

This module provides a Graphical User Interface for the LLAMAS data reduction pipeline, making it easier for users to process and visualise spectroscopic data.

## Core Functionality

The GUI module offers:
- Interactive file selection and batch processing
- Real-time visualisation of traces and spectra
- Progress monitoring for reduction steps
- Configuration parameter adjustment
- Results preview and validation

## Key Files

### `llamasgui.py`
Main GUI application containing:
- `LlamasGUI` class: Primary GUI window
- File handling interface
- Processing controls
- Visualisation widgets

### `guiutils.py`
Helper functions for the GUI including:
- Custom widgets and dialogs
- Data visualisation tools
- Parameter validation
- State management

## Usage

Launch the GUI application:

```python
from llamas_pyjamas.GUI.llamasgui import LlamasGUI

# Create and show GUI
gui = LlamasGUI()
gui.show()
```

The GUI provides an intuitive interface for:
- Loading raw FITS files
- Setting reduction parameters
- Running trace identification
- Performing spectral extraction
- Visualising results
- Saving processed data

Note: The GUI requires PyQt5/PySide2 and matplotlib for operation.