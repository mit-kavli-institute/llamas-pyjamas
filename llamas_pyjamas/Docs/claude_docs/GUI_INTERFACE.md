# GUI Interface Feature

## Overview
The GUI Interface feature provides an interactive graphical user interface for the LLAMAS data reduction pipeline, making complex spectroscopic data processing accessible to users without command-line experience. It offers visual workflow management, real-time progress monitoring, and integrated data visualization.

## Core Functionality
- **Interactive File Selection**: Browse and select input files with validation
- **Workflow Management**: Visual pipeline control with step-by-step processing
- **Parameter Configuration**: GUI-based parameter adjustment with validation
- **Progress Monitoring**: Real-time processing status and progress bars
- **Data Visualization**: Integrated plotting and image display
- **Batch Processing**: Handle multiple files and observations efficiently
- **Error Handling**: User-friendly error reporting and recovery options

## Key Files
- `GUI/guiExtract.py` - Main extraction workflow GUI
  - `GUI_extract()`: Primary interface for spectrum extraction
  - File handling and validation
  - Processing orchestration and monitoring
- `GUI/header_qt.py` - Qt-based header viewing and editing
- `GUI/obslog_qt.py` - Observation log management interface
- `GUI/obslog.py` - Legacy observation logging utilities

## Interface Components

### File Management
- **File Browser**: Navigate file system with filtering by type
- **Batch Selection**: Multi-file selection with validation
- **File Preview**: Quick inspection of FITS headers and data
- **Path Management**: Save/load common directory paths
- **Format Validation**: Check file format compatibility

### Processing Controls
- **Pipeline Steps**: Visual representation of reduction workflow
- **Parameter Panels**: Organized parameter adjustment interfaces  
- **Processing Queue**: Manage multiple processing jobs
- **Status Display**: Real-time processing status and logging
- **Progress Indicators**: Step completion and overall progress

### Visualization Integration
- **2D Image Display**: View raw and processed detector images
- **1D Spectrum Plots**: Interactive spectrum visualization
- **Trace Overlays**: Display fiber traces on detector images  
- **QA Plots**: Integrated quality assessment visualization
- **Comparison Tools**: Side-by-side before/after displays

## Usage Patterns
```python
from llamas_pyjamas.GUI.guiExtract import GUI_extract

# Interactive extraction with GUI
extraction_file, metadata = GUI_extract(
    fitsfile="science_observation.fits",
    output_dir="reductions/",
    use_bias=bias_file,
    trace_dir="traces/"
)

# Batch processing through GUI
for fits_file in observation_list:
    GUI_extract(fits_file, output_dir=output_dir)
```

## Workflow Integration
- **Pipeline Entry Point**: Main interface for pipeline workflows
- **Configuration Management**: Save/load processing configurations
- **Result Integration**: Seamless connection to downstream processing
- **Quality Control**: Built-in QA visualization and validation
- **Export Options**: Multiple output format support

## User Interface Design

### Main Window Layout
- **Menu Bar**: File operations, settings, help documentation
- **Tool Bar**: Quick access to common operations
- **File Panel**: Input file selection and management
- **Parameter Panel**: Processing parameter configuration
- **Preview Panel**: Data and result visualization  
- **Status Panel**: Progress monitoring and logging
- **Control Panel**: Start/stop/pause processing controls

### Dialog Windows
- **Configuration Dialogs**: Advanced parameter settings
- **Progress Dialogs**: Detailed processing monitoring
- **Error Dialogs**: User-friendly error reporting
- **Help Dialogs**: Context-sensitive documentation
- **Export Dialogs**: Output format and location selection

## Configuration Management
- **Settings Persistence**: Save user preferences between sessions
- **Template Configurations**: Pre-defined parameter sets for common tasks
- **Custom Workflows**: User-defined processing sequences
- **Default Parameters**: Intelligent defaults based on data characteristics
- **Validation**: Real-time parameter validation with feedback

## Error Handling and Recovery
- **User-Friendly Messages**: Clear error descriptions with suggested solutions
- **Processing Recovery**: Resume interrupted processing where possible
- **Diagnostic Tools**: Built-in troubleshooting and validation utilities
- **Log Management**: Detailed logging with user-accessible logs
- **Support Integration**: Easy error reporting and support request generation

## Data Visualization
- **Interactive Plots**: Zoom, pan, and measurement tools
- **Color Schemes**: Customizable color maps and scaling
- **Export Options**: Save plots in publication-ready formats
- **Animation Support**: Time-series and wavelength animations
- **3D Visualization**: Basic cube data visualization capabilities

## Batch Processing Support
- **Queue Management**: Process multiple files with queuing
- **Template Application**: Apply same parameters to multiple datasets
- **Progress Tracking**: Monitor batch processing status
- **Error Recovery**: Handle individual file failures gracefully
- **Result Organization**: Systematic output file organization

## Dependencies
- PyQt5/PyQt6 or PySide2/PySide6 (GUI framework)
- Matplotlib (integrated plotting)
- Astropy (FITS file handling and preview)
- NumPy (data handling and visualization)
- Threading/multiprocessing (background processing)

## Performance Considerations
- **Responsive Interface**: Non-blocking GUI with background processing
- **Memory Management**: Efficient handling of large FITS files
- **Preview Generation**: Fast thumbnail and preview creation
- **Resource Monitoring**: Display memory and CPU usage
- **Optimization Options**: User-selectable performance vs quality trade-offs

## Accessibility Features
- **Keyboard Navigation**: Full keyboard accessibility
- **Scalable Interface**: Adjustable font sizes and element scaling
- **Color Blind Support**: Alternative color schemes
- **Documentation Integration**: Built-in help and tutorials
- **Status Announcements**: Screen reader compatible status updates

## Installation and Deployment
- **Cross-Platform**: Windows, macOS, and Linux support
- **Packaging**: Standalone executable generation
- **Dependencies**: Automatic dependency management
- **Updates**: Built-in update checking and installation
- **Configuration**: System-specific optimization

## Advanced Features
- **Plugin Architecture**: Extensible interface for custom tools
- **Remote Processing**: Submit jobs to remote compute resources
- **Database Integration**: Connect to observation databases
- **Web Interface**: Browser-based interface option (future)
- **Mobile Support**: Basic mobile device compatibility (future)

## User Experience Design
- **Intuitive Workflow**: Natural progression through processing steps
- **Visual Feedback**: Clear indication of current state and next steps  
- **Undo/Redo**: Ability to reverse parameter changes
- **Favorites**: Quick access to frequently used files and settings
- **Recent Items**: History of recent files and configurations