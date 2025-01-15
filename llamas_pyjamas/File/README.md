# File Module

This module handles file management and I/O operations for the LLAMAS data reduction pipeline.

## Core Functionality

The file module performs:
- FITS file handling and validation
- File sorting and organization
- Header keyword management
- Raw data import/export
- Directory structure maintenance
- Data product output formatting

## Key Files

### `fileLlamas.py`
Main file handling class containing:
- `FileLlamas` class: Core file operations
- FITS file reading/writing
- Header manipulation tools
- File type identification
- Path management utilities
- Output formatting standards

## Usage

```python
from llamas_pyjamas.File.fileLlamas import FileLlamas

# Create file handler
file_handler = FileLlamas()

# Sort raw files
sorted_files = file_handler.sort_raw_files(directory)

# Read FITS file
data, header = file_handler.read_fits(filename)

# Write processed data
file_handler.write_fits(data, header, output_file)

# Validate FITS header
is_valid = file_handler.validate_header(header)
```

The file module provides consistent file handling across the entire reduction pipeline, ensuring data integrity and proper metadata management.