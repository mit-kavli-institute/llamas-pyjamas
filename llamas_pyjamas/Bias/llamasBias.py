

"""
This module provides functionality to create a master bias frame from a directory of bias images or a list of files.
Classes:
    BiasLlamas: A class to handle bias image processing and creation of a master bias frame.
Usage:
    To use this module, instantiate the BiasLlamas class with a directory path containing bias images or a list of bias image files.
    Then call the `master_bias` method to create and save the master bias frame.
Example:
    bias_llamas = BiasLlamas('/path/to/bias/images')

    A class to handle bias image processing and creation of a master bias frame.
    Attributes:
        bias_path (str): The directory path containing bias images.
        files (list): A list of bias image files.
    Methods:
        master_bias(): Creates and saves the master bias frame from the provided bias images.

        Initializes the BiasLlamas class with the provided input data.
        Args:
            input_data (str or list): A directory path containing bias images or a list of bias image files.
        Raises:
            ValueError: If the provided directory does not exist or no .fits files are found.
            TypeError: If the input data is not a string or a list.

        Creates and saves the master bias frame from the provided bias images.
        The method reads bias images from the specified directory or list, combines them using average method with sigma clipping,
        and saves the combined bias frame as 'combined_bias.fit' in the specified directory.
        Raises:
            ValueError: If no bias images are found in the specified directory or list.
        """
import os
from astropy.nddata import CCDData
from astropy.stats import mad_std

#import ccdproc as ccdp
import matplotlib.pyplot as plt
import numpy as np
import argparse
from astropy.io import fits
from llamas_pyjamas.config import CALIB_DIR



class BiasLlamas:
    def __init__(self, input_data) -> None:
        """
        Initializes the BiasLlamas object with either a directory path, a file path, or a list of file paths.

        Parameters
        ----------
        input_data : str or list
            If a string, it is treated as either a directory (from which all .fits files are grabbed)
            or a full file path. If a list, it should contain file paths to .fits files.
        """
        from llamas_pyjamas.config import CALIB_DIR  # ensure CALIB_DIR is available

        # Case when a list is provided
        if isinstance(input_data, list):
            # Keep only files ending with '.fits'
            self.files = [file for file in input_data if str(file).endswith('.fits')]
            if not self.files:
                raise ValueError("No .fits files found in the provided list.")
            # Use the directory of the first file if available, else default to CALIB_DIR
            self.bias_path = os.path.dirname(self.files[0]) if os.path.dirname(self.files[0]) else CALIB_DIR

        # Case when a string is provided
        elif isinstance(input_data, str):
            # If it's an existing file, use it directly
            if os.path.isfile(input_data):
                self.files = [input_data]
                self.bias_path = os.path.dirname(input_data)
            # If it's an existing directory, search for .fits files
            elif os.path.isdir(input_data):
                self.bias_path = input_data
                self.files = [os.path.join(input_data, f) for f in os.listdir(input_data) if f.endswith('.fits')]
            else:
                # If the string is just a filename without path, assume it resides in CALIB_DIR
                potential = os.path.join(CALIB_DIR, input_data)
                if os.path.isfile(potential):
                    self.files = [potential]
                    self.bias_path = os.path.dirname(potential)
                else:
                    raise ValueError("Provided input_data does not exist as a file or a directory.")
            if not self.files:
                raise ValueError("No .fits files found in the provided input_data.")

        else:
            raise TypeError("input_data must be a string (file or directory path) or a list of file paths.")

        
    def master_bias(self) -> None:
        """
        Create a master bias frame from a list of bias images.
        Each input FITS file is assumed to have a primary HDU and one or more extensions,
        where each extension contains a bias image from a detector.
        The function first asserts that each file’s primary HDU has matching EXPTIME and READOUT
        values. Then, for each file, it groups the extensions by the 'COLOR' and 'BENCHSIDE' header
        keywords, stacks the corresponding images across all files, computes their median (using nanmedian),
        and builds a new FITS file with:
          - A primary HDU (using the primary header from the first file)
          - One ImageHDU per group, where the data is the median combination of all matching extensions.
        The resulting combined FITS file is written to 'combined_bias.fits' in the bias_path.
        """
        # Open the first file to get the primary header as reference
        first_file = os.path.join(self.bias_path, self.files[0]) if not os.path.isabs(self.files[0]) else self.files[0]
        with fits.open(first_file) as hdulist_first:
            primary_hdr = hdulist_first[0].header.copy()
            
        combined_hdus = []
        # Create primary HDU using the primary header from the first file
        primary_hdu = fits.PrimaryHDU(header=primary_hdr)
        combined_hdus.append(primary_hdu)
        
        # Dictionary to group extension data by (COLOR, BENCHSIDE)
        groups = {}
        
        # Loop through each file
        for file in self.files:
            # Handle file path (if not absolute, join with bias_path)
            file_path = os.path.join(self.bias_path, file) if not os.path.isabs(file) else file
            with fits.open(file_path) as hdul:
                # Assert primary HDU header values are consistent among all files
                file_primary_hdr = hdul[0].header
                
                # Loop through each extension (skip primary, i.e. index 0)
                for ext in range(1, len(hdul)):
                    hdr = hdul[ext].header.copy()
                    try:
                        # Group by 'COLOR' and 'BENCHSIDE' header keys
                        key = (hdr["COLOR"], hdr["BENCH"], hdr["SIDE"])
                    except KeyError as e:
                        raise KeyError(f"Extension in file {file} missing required key: {e}")
                    
                    # Append data to the corresponding group
                    if key not in groups:
                        groups[key] = {"data": [], "header": hdr}
                    groups[key]["data"].append(hdul[ext].data)
        
        # Process each group by computing the median combined data
        for (color, bench, side), group in groups.items():
            data_array = np.array(group["data"])
            combined_data = np.nanmedian(data_array, axis=0)
            combined_header = group["header"]
            combined_header["COMBINED"] = True
            # Record the grouping information in the header for convenience
            combined_header["COLOR"] = color
            combined_header["BENCH"] = bench  # note: shortened key to 8 characters if needed
            combined_header["SIDE"] = side
            # Create an image HDU for this group. The HDU name encodes the group key.
            image_hdu = fits.ImageHDU(data=combined_data, header=combined_header, name=f"COL_{color}_BNCH_{bench}")
            combined_hdus.append(image_hdu)
        
        # Assemble all HDUs into an HDUList and write to file
        hdul_out = fits.HDUList(combined_hdus)
        out_filename = os.path.join(self.bias_path, 'combined_bias.fits')
        hdul_out.writeto(out_filename, overwrite=True)

        return out_filename

 

# ---------------------------------------------------------------------------
# Median combination of raw multi-extension frames (biases, darks)
# ---------------------------------------------------------------------------

def detector_key(header):
    """``(color, bench, side)`` of an image extension from COLOR/BENCH/SIDE, else
    CAM_NAME (e.g. ``1A_Red``); None when neither identifies the detector."""
    if 'COLOR' in header:
        return (str(header['COLOR']).lower(), str(header.get('BENCH', '')),
                str(header.get('SIDE', '')).upper())
    parts = str(header.get('CAM_NAME', '')).split('_')
    if len(parts) >= 2 and len(parts[0]) >= 2:
        return parts[1].lower(), parts[0][0], parts[0][1].upper()
    return None


def _scaled_image(hdu, dtype=np.float32):
    """Stored values of an extension with BSCALE/BZERO applied, as *dtype*."""
    data = hdu.data.astype(dtype)
    bscale = hdu.header.get('BSCALE', 1)
    bzero = hdu.header.get('BZERO', 0)
    if bscale != 1:
        data *= dtype(bscale)
    if bzero:
        data += dtype(bzero)
    return data


def median_combine_mef(paths, out_path, subtract=None, extra_primary=None):
    """Median-combine raw LLAMAS MEFs extension by extension and write *out_path*.

    Extensions are matched across files by detector (``detector_key``), not by
    position, so a missing or reordered camera in one frame only drops that
    detector from the combination; an extension without COLOR/CAM_NAME is
    matched by HDU position instead. Each output extension is the per-pixel median
    of the frames that have that detector, as float32 in the raw orientation
    (no trimming, no flip), with the first frame's extension header. The primary
    header is the first frame's, plus ``NCOMBINE`` and one ``COMBSRCn`` card per
    input. Frames are read memory-mapped one detector at a time.

    Parameters
    ----------
    paths : list of str
        Raw frames to combine (one or more).
    out_path : str
        Output file, overwritten.
    subtract : str, optional
        A MEF (e.g. the master bias) subtracted from every frame before the
        median, matched by detector. Used to build a bias-subtracted master dark.
    extra_primary : dict, optional
        ``{key: value | (value, comment)}`` written to the primary header.

    Returns
    -------
    str
        *out_path*.
    """
    if not paths:
        raise ValueError('median_combine_mef: no input frames')
    hduls = [fits.open(p, do_not_scale_image_data=True, memmap=True) for p in paths]
    sub_hdul = fits.open(subtract, do_not_scale_image_data=True, memmap=True) if subtract else None
    try:
        sub_by_key = {}
        if sub_hdul is not None:
            for hdu in sub_hdul[1:]:
                key = detector_key(hdu.header)
                if key is not None and hdu.data is not None:
                    sub_by_key[key] = hdu

        primary = fits.PrimaryHDU(header=hduls[0][0].header.copy())
        primary.header['NCOMBINE'] = (len(paths), 'Frames median-combined')
        for i, p in enumerate(paths, start=1):
            primary.header[f'COMBSRC{i}'] = (os.path.basename(p), 'Input frame')
        if subtract:
            primary.header['COMBSUB'] = (os.path.basename(subtract), 'Subtracted from each frame')
        for key, value in (extra_primary or {}).items():
            primary.header[key] = value
        out = fits.HDUList([primary])

        # Output extension order follows the first frame. Extensions are matched
        # by detector; one without COLOR/CAM_NAME is matched by HDU position.
        for position, first_hdu in enumerate(hduls[0][1:], start=1):
            if first_hdu.data is None:
                continue
            key = detector_key(first_hdu.header)
            stack = []
            for hdul in hduls:
                if key is None:
                    if position < len(hdul) and hdul[position].data is not None:
                        stack.append(_scaled_image(hdul[position]))
                    continue
                for hdu in hdul[1:]:
                    if detector_key(hdu.header) == key and hdu.data is not None:
                        stack.append(_scaled_image(hdu))
                        break
            data = np.median(np.stack(stack), axis=0).astype(np.float32)
            if key in sub_by_key:
                data -= _scaled_image(sub_by_key[key])
            elif key is None and sub_hdul is not None and position < len(sub_hdul):
                data -= _scaled_image(sub_hdul[position])
            header = first_hdu.header.copy()
            for card in ('BSCALE', 'BZERO', 'BITPIX'):
                header.pop(card, None)
            header['NCOMBINE'] = (len(stack), 'Frames median-combined for this detector')
            out.append(fits.ImageHDU(data=data, header=header, name=first_hdu.name))
    finally:
        for hdul in hduls:
            hdul.close()
        if sub_hdul is not None:
            sub_hdul.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.writeto(out_path, overwrite=True)
    return out_path
