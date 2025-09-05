#!/usr/bin/env python3
import sys
from astropy.io import fits
import numpy as np

def subtract_fits(file1, file2, output_file):
    ## From the github repo 
    # Open the two FITS files
    hdulist1 = fits.open(file1)
    hdulist2 = fits.open(file2)

    # Check that both files have the same number of HDUs
    if len(hdulist1) != len(hdulist2):
        print("Error: The FITS files have a different number of HDUs.")
        sys.exit(1)

    new_hdus = []

    # Iterate over the HDUs in both files
    for i, (hdu1, hdu2) in enumerate(zip(hdulist1, hdulist2)):
        if hdu1.data is None or hdu2.data is None:
            # Copy headers only for empty HDUs
            new_hdu = hdu1.copy()
        elif isinstance(hdu1.data, np.ndarray) and hdu1.data.dtype.names is None:
            # This is an image HDU
            if hdu1.data.shape != hdu2.data.shape:
                print(f"Error: Data shape mismatch in HDU {i}: {hdu1.data.shape} vs {hdu2.data.shape}")
                sys.exit(1)
            diff_data = hdu1.data.astype(np.float64) - hdu2.data.astype(np.float64)
            new_hdu = fits.PrimaryHDU(data=diff_data, header=hdu1.header) if i == 0 else fits.ImageHDU(data=diff_data, header=hdu1.header)
        elif hdu1.data.dtype.names is not None:
            # This is a table HDU
            col_names = hdu1.data.dtype.names
            new_table_data = []
            
            for col in col_names:
                if hdu1.data[col].dtype.kind in 'fi':  # Check if column contains floats/ints
                    diff_col = hdu1.data[col] - hdu2.data[col]  # Subtract numeric values
                else:
                    diff_col = hdu1.data[col]  # Keep non-numeric data unchanged
                
                new_table_data.append(fits.Column(name=col, array=diff_col, format=hdu1.columns[col].format))
            
            new_hdu = fits.BinTableHDU.from_columns(new_table_data, header=hdu1.header)
        else:
            print(f"Skipping HDU {i}: Unknown data format")
            new_hdu = hdu1.copy()

        new_hdus.append(new_hdu)

    # Create a new HDUList from the processed HDUs and write it out
    new_hdulist = fits.HDUList(new_hdus)
    new_hdulist.writeto(output_file, overwrite=True)
    print(f"Difference file written to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python subtract_fits.py <file1.fits> <file2.fits> <output.fits>")
        sys.exit(1)
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]
    subtract_fits(file1, file2, output_file)
