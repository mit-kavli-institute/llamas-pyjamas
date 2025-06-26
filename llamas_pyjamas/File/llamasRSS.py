import cloudpickle as pickle
import os
from astropy.io import fits
import numpy as np



class RSSgeneration:
    def __init__(self):
        
        return
    
    def generate_rss(self, extraction_file, output_file):
        
        with open(extraction_file, 'rb') as f:
            _data = pickle.load(f)
            
        extraction_objects = _data['extractions']
        _metadata = _data['metadata']
        
        # Create a new HDU list for the RSS FITS file
        hdul = fits.HDUList()

        # Primary HDU (Header)
        primary_hdu = fits.PrimaryHDU()
        # Copy header from the first extraction object
        if extraction_objects:
            primary_hdu.header = extraction_objects[0].hdr
        hdul.append(primary_hdu)

        # Data extension (contains all extracted spectra)
        if extraction_objects:
            # Stack all extracted spectra into a 2D array
            spectra = np.array([obj.counts for obj in extraction_objects])
            data_hdu = fits.ImageHDU(data=spectra, name='FLUX')
            hdul.append(data_hdu)

            # Create error extension
            err_hdu = fits.ImageHDU(name='ERR')
            # Placeholder for error data (to be populated later)
            err_hdu.data = np.zeros_like(spectra)
            hdul.append(err_hdu)
            
            # Create FITSTABLE extension for metadata
            cols = []
            # Add placeholder columns for metadata (to be populated later)
            cols.append(fits.Column(name='OBJID', format='K', array=np.arange(len(extraction_objects))))
            table_hdu = fits.BinTableHDU.from_columns(cols, name='FITSTABLE')
            hdul.append(table_hdu)
            
            # Create wavelength extension for each extraction object
            for i, obj in enumerate(extraction_objects):
                wave_hdu = fits.ImageHDU(data=obj.xshift, name=f'WAVE_{i}')
                hdul.append(wave_hdu)

        # Write to output file
        hdul.writeto(output_file, overwrite=True)
        
        
        
        return