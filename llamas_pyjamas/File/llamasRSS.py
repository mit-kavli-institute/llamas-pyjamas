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
        primary_hdr = _data['primary_header']    
        extraction_objects = _data['extractions']
        _metadata = _data['metadata']
        
        # Create a new HDU list for the RSS FITS file
        hdul = fits.HDUList()

        # Primary HDU with the primary header from the extraction file
        primary_hdu = fits.PrimaryHDU(header=primary_hdr)
        hdul.append(primary_hdu)
        
        # Create separate SCI, ERR, and DQ extensions for each extraction object
        if extraction_objects:
            print(f"Creating {len(extraction_objects)} sets of extensions")
            
            for i, (obj, meta) in enumerate(zip(extraction_objects, _metadata)):
                counts = obj.counts
                print(f"Processing extraction {i}: {counts.shape}")
                
                # SCI extension for this extraction
                sci_hdu = fits.ImageHDU(data=counts.astype(np.float32), name='SCI')
                for k, v in meta.items():
                    sci_hdu.header[k.upper()] = v
                sci_hdu.header['EXTNAME'] = 'SCI'
                sci_hdu.header['EXTVER'] = i+1
                hdul.append(sci_hdu)
                
                # ERR extension for this extraction
                errors = getattr(obj, 'errors', None)
                if errors is None or errors.shape != counts.shape:
                    errors = np.zeros_like(counts, dtype=np.float32)
                err_hdu = fits.ImageHDU(data=errors.astype(np.float32), name='ERR')
                for k, v in meta.items():
                    err_hdu.header[k.upper()] = v
                err_hdu.header['EXTNAME'] = 'ERR'
                err_hdu.header['EXTVER'] = i+1
                hdul.append(err_hdu)
                
                # DQ extension for this extraction
                dq = getattr(obj, 'dq', None)
                if dq is None or dq.shape != counts.shape:
                    dq = np.zeros_like(counts, dtype=np.int16)
                dq_hdu = fits.ImageHDU(data=dq.astype(np.int16), name='DQ')
                for k, v in meta.items():
                    dq_hdu.header[k.upper()] = v
                dq_hdu.header['EXTNAME'] = 'DQ'
                dq_hdu.header['EXTVER'] = i+1
                hdul.append(dq_hdu)

        # Write to output file
        hdul.writeto(output_file, overwrite=True)
        print(f"RSS file written to: {output_file}")
        
        return
