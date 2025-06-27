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

        # Create separate IMAGE, ERR, and FITSTABLE extensions for each extraction object
        if extraction_objects:
            print(f"Creating {len(extraction_objects)} sets of extensions")
            
            for i, (obj, meta) in enumerate(zip(extraction_objects, _metadata)):
                counts = obj.counts
                print(f"Processing extraction {i}: {counts.shape}")
                
                # IMAGE extension for this extraction
                image_name = f'IMAGE_{i:02d}'
                data_hdu = fits.ImageHDU(data=counts.astype(np.float32), name=image_name)
                # Add metadata to header
                data_hdu.header['BENCH'] = meta.get('bench', '')
                data_hdu.header['SIDE'] = meta.get('side', '')
                data_hdu.header['CHANNEL'] = meta.get('channel', '')
                data_hdu.header['NFIBERS'] = meta.get('nfibers', counts.shape[0])
                data_hdu.header['EXTNUM'] = i
                hdul.append(data_hdu)
                
                # ERR extension for this extraction (initialize with zeros for now)
                err_name = f'ERR_{i:02d}'
                err_hdu = fits.ImageHDU(data=np.zeros_like(counts, dtype=np.float32), name=err_name)
                err_hdu.header['BENCH'] = meta.get('bench', '')
                err_hdu.header['SIDE'] = meta.get('side', '')
                err_hdu.header['CHANNEL'] = meta.get('channel', '')
                err_hdu.header['NFIBERS'] = meta.get('nfibers', counts.shape[0])
                err_hdu.header['EXTNUM'] = i
                hdul.append(err_hdu)
                
                # FITSTABLE extension for this extraction
                table_name = f'FITSTABLE_{i:02d}'
                n_fibers = counts.shape[0]
                
                # Create columns for fiber information
                cols = []
                cols.append(fits.Column(name='FIBER', format='K', array=np.arange(n_fibers)))
                cols.append(fits.Column(name='BENCH', format='A10', array=[meta.get('bench', '')] * n_fibers))
                cols.append(fits.Column(name='SIDE', format='A10', array=[meta.get('side', '')] * n_fibers))
                cols.append(fits.Column(name='CHANNEL', format='A10', array=[meta.get('channel', '')] * n_fibers))
                cols.append(fits.Column(name='EXTNUM', format='K', array=[i] * n_fibers))
                
                table_hdu = fits.BinTableHDU.from_columns(cols, name=table_name)
                table_hdu.header['BENCH'] = meta.get('bench', '')
                table_hdu.header['SIDE'] = meta.get('side', '')
                table_hdu.header['CHANNEL'] = meta.get('channel', '')
                table_hdu.header['NFIBERS'] = n_fibers
                table_hdu.header['EXTNUM'] = i
                hdul.append(table_hdu)

        # Write to output file
        hdul.writeto(output_file, overwrite=True)
        print(f"RSS file written to: {output_file}")
        
        return
