import cloudpickle as pickle
import os
from astropy.io import fits
import numpy as np
import re
from astropy.coordinates import SkyCoord
from astropy import units as u


class RSSgeneration:
    def __init__(self):
        return


    def generate_rss(self, extraction_file, output_file):
        """
        Generate a row-stacked spectra (RSS) FITS file using binary tables to store the data.
        Each fiber will have its own row in the table with columns for flux, error, wavelength, etc.
        
        Parameters:
        -----------
        extraction_file : str
            Path to the extraction pickle file
        output_file : str
            Path to save the output RSS FITS file
        """
        with open(extraction_file, 'rb') as f:
            _data = pickle.load(f)
        primary_hdr = _data['primary_header']    
        extraction_objects = _data['extractions']
        _metadata = _data['metadata']

        # Group by channel
        channel_groups = {}
        meta_groups = {}
        for obj, meta in zip(extraction_objects, _metadata):
            channel = meta.get('channel', 'UNKNOWN')
            if channel not in channel_groups:
                channel_groups[channel] = []
                meta_groups[channel] = []
            channel_groups[channel].append(obj)
            meta_groups[channel].append(meta)

        # Create HDU list with primary header
        hdul = fits.HDUList()
        primary_hdu = fits.PrimaryHDU(header=primary_hdr)
        hdul.append(primary_hdu)

        # Process each channel separately
        for channel, obj_list in channel_groups.items():
            meta_list = meta_groups[channel]
            
            # Collect data for all fibers in this channel
            all_flux = []
            all_errors = []
            all_waves = []
            all_dq = []
            fiber_ids = []
            benchsides = []
            extnums = []
            
            # Process each extraction object
            for i, (obj, meta) in enumerate(zip(obj_list, meta_list)):
                counts = obj.counts
                n_fibers = counts.shape[0]
                
                # Get or create error arrays
                errors = getattr(obj, 'errors', None)
                if errors is None or errors.shape != counts.shape:
                    errors = np.zeros_like(counts, dtype=np.float32)
                    
                # Get or create data quality arrays
                dq = getattr(obj, 'dq', None)
                if dq is None or dq.shape != counts.shape:
                    dq = np.zeros_like(counts, dtype=np.int16)
                    
                # Get wavelength arrays
                waves = getattr(obj, 'wave', None)
                if waves is None or waves.shape != counts.shape:
                    # If no valid wavelength data, use NaN arrays
                    print(f"Warning: No valid wavelength data for object {i}. Using NaN arrays.")
                    waves = np.full(counts.shape, np.nan, dtype=np.float32)
                
                # Add data for each fiber
                all_flux.append(counts)
                all_errors.append(errors)
                all_waves.append(waves)
                all_dq.append(dq)
                
                # Add metadata
                benchside_str = f"{meta.get('bench', '')}{meta.get('side', '')}"
                benchsides.extend([benchside_str] * n_fibers)
                fiber_ids.extend(np.arange(n_fibers))
                extnums.extend([i] * n_fibers)
            
            # Stack all arrays
            flux_stack = np.vstack(all_flux)
            error_stack = np.vstack(all_errors)
            wave_stack = np.vstack(all_waves)
            dq_stack = np.vstack(all_dq)
            
            # Get array dimensions
            n_total_fibers, n_pixels = flux_stack.shape
            
            # Create columns for binary table
            cols = [
                fits.Column(name='FIBER', format='J', array=np.array(fiber_ids)),
                fits.Column(name='BENCHSIDE', format='10A', array=np.array(benchsides)),
                fits.Column(name='EXTNUM', format='J', array=np.array(extnums)),
                fits.Column(name='FLUX', format=f'{n_pixels}E', array=flux_stack),
                fits.Column(name='ERROR', format=f'{n_pixels}E', array=error_stack),
                fits.Column(name='WAVELENGTH', format=f'{n_pixels}E', array=wave_stack),
                fits.Column(name='DQ', format=f'{n_pixels}J', array=dq_stack)
            ]
            
            # Create binary table HDU
            table_hdu = fits.BinTableHDU.from_columns(cols)
            table_hdu.header['EXTNAME'] = f'SPEC_{channel}'
            table_hdu.header['CHANNEL'] = channel
            table_hdu.header['NFIBERS'] = n_total_fibers
            table_hdu.header['NPIXELS'] = n_pixels
            
            # Add metadata about wavelength range if available
            try:
                valid_waves = wave_stack[~np.isnan(wave_stack)]
                if len(valid_waves) > 0:
                    min_wave = np.min(valid_waves)
                    max_wave = np.max(valid_waves)
                    table_hdu.header['WAVEMIN'] = min_wave
                    table_hdu.header['WAVEMAX'] = max_wave
                    table_hdu.header['COMMENT'] = f'Wavelength range: {min_wave:.2f}-{max_wave:.2f} Angstroms'
            except Exception as e:
                print(f"Warning: Could not determine wavelength range: {e}")
            
            # Add table to HDU list
            hdul.append(table_hdu)
            print(f"Added SPEC_{channel} binary table with {n_total_fibers} fibers, {n_pixels} pixels each")
        
        # Write to file
        hdul.writeto(output_file, overwrite=True)
        print(f"RSS file written to: {output_file}")
        
        return




#############
def update_ra_dec_in_fits(fits_file):
    with fits.open(fits_file, mode='update') as hdul:
        primary_hdr = hdul[0].header

        ra = primary_hdr.get('RA')
        dec = primary_hdr.get('DEC')

        missing_ra = (ra is None or str(ra).strip() == "")
        missing_dec = (dec is None or str(dec).strip() == "")

        if missing_ra or missing_dec:
            tel_ra = primary_hdr.get('HIERARCH TEL RA')
            tel_dec = primary_hdr.get('HIERARCH TEL DEC')

            if tel_ra is None or tel_dec is None:
                raise ValueError("Primary header missing HIERARCH TEL RA and/or HIERARCH TEL DEC")

            # Convert telescope coordinates from sexagesimal to decimal assuming both are in degrees.
            c = SkyCoord(ra=str(tel_ra), dec=str(tel_dec), unit=(u.deg, u.deg))

            if missing_ra:
                primary_hdr['RA'] = c.ra.deg
            if missing_dec:
                primary_hdr['DEC'] = c.dec.deg

        hdul.flush()
        print(f"Primary header updated in {fits_file}")
