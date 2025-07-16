import cloudpickle as pickle
import os
from astropy.io import fits
import numpy as np
import re
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging
from datetime import datetime
from llamas_pyjamas.Utils.utils import setup_logger


class RSSgeneration:
    def __init__(self, logger=None):
        """
        Initialize the RSSgeneration class.
        
        Parameters:
            logger (Logger, optional): Logger instance. If None, creates a new one.
        """
        # Set up logging
        if logger is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.logger = setup_logger(__name__, f'RSSgeneration_{timestamp}.log')
        else:
            self.logger = logger
            
        self.logger.info("RSSgeneration initialized successfully")
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
        self.logger.info(f"Generating RSS file from {extraction_file}")
        
        try:
            with open(extraction_file, 'rb') as f:
                _data = pickle.load(f)
            primary_hdr = _data['primary_header']    
            extraction_objects = _data['extractions']
            _metadata = _data['metadata']
            
            self.logger.info(f"Loaded extraction file with {len(extraction_objects)} extraction objects")

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
            
            self.logger.info(f"Found channels: {list(channel_groups.keys())}")

            # Create HDU list with primary header
            hdul = fits.HDUList()
            primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            hdul.append(primary_hdu)

            # Process each channel separately
            for channel, obj_list in channel_groups.items():
                meta_list = meta_groups[channel]
                self.logger.info(f"Processing channel {channel} with {len(obj_list)} extraction objects")
                
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
                    

                    # Get or create error arrays
                    errors = getattr(obj, 'errors', None)
                    if errors is None or errors.shape != counts.shape:
                        self.logger.warning(f"No valid error data for object {i}. Creating zero array.")
                        errors = np.zeros_like(counts, dtype=np.float32)
                        
                    # Get or create data quality arrays
                    dq = getattr(obj, 'dq', None)
                    if dq is None or dq.shape != counts.shape:
                        self.logger.warning(f"No valid DQ data for object {i}. Creating zero array.")
                        dq = np.zeros_like(counts, dtype=np.int16)
                        
                    # Get wavelength arrays
                    waves = getattr(obj, 'wave', None)
                    if waves is None or waves.shape != counts.shape:
                        self.logger.warning(f"No valid wavelength data for object {i}. Using NaN arrays.")
                        waves = np.full(counts.shape, np.nan, dtype=np.float32)

                    dead_fibers = getattr(obj, 'dead_fibers', None)
                    if dead_fibers is not None:
                        self.logger.info(f"Object {i} has dead fibers: {dead_fibers}")
                        # Validate each dead fiber
                        valid_dead_fibers = []
                        for dead_fiber in dead_fibers:
                            # Verify that the dead fiber index is valid
                            if 0 <= dead_fiber < n_fibers:
                                # Check if all values in the row are zero (or close to zero)
                                is_zero_row = np.allclose(counts[dead_fiber], 0, atol=1e-10)
                                if is_zero_row:
                                    valid_dead_fibers.append(dead_fiber)
                                else:
                                    self.logger.warning(f"Dead fiber {dead_fiber} in object {i} has non-zero values - not removing")
                            else:
                                self.logger.warning(f"Dead fiber index {dead_fiber} is out of range for object {i} with {n_fibers} fibers")
                        
                        # Remove the valid dead fibers
                        if valid_dead_fibers:
                            # Sort in descending order if there are multiple fibers to avoid index shifting
                            if len(valid_dead_fibers) > 1:
                                valid_dead_fibers.sort(reverse=True)
                            
                            self.logger.info(f"Removing confirmed dead fibers {valid_dead_fibers} from object {i}")
                            for dead_fiber in valid_dead_fibers:
                                # Double check as a sanity check that the row is all zeros
                                is_zero_row = np.allclose(counts[dead_fiber], 0, atol=1e-10)
                                self.logger.info(f"Sanity check for fiber {dead_fiber}: all zeros = {is_zero_row}")
                                if not is_zero_row:
                                    self.logger.warning(f"Skipping removal of fiber {dead_fiber} as it contains non-zero values")
                                    continue
                                    
                                # Remove the dead fiber from all arrays
                                counts = np.delete(counts, dead_fiber, axis=0)
                                errors = np.delete(errors, dead_fiber, axis=0)
                                waves = np.delete(waves, dead_fiber, axis=0)
                                dq = np.delete(dq, dead_fiber, axis=0)
                                
                            # Log the final shape after all removals
                            self.logger.info(f"New arrays shape after removal: {counts.shape}")

                    n_fibers = counts.shape[0]
                    # Get or create error arrays
                    errors = getattr(obj, 'errors', None)
                    if errors is None or errors.shape != counts.shape:
                        self.logger.warning(f"No valid error data for object {i}. Creating zero array.")
                        errors = np.zeros_like(counts, dtype=np.float32)
                        
                    # Get or create data quality arrays
                    dq = getattr(obj, 'dq', None)
                    if dq is None or dq.shape != counts.shape:
                        self.logger.warning(f"No valid DQ data for object {i}. Creating zero array.")
                        dq = np.zeros_like(counts, dtype=np.int16)
                        
                    # Get wavelength arrays
                    waves = getattr(obj, 'wave', None)
                    self.logger.info(f"Object {i} wavelength shape: {waves.shape if waves is not None else 'None'}, counts shape: {counts.shape}")
                    if waves is None:
                        self.logger.warning(f"No wavelength attribute found for object {i}. Metadata: {meta}")

                    if waves is None or waves.shape != counts.shape:
                        # If no valid wavelength data, use NaN arrays
                        self.logger.warning(f"No valid wavelength data for object {i}. Using NaN arrays.")
                        waves = np.full(counts.shape, np.nan, dtype=np.float32)
                    else:
                        # Log wavelength stats
                        nan_count = np.sum(np.isnan(waves))
                        nan_percentage = (nan_count / waves.size) * 100
                        self.logger.info(f"Wavelength array for object {i}: shape={waves.shape}, " +
                                        f"NaN count={nan_count} ({nan_percentage:.2f}%), " +
                                        f"valid range={np.nanmin(waves):.2f}-{np.nanmax(waves):.2f}")
                    
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
                self.logger.info(f"Channel {channel} stacked data shape: {flux_stack.shape} ({n_total_fibers} fibers, {n_pixels} pixels)")
                # Log wavelength stack information
                self.logger.info(f"Channel {channel} wavelength stack shape: {wave_stack.shape}")
                self.logger.info(f"Flux stack min/max: {np.nanmin(flux_stack):.4f} / {np.nanmax(flux_stack):.4f}")
                self.logger.info(f"Wavelength stack min/max: {np.nanmin(wave_stack):.4f} / {np.nanmax(wave_stack):.4f}")
                self.logger.info(f"Error stack min/max: {np.nanmin(error_stack):.4f} / {np.nanmax(error_stack):.4f}")
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
                    valid_count = len(valid_waves)
                    total_count = wave_stack.size
                    valid_percentage = (valid_count / total_count) * 100;
                    
                    self.logger.info(f"Channel {channel} has {valid_count} valid wavelength points " +
                                    f"out of {total_count} total points ({valid_percentage:.2f}%)")
                    
                    if valid_count > 0:
                        min_wave = np.min(valid_waves)
                        max_wave = np.max(valid_waves)
                        table_hdu.header['WAVEMIN'] = min_wave
                        table_hdu.header['WAVEMAX'] = max_wave
                        table_hdu.header['COMMENT'] = f'Wavelength range: {min_wave:.2f}-{max_wave:.2f} Angstroms'
                        self.logger.info(f"Channel {channel} wavelength range: {min_wave:.2f}-{max_wave:.2f} Angstroms")
                except Exception as e:
                    self.logger.error(f"Could not determine wavelength range: {str(e)}")
                
                # Add table to HDU list
                hdul.append(table_hdu)
                self.logger.info(f"Added SPEC_{channel} binary table with {n_total_fibers} fibers, {n_pixels} pixels each")
            
            # Write to file
            hdul.writeto(output_file, overwrite=True)
            self.logger.info(f"RSS file written to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating RSS file: {str(e)}", exc_info=True)
            raise
        
        return




#############
def update_ra_dec_in_fits(fits_file, logger=None):
    """
    Update RA and DEC values in a FITS file header using HIERARCH TEL values if needed.
    
    Parameters:
    -----------
    fits_file : str
        Path to the FITS file to update
    logger : Logger, optional
        Logger instance. If None, creates a new one.
    """
    # Set up logging if not provided
    if logger is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger = setup_logger(__name__, f'update_ra_dec_{timestamp}.log')
    
    logger.info(f"Updating RA/DEC in FITS file: {fits_file}")
    
    try:
        with fits.open(fits_file, mode='update') as hdul:
            primary_hdr = hdul[0].header

            ra = primary_hdr.get('RA')
            dec = primary_hdr.get('DEC')

            missing_ra = (ra is None or str(ra).strip() == "")
            missing_dec = (dec is None or str(dec).strip() == "")

            if missing_ra or missing_dec:
                logger.info(f"Missing RA={missing_ra} or DEC={missing_dec}, attempting to use HIERARCH TEL values")
                tel_ra = primary_hdr.get('HIERARCH TEL RA')
                tel_dec = primary_hdr.get('HIERARCH TEL DEC')

                if tel_ra is None or tel_dec is None:
                    error_msg = "Primary header missing HIERARCH TEL RA and/or HIERARCH TEL DEC"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Convert telescope coordinates from sexagesimal to decimal assuming both are in degrees.
                c = SkyCoord(ra=str(tel_ra), dec=str(tel_dec), unit=(u.deg, u.deg))
                
                logger.info(f"Converted HIERARCH TEL coordinates: RA={c.ra.deg}, DEC={c.dec.deg}")

                if missing_ra:
                    primary_hdr['RA'] = c.ra.deg
                    logger.info(f"Updated RA value to {c.ra.deg}")
                if missing_dec:
                    primary_hdr['DEC'] = c.dec.deg
                    logger.info(f"Updated DEC value to {c.dec.deg}")

            hdul.flush()
            logger.info(f"Primary header updated in {fits_file}")
    except Exception as e:
        logger.error(f"Error updating RA/DEC in FITS file: {str(e)}", exc_info=True)
        raise
