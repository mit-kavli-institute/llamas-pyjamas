import cloudpickle as pickle
import os
from astropy.io import fits
import numpy as np
import re
from astropy.coordinates import SkyCoord
from astropy import units as u
import logging
from datetime import datetime


class RSSgeneration:
    def __init__(self, logger=None):
        """
        Initialize the RSSgeneration class.
        
        Parameters:
            logger (Logger, optional): Logger instance. If None, creates a new one.
        """
        # Set up logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.logger.info("RSSgeneration initialized successfully")
        return


    def generate_rss(self, extraction_file, output_file, subtract_sky=True, noflat_file=None):
        """
        Generate a row-stacked spectra (RSS) FITS file with the following structure:
        Extension 0 - PRIMARY: primary header only, no data
        Extension 1 - FLUX: sky-subtracted flux (counts - sky) [NFIBER x NWAVE]
                      If subtract_sky=False or no sky model, same as COUNTS.
        Extension 2 - NOFLAT (optional): like FLUX but without 2D pixel flat, only throughput-corrected.
                      Present only when noflat_file is provided.
        Extension 3 - COUNTS: raw extracted counts prior to sky subtraction [NFIBER x NWAVE]
        Extension 4 - SKY: sky model for each fiber [NFIBER x NWAVE]
        Extension 5 - MASK: the pixel mask array for each fiber [NFIBER x NWAVE]
        Extension 6 - WAVE: the wavelength array for each fiber [NFIBER x NWAVE]
        Extension 7 - FWHM: the full width half max array for each fiber [NFIBER x NWAVE]
        Extension 8 - FIBERMAP: the complete fibermap [BINARY FITS TABLE]

        Note: Each row in the data extensions represents one fiber's full spectrum across all wavelengths.
              Access fiber N's data as: data[N, :] (row N, all columns)

        Parameters:
        -----------
        extraction_file : str
            Path to the extraction pickle file
        output_file : str
            Path to save the output RSS FITS file
        subtract_sky : bool, optional
            If True (default), subtract the sky model from counts in the FLUX extension.
            The sky model is always saved in the SKY extension regardless of this setting.
            If the extraction has no sky model, FLUX is unchanged and SKY is all zeros.
        noflat_file : str, optional
            Path to a second extraction pickle produced from the un-flat-corrected science
            FITS.  When provided a NOFLAT extension is inserted between FLUX and COUNTS
            containing (noflat_counts - sky) / throughput — i.e. sky-subtracted and
            throughput-corrected but without the 2D pixel flat divided through.
        """
        self.logger.info(f"Generating RSS file from {extraction_file}")

        self.new_rss_filenames = []

        # Optionally load a parallel extraction from the un-flat-corrected FITS for NOFLAT ext.
        noflat_objects = None
        noflat_meta = None
        if noflat_file is not None:
            try:
                with open(noflat_file, 'rb') as f:
                    _nf = pickle.load(f)
                noflat_objects = _nf['extractions']
                noflat_meta    = _nf['metadata']
                self.logger.info(f"Loaded NOFLAT extraction from {noflat_file} ({len(noflat_objects)} objects)")
            except Exception as e:
                self.logger.warning(f"Could not load noflat_file {noflat_file}: {e}. NOFLAT extension will be skipped.")
                noflat_objects = None

        try:
            with open(extraction_file, 'rb') as f:
                _data = pickle.load(f)
            primary_hdr = _data.get('primary_header')
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

            # Group noflat objects by channel (if provided)
            noflat_channel_groups = {}
            if noflat_objects is not None and noflat_meta is not None:
                for obj, meta in zip(noflat_objects, noflat_meta):
                    ch = meta.get('channel', 'UNKNOWN')
                    if ch not in noflat_channel_groups:
                        noflat_channel_groups[ch] = []
                    noflat_channel_groups[ch].append(obj)

            # Process each channel separately - we'll create one RSS file per channel
            for channel, obj_list in channel_groups.items():
                meta_list = meta_groups[channel]
                self.logger.info(f"Processing channel {channel} with {len(obj_list)} extraction objects")
                
                channel_output_file = output_file
                if len(channel_groups) > 1:
                    # If multiple channels, append channel name to output file
                    base, ext = os.path.splitext(output_file)
                    channel_output_file = f"{base}_{channel}{ext}"
                    self.new_rss_filenames.append(channel_output_file)
                    self.logger.info(f"Creating channel-specific output file: {channel_output_file}")
                
                # Create HDU list with primary header (Extension 0 - PRIMARY)
                hdul = fits.HDUList()
                primary_hdu = fits.PrimaryHDU(header=primary_hdr)
                primary_hdu.header['CHANNEL'] = channel
                hdul.append(primary_hdu)
                
                # Collect data for all fibers in this channel
                all_flux = []
                all_errors = []
                all_waves = []
                all_dq = []
                all_fwhm = []
                all_sky = []
                fiber_ids = []
                benchsides = []
                fiber_types = []
                fiber_ras = []
                fiber_decs = []
                
                # Process each extraction object
                for i, (obj, meta) in enumerate(zip(obj_list, meta_list)):
                    # Get the counts (flux)
                    counts = obj.counts
                    n_fibers = counts.shape[0]
                    original_n_fibers = n_fibers  # Track original count before any removals

                    # Get relative throughput (1D, one scalar per fiber); fall back to 1.0
                    throughput = getattr(obj, 'relative_throughput', None)
                    if throughput is None or len(throughput) != n_fibers:
                        self.logger.warning(f"Object {i}: missing or mismatched relative_throughput — no throughput correction applied")
                        throughput = np.ones(n_fibers, dtype=np.float32)
                    else:
                        throughput = np.array(throughput, dtype=np.float32)
                        bad_tp = ~np.isfinite(throughput) | (throughput <= 0)
                        if np.any(bad_tp):
                            self.logger.warning(f"Object {i}: {bad_tp.sum()} fibers have invalid throughput values — setting those to 1.0")
                            throughput[bad_tp] = 1.0
                    
                    # Get or create error arrays
                    errors = getattr(obj, 'errors', None)
                    if errors is None or errors.shape != counts.shape:
                        self.logger.warning(f"No valid error data for object {i}. Creating zero array.")
                        errors = np.zeros_like(counts, dtype=np.float32)
                    
                    # Get or create data quality (mask) arrays
                    dq = getattr(obj, 'dq', None)
                    if dq is None or dq.shape != counts.shape:
                        self.logger.warning(f"No valid DQ data for object {i}. Creating zero array.")
                        dq = np.zeros_like(counts, dtype=np.int16)
                    
                    # Get wavelength arrays with improved shape handling
                    waves = getattr(obj, 'wave', None)
                    if waves is None:
                        self.logger.warning(f"No wavelength attribute for object {i}. Using NaN arrays.")
                        waves = np.full(counts.shape, np.nan, dtype=np.float32)
                    elif waves.shape != counts.shape:
                        # Shape mismatch might be due to dead fiber insertion during extraction
                        self.logger.warning(f"Object {i} wavelength shape {waves.shape} != counts shape {counts.shape}")

                        # Check if wavelength array is smaller (missing dead fiber rows)
                        if waves.shape[0] < counts.shape[0] and waves.shape[1] == counts.shape[1]:
                            # Wavelength array is missing dead fiber rows - need to insert them
                            n_missing = counts.shape[0] - waves.shape[0]
                            self.logger.info(f"Wavelength array missing {n_missing} rows. This may be due to dead fiber insertion.")

                            # Get dead fibers that should have been inserted
                            dead_fibers_list = getattr(obj, 'dead_fibers', None)
                            if dead_fibers_list and len(dead_fibers_list) == n_missing:
                                # Insert NaN rows at dead fiber positions
                                self.logger.info(f"Inserting NaN rows at dead fiber positions: {dead_fibers_list}")
                                for dead_idx in sorted(dead_fibers_list):
                                    nan_row = np.full((1, waves.shape[1]), np.nan, dtype=np.float32)
                                    waves = np.insert(waves, dead_idx, nan_row, axis=0)
                                self.logger.info(f"After insertion, wavelength shape: {waves.shape}")
                            else:
                                self.logger.warning(f"Cannot reconcile shape mismatch. Using NaN arrays.")
                                waves = np.full(counts.shape, np.nan, dtype=np.float32)
                        else:
                            # Other shape mismatch - use NaN arrays
                            self.logger.warning(f"Incompatible wavelength shape. Using NaN arrays.")
                            waves = np.full(counts.shape, np.nan, dtype=np.float32)

                    # Final validation
                    if waves.shape != counts.shape:
                        self.logger.error(f"Object {i} wavelength shape {waves.shape} still != counts shape {counts.shape}. Using NaN arrays.")
                        waves = np.full(counts.shape, np.nan, dtype=np.float32)
                    
                    # Get or create FWHM arrays (may not exist in all extractions)
                    fwhm = getattr(obj, 'fwhm', None)
                    if fwhm is None or fwhm.shape != counts.shape:
                        self.logger.warning(f"No valid FWHM data for object {i}. Creating default array.")
                        fwhm = np.full(counts.shape, 2.5, dtype=np.float32)  # Default FWHM of 2.5 pixels

                    # Get sky model array (populated by skyModel_1d; zeros if not available)
                    sky = getattr(obj, 'sky', None)
                    if sky is None:
                        sky = np.zeros_like(counts, dtype=np.float32)
                    elif sky.shape != counts.shape:
                        # Sky may be missing dead-fiber rows inserted into counts during extraction
                        if sky.shape[0] < counts.shape[0] and sky.shape[1] == counts.shape[1]:
                            n_missing = counts.shape[0] - sky.shape[0]
                            dead_fibers_list = getattr(obj, 'dead_fibers', None)
                            if dead_fibers_list and len(dead_fibers_list) == n_missing:
                                self.logger.info(f"Inserting {n_missing} zero rows into sky at dead fiber positions: {dead_fibers_list}")
                                for dead_idx in sorted(dead_fibers_list):
                                    zero_row = np.zeros((1, sky.shape[1]), dtype=np.float32)
                                    sky = np.insert(sky, dead_idx, zero_row, axis=0)
                            else:
                                self.logger.warning(f"Sky shape {sky.shape} != counts shape {counts.shape} for object {i}, cannot reconcile. Using zeros.")
                                sky = np.zeros_like(counts, dtype=np.float32)
                        else:
                            self.logger.warning(f"Sky shape {sky.shape} != counts shape {counts.shape} for object {i}. Using zeros.")
                            sky = np.zeros_like(counts, dtype=np.float32)

                    # Handle dead fibers
                    dead_fibers = getattr(obj, 'dead_fibers', None)
                    removed_dead_fibers = []  # Track which fibers were actually removed
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
                                    self.logger.info(f"Validated dead fiber {dead_fiber}: all zeros = True")
                                else:
                                    self.logger.warning(f"Dead fiber {dead_fiber} in object {i} has non-zero values - not removing")
                            else:
                                self.logger.warning(f"Dead fiber index {dead_fiber} is out of range for object {i} with {n_fibers} fibers")

                        # Remove the valid dead fibers using boolean mask to avoid index shifting issues
                        if valid_dead_fibers:
                            self.logger.info(f"Removing confirmed dead fibers {valid_dead_fibers} from object {i}")

                            # Log wavelength stats BEFORE removal for debugging
                            self.logger.info(f"BEFORE dead fiber removal - Wavelength shape: {waves.shape}, " +
                                           f"valid range: {np.nanmin(waves):.2f}-{np.nanmax(waves):.2f}")

                            # Create boolean mask: True for fibers to KEEP, False for fibers to REMOVE
                            keep_mask = np.ones(n_fibers, dtype=bool)
                            keep_mask[valid_dead_fibers] = False

                            # Apply mask to all arrays simultaneously
                            counts = counts[keep_mask]
                            errors = errors[keep_mask]
                            waves = waves[keep_mask]
                            dq = dq[keep_mask]
                            fwhm = fwhm[keep_mask]
                            sky = sky[keep_mask]
                            throughput = throughput[keep_mask]

                            # Track which fibers were removed for fiber ID generation
                            removed_dead_fibers = valid_dead_fibers

                            # Log the final shape and wavelength stats after removal
                            self.logger.info(f"New arrays shape after removal: {counts.shape}")
                            self.logger.info(f"AFTER dead fiber removal - Wavelength shape: {waves.shape}, " +
                                           f"valid range: {np.nanmin(waves):.2f}-{np.nanmax(waves):.2f}")
                            n_fibers = counts.shape[0]  # Update fiber count
                    
                    # Enhanced wavelength validation logging
                    if waves is not None:
                        nan_count = np.sum(np.isnan(waves))
                        zero_count = np.sum(waves == 0)
                        valid_mask = ~np.isnan(waves) & (waves != 0)
                        valid_count = np.sum(valid_mask)

                        nan_percentage = (nan_count / waves.size) * 100
                        valid_percentage = (valid_count / waves.size) * 100

                        self.logger.info(f"Wavelength array for object {i}:")
                        self.logger.info(f"  Shape: {waves.shape}")
                        self.logger.info(f"  NaN count: {nan_count} ({nan_percentage:.2f}%)")
                        self.logger.info(f"  Zero count: {zero_count} ({100*zero_count/waves.size:.2f}%)")
                        self.logger.info(f"  Valid count: {valid_count} ({valid_percentage:.2f}%)")

                        if valid_count > 0:
                            self.logger.info(f"  Valid range: {waves[valid_mask].min():.2f}-{waves[valid_mask].max():.2f} Å")

                            # Check for wavelength inversions per fiber
                            n_inversions = 0
                            for fiber_idx in range(waves.shape[0]):
                                fiber_wave = waves[fiber_idx]
                                diffs = np.diff(fiber_wave)
                                n_backwards = np.sum(diffs < 0)
                                if n_backwards > 10:  # More than 10 backwards steps is problematic
                                    n_inversions += 1
                                    self.logger.warning(f"  Fiber {fiber_idx} has {n_backwards} wavelength inversions!")

                            if n_inversions > 0:
                                self.logger.warning(f"  Total fibers with inversions: {n_inversions}/{waves.shape[0]}")
                            else:
                                self.logger.info(f"  No wavelength inversions detected")
                        else:
                            self.logger.error(f"  ERROR: No valid wavelength data!")
                    
                    # Apply relative throughput correction — divides out fiber-to-fiber
                    # sensitivity variations so all extensions are on the same scale.
                    # Broadcast (n_fibers,) → (n_fibers, n_pixels) with [:, np.newaxis].
                    tp = throughput[:, np.newaxis]
                    counts = counts / tp
                    sky    = sky    / tp
                    errors = errors / tp

                    # Add data for each fiber
                    all_flux.append(counts)
                    all_errors.append(errors)
                    all_waves.append(waves)
                    all_dq.append(dq)
                    all_fwhm.append(fwhm)
                    all_sky.append(sky)
                    
                    # Add metadata for fiber map
                    benchside_str = f"{meta.get('bench', '')}{meta.get('side', '')}"
                    fiber_type = meta.get('fiber_type', ['UNKNOWN'] * n_fibers)
                    fiber_ra = meta.get('fiber_ra', [np.nan] * n_fibers)
                    fiber_dec = meta.get('fiber_dec', [np.nan] * n_fibers)

                    benchsides.extend([benchside_str] * n_fibers)

                    # Generate fiber IDs that preserve original fiber indices (skip dead fibers)
                    if removed_dead_fibers:
                        # Build list of alive fiber IDs by skipping dead ones
                        fibers = []
                        counter = 0
                        while len(fibers) < n_fibers:
                            if counter in removed_dead_fibers:
                                counter += 1
                            else:
                                fibers.append(counter)
                                counter += 1
                        fiber_ids.extend(fibers)
                        self.logger.info(f"Object {i}: Generated fiber IDs {fibers} (skipped dead fibers {removed_dead_fibers})")
                    else:
                        # No dead fibers removed, use sequential IDs
                        fiber_ids.extend(np.arange(n_fibers))

                    fiber_types.extend(fiber_type if len(fiber_type) == n_fibers else ['UNKNOWN'] * n_fibers)
                    fiber_ras.extend(fiber_ra if len(fiber_ra) == n_fibers else [np.nan] * n_fibers)
                    fiber_decs.extend(fiber_dec if len(fiber_dec) == n_fibers else [np.nan] * n_fibers)
                
                # Stack all arrays
                counts_stack = np.vstack(all_flux)   # raw counts, always preserved
                error_stack = np.vstack(all_errors)
                wave_stack = np.vstack(all_waves)
                dq_stack = np.vstack(all_dq)
                fwhm_stack = np.vstack(all_fwhm)
                sky_stack = np.vstack(all_sky)

                # FLUX = counts - sky if subtract_sky=True and sky model is non-zero
                has_sky = np.any(sky_stack != 0)
                if subtract_sky and has_sky:
                    self.logger.info(f"Subtracting sky model from FLUX for channel {channel}")
                    flux_stack = counts_stack - sky_stack
                else:
                    flux_stack = counts_stack
                    if subtract_sky and not has_sky:
                        self.logger.warning(f"subtract_sky=True but sky model is all zeros for channel {channel}. FLUX unchanged.")
                
                # Get array dimensions
                n_total_fibers, n_pixels = flux_stack.shape
                self.logger.info(f"Channel {channel} stacked data shape: {flux_stack.shape} ({n_total_fibers} fibers, {n_pixels} pixels)")
                
                # Log some statistics
                self.logger.info(f"Counts stack min/max: {np.nanmin(counts_stack):.4f} / {np.nanmax(counts_stack):.4f}")
                self.logger.info(f"Wavelength stack min/max: {np.nanmin(wave_stack):.4f} / {np.nanmax(wave_stack):.4f}")
                self.logger.info(f"Error stack min/max: {np.nanmin(error_stack):.4f} / {np.nanmax(error_stack):.4f}")
                
                # Create common header for data extensions
                common_header = fits.Header()
                common_header['CHANNEL'] = channel
                common_header['NFIBERS'] = n_total_fibers
                common_header['NWAVE'] = n_pixels
                
                # Try to determine wavelength range if available
                try:
                    valid_waves = wave_stack[~np.isnan(wave_stack)]
                    valid_count = len(valid_waves)
                    total_count = wave_stack.size
                    valid_percentage = (valid_count / total_count) * 100
                    
                    self.logger.info(f"Channel {channel} has {valid_count} valid wavelength points " +
                                    f"out of {total_count} total points ({valid_percentage:.2f}%)")
                    
                    if valid_count > 0:
                        min_wave = np.min(valid_waves)
                        max_wave = np.max(valid_waves)
                        common_header['WAVEMIN'] = min_wave
                        common_header['WAVEMAX'] = max_wave
                        common_header['COMMENT'] = f'Wavelength range: {min_wave:.2f}-{max_wave:.2f} Angstroms'
                        self.logger.info(f"Channel {channel} wavelength range: {min_wave:.2f}-{max_wave:.2f} Angstroms")
                except Exception as e:
                    self.logger.error(f"Could not determine wavelength range: {str(e)}")
                
                # Extension 1 - FLUX: sky-subtracted flux [NFIBER x NWAVE]
                # FLUX = COUNTS - SKY when subtract_sky=True and sky model is present
                flux_hdu = fits.ImageHDU(flux_stack, header=common_header)
                flux_hdu.header['EXTNAME'] = 'FLUX'
                flux_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                flux_hdu.header['SKYSUB'] = subtract_sky and has_sky
                hdul.append(flux_hdu)
                self.logger.info(f"Added FLUX extension with shape {flux_stack.shape}")

                # Extension 2 - ERROR: the error array [NFIBER x NWAVE]
                error_hdu = fits.ImageHDU(error_stack, header=common_header)
                error_hdu.header['EXTNAME'] = 'ERROR'
                error_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                hdul.append(error_hdu)
                self.logger.info(f"Added ERROR extension with shape {error_stack.shape}")

                # Extension 3 (optional) - NOFLAT: sky-subtracted, throughput-corrected,
                # but extracted from the un-flat-corrected FITS (no 2D pixel flat applied).
                if noflat_channel_groups.get(channel):
                    nf_obj_list = noflat_channel_groups[channel]
                    all_nf = []
                    for nf_obj in nf_obj_list:
                        nf_counts = nf_obj.counts.copy().astype(np.float32)
                        n_nf = nf_counts.shape[0]
                        nf_tp = getattr(nf_obj, 'relative_throughput', None)
                        if nf_tp is None or len(nf_tp) != n_nf:
                            nf_tp = np.ones(n_nf, dtype=np.float32)
                        else:
                            nf_tp = np.array(nf_tp, dtype=np.float32)
                            bad = ~np.isfinite(nf_tp) | (nf_tp <= 0)
                            nf_tp[bad] = 1.0
                        nf_sky = getattr(nf_obj, 'sky', None)
                        if nf_sky is None or nf_sky.shape != nf_counts.shape:
                            nf_sky = np.zeros_like(nf_counts)

                        # Remove dead fibers to match the row layout of the main FLUX extension
                        dead_fibers = getattr(nf_obj, 'dead_fibers', None)
                        if dead_fibers:
                            valid_dead = [d for d in dead_fibers
                                          if 0 <= d < n_nf and np.allclose(nf_counts[d], 0, atol=1e-10)]
                            if valid_dead:
                                keep = np.ones(n_nf, dtype=bool)
                                keep[valid_dead] = False
                                nf_counts = nf_counts[keep]
                                nf_sky    = nf_sky[keep]
                                nf_tp     = nf_tp[keep]

                        nf_tp2d = nf_tp[:, np.newaxis]
                        nf_flux = (nf_counts - nf_sky) / nf_tp2d
                        all_nf.append(nf_flux)
                    if all_nf:
                        noflat_stack = np.vstack(all_nf)
                        # Trim/pad rows to match flux_stack if fiber counts differ
                        n_main = flux_stack.shape[0]
                        if noflat_stack.shape[0] > n_main:
                            noflat_stack = noflat_stack[:n_main, :]
                        elif noflat_stack.shape[0] < n_main:
                            pad = np.full((n_main - noflat_stack.shape[0], noflat_stack.shape[1]),
                                         np.nan, dtype=np.float32)
                            noflat_stack = np.vstack([noflat_stack, pad])
                        noflat_hdu = fits.ImageHDU(noflat_stack, header=common_header)
                        noflat_hdu.header['EXTNAME'] = 'NOFLAT'
                        noflat_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                        noflat_hdu.header['SKYSUB'] = subtract_sky and has_sky
                        noflat_hdu.header['COMMENT'] = 'Sky-subtracted, throughput-corrected; NO 2D pixel flat applied'
                        hdul.append(noflat_hdu)
                        self.logger.info(f"Added NOFLAT extension with shape {noflat_stack.shape}")

                # Extension 3 - COUNTS: raw extracted counts before sky subtraction [NFIBER x NWAVE]
                counts_hdu = fits.ImageHDU(counts_stack, header=common_header)
                counts_hdu.header['EXTNAME'] = 'COUNTS'
                counts_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                counts_hdu.header['COMMENT'] = 'Raw extracted counts prior to sky subtraction'
                hdul.append(counts_hdu)
                self.logger.info(f"Added COUNTS extension with shape {counts_hdu.data.shape}")

                # Extension 3 - SKY: the sky model for each fiber [NFIBER x NWAVE]
                sky_hdu = fits.ImageHDU(sky_stack, header=common_header)
                sky_hdu.header['EXTNAME'] = 'SKY'
                sky_hdu.header['BUNIT'] = '10^(-17) erg/s/cm2/Ang/fiber'
                sky_hdu.header['SKYSUB'] = subtract_sky
                sky_hdu.header['COMMENT'] = 'Sky model per fiber. FLUX = COUNTS - SKY if SKYSUB=True.'
                hdul.append(sky_hdu)
                self.logger.info(f"Added SKY extension with shape {sky_stack.shape} (subtract_sky={subtract_sky}, has_sky={has_sky})")

                # Extension 4 - MASK: the pixel mask array [NFIBER x NWAVE]
                mask_hdu = fits.ImageHDU(dq_stack, header=common_header)
                mask_hdu.header['EXTNAME'] = 'MASK'
                mask_hdu.header['COMMENT'] = 'Pixel mask: 0=good, >0=various issues'
                hdul.append(mask_hdu)
                self.logger.info(f"Added MASK extension with shape {dq_stack.shape}")

                # Extension 5 - WAVE: the wavelength array for each fiber [NFIBER x NWAVE]
                wave_hdu = fits.ImageHDU(wave_stack, header=common_header)
                wave_hdu.header['EXTNAME'] = 'WAVE'
                wave_hdu.header['BUNIT'] = 'Angstrom'
                if valid_count > 0:
                    wave_hdu.header['WAVESTAT'] = 'GOOD'
                    valid_percentage = (valid_count / total_count) * 100
                    wave_hdu.header['COMMENT'] = f'Wavelength data: {valid_percentage:.2f}% valid points'
                else:
                    wave_hdu.header['WAVESTAT'] = 'POOR'
                    wave_hdu.header['COMMENT'] = 'Warning: Limited or no valid wavelength data'
                hdul.append(wave_hdu)
                self.logger.info(f"Added WAVE extension with shape {wave_stack.shape}")

                # Extension 6 - FWHM: the full width half max array [NFIBER x NWAVE]
                fwhm_hdu = fits.ImageHDU(fwhm_stack, header=common_header)
                fwhm_hdu.header['EXTNAME'] = 'FWHM'
                fwhm_hdu.header['BUNIT'] = 'Pixels'
                hdul.append(fwhm_hdu)
                self.logger.info(f"Added FWHM extension with shape {fwhm_stack.shape}")

                # Extension 7 - FIBERMAP: binary table with fiber information
                fibermap_cols = [
                    fits.Column(name='FIBER_ID', format='J', array=np.array(fiber_ids)),
                    fits.Column(name='BENCHSIDE', format='10A', array=np.array(benchsides)),
                    fits.Column(name='FIBER_TYPE', format='10A', array=np.array(fiber_types)),
                    fits.Column(name='RA', format='D', array=np.array(fiber_ras)),
                    fits.Column(name='DEC', format='D', array=np.array(fiber_decs))
                ]
                
                fibermap_hdu = fits.BinTableHDU.from_columns(fibermap_cols)
                fibermap_hdu.header['EXTNAME'] = 'FIBERMAP'
                hdul.append(fibermap_hdu)
                self.logger.info(f"Added FIBERMAP binary table with {len(fiber_ids)} entries")
                
                # Write to file
                hdul.writeto(channel_output_file, overwrite=True)
                self.logger.info(f"RSS file written to: {channel_output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating RSS file: {str(e)}", exc_info=True)
            raise
        
        return self.new_rss_filenames




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
        logger = logging.getLogger(__name__)
    
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
                    logger.warning("HIERARCH TEL RA and/or HIERARCH TEL DEC not found in primary header; "
                                   "leaving RA/DEC unchanged")
                    hdul.flush()
                    return

                # Convert telescope coordinates from sexagesimal to decimal.
                # TEL RA is in HH:MM:SS.s (hourangle), TEL DEC is in DD:MM:SS.s (degrees).
                c = SkyCoord(ra=str(tel_ra), dec=str(tel_dec), unit=(u.hourangle, u.deg))
                
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
