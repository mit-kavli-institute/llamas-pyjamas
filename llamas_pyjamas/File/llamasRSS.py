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

        hdul = fits.HDUList()
        primary_hdu = fits.PrimaryHDU(header=primary_hdr)
        hdul.append(primary_hdu)

        for channel, obj_list in channel_groups.items():
            meta_list = meta_groups[channel]
            # Stack all counts, errors, dq for this channel
            flux_list = []
            err_list = []
            dq_list = []
            fiber_ids = []
            channels = []
            extnums = []
            benchsides = []

            # Get wavelength information for this channel
            wavelength_data = None
            for obj in obj_list:
                if hasattr(obj, 'wave') and obj.wave is not None:
                    wavelength_data = obj.wave
                    print(f"wavelength_data found: {wavelength_data}")
                    break

            for i, (obj, meta) in enumerate(zip(obj_list, meta_list)):
                counts = obj.counts
                errors = getattr(obj, 'errors', None)
                if errors is None or errors.shape != counts.shape:
                    errors = np.zeros_like(counts, dtype=np.float32)
                dq = getattr(obj, 'dq', None)
                if dq is None or dq.shape != counts.shape:
                    dq = np.zeros_like(counts, dtype=np.int16)

                n_fibers = counts.shape[0]
                benchside_str = f"{meta.get('bench', '')}{meta.get('side', '')}"
                benchsides.extend([benchside_str] * n_fibers)

                flux_list.append(counts)
                err_list.append(errors)
                dq_list.append(dq)
                fiber_ids.extend(np.arange(n_fibers))
                channels.extend([meta.get('channel', '')]*n_fibers)
                extnums.extend([i]*n_fibers)

                # Try to get wavelength from this object if we don't have it yet
                if wavelength_data is None and hasattr(obj, 'wave') and obj.wave is not None:
                    wavelength_data = obj.wave
                    print(f"wavelength_data if not already found: {wavelength_data}")

            # Stack along fiber axis
            flux_stack = np.vstack(flux_list)
            err_stack = np.vstack(err_list)
            dq_stack = np.vstack(dq_list)

            # SCI extension
            sci_hdu = fits.ImageHDU(data=flux_stack.astype(np.float32), name=f'SCI_{channel}')
            sci_hdu.header['EXTNAME'] = f'SCI_{channel}'

            # Add wavelength information to SCI header if available
            if wavelength_data is not None:
                if isinstance(wavelength_data, np.ndarray) and wavelength_data.size > 1:
                    # Set wavelength reference values in header
                    sci_hdu.header['CRVAL1'] = float(wavelength_data[0])
                    sci_hdu.header['CDELT1'] = float(wavelength_data[1] - wavelength_data[0])
                    sci_hdu.header['CRPIX1'] = 1
                    sci_hdu.header['CTYPE1'] = 'WAVE'
                    sci_hdu.header['CUNIT1'] = 'Angstrom'

                    # Add comment about wavelength calibration
                    sci_hdu.header['COMMENT'] = f'Wavelength range: {wavelength_data[0]:.2f}-{wavelength_data[-1]:.2f} Angstroms'
                    print(f"Added wavelength calibration to SCI_{channel} header: {wavelength_data[0]:.2f}-{wavelength_data[-1]:.2f} Å")

            hdul.append(sci_hdu)

            # ERR extension
            err_hdu = fits.ImageHDU(data=err_stack.astype(np.float32), name=f'ERR_{channel}')
            err_hdu.header['EXTNAME'] = f'ERR_{channel}'
            hdul.append(err_hdu)

            # Add WAVELENGTH extension if data is available
            if wavelength_data is not None:
                if isinstance(wavelength_data, np.ndarray) and wavelength_data.size > 1:
                    wave_hdu = fits.ImageHDU(data=wavelength_data.astype(np.float32), name=f'WAVELENGTH_{channel}')
                    wave_hdu.header['EXTNAME'] = f'WAVELENGTH_{channel}'
                    wave_hdu.header['BUNIT'] = 'Angstrom'
                    hdul.append(wave_hdu)
                    print(f"Added WAVELENGTH_{channel} extension with {len(wavelength_data)} points")

            # FITS table extension
            cols = [
                fits.Column(name='FIBER', format='K', array=np.array(fiber_ids)),
                fits.Column(name='BENCHSIDE', format='A10', array=np.array(benchsides)),
                fits.Column(name='CHANNEL', format='A10', array=np.array(channels)),
                fits.Column(name='EXTNUM', format='K', array=np.array(extnums)),
            ]
            table_hdu = fits.BinTableHDU.from_columns(cols, name=f'TABLE_{channel}')
            table_hdu.header['EXTNAME'] = f'TABLE_{channel}'
            hdul.append(table_hdu)

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
