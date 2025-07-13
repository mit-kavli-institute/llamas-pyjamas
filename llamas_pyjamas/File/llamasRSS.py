import cloudpickle as pickle
import os
from astropy.io import fits
import numpy as np
import re


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

            # Stack along fiber axis
            flux_stack = np.vstack(flux_list)
            err_stack = np.vstack(err_list)
            dq_stack = np.vstack(dq_list)

            # SCI extension
            sci_hdu = fits.ImageHDU(data=flux_stack.astype(np.float32), name=f'SCI_{channel}')
            sci_hdu.header['EXTNAME'] = f'SCI_{channel}'
            hdul.append(sci_hdu)

            # ERR extension
            err_hdu = fits.ImageHDU(data=err_stack.astype(np.float32), name=f'ERR_{channel}')
            err_hdu.header['EXTNAME'] = f'ERR_{channel}'
            hdul.append(err_hdu)

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

    def sexagesimal_to_decimal_ra(ra_str):
        # Assume ra_str is in "HH MM SS" or "HH:MM:SS" format
        parts = re.split('[: ]+', ra_str.strip())
        if len(parts) != 3:
            raise ValueError("RA string must have three parts: hours, minutes, seconds")
        hours, minutes, seconds = map(float, parts)
        # Convert hours to degrees (15 degrees per hour)
        return 15 * (hours + minutes / 60 + seconds / 3600)

    def sexagesimal_to_decimal_dec(dec_str):
        # Assume dec_str is in "DD MM SS" or "DD:MM:SS" format, may have sign in the first part
        parts = re.split('[: ]+', dec_str.strip())
        if len(parts) != 3:
            raise ValueError("DEC string must have three parts: degrees, minutes, seconds")
        deg = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        sign = -1 if deg < 0 else 1
        return sign * (abs(deg) + minutes / 60 + seconds / 3600)

    with fits.open(fits_file, mode='update') as hdul:
        primary_hdr = hdul[0].header

        ra = primary_hdr.get('RA')
        dec = primary_hdr.get('DEC')

        # Check if RA and DEC are empty, None, or blank strings
        if not ra or str(ra).strip() == "":
            tel_ra = primary_hdr.get('HIERARCH TEL RA')
            if tel_ra is None:
                raise ValueError("Primary header missing HIERARCH TEL RA")
            # Convert RA from sexagesimal to decimal degrees
            dec_ra = sexagesimal_to_decimal_ra(str(tel_ra))
            primary_hdr['RA'] = dec_ra

        if not dec or str(dec).strip() == "":
            tel_dec = primary_hdr.get('HIERARCH TEL DEC')
            if tel_dec is None:
                raise ValueError("Primary header missing HIERARCH TEL DEC")
            # Convert DEC from sexagesimal to decimal degrees
            dec_dec = sexagesimal_to_decimal_dec(str(tel_dec))
            primary_hdr['DEC'] = dec_dec

        hdul.flush()
        print(f"Primary header updated in {fits_file}")