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
            benchs = []
            sides = []
            channels = []
            extnums = []
            for i, (obj, meta) in enumerate(zip(obj_list, meta_list)):
                counts = obj.counts
                errors = getattr(obj, 'errors', None)
                if errors is None or errors.shape != counts.shape:
                    errors = np.zeros_like(counts, dtype=np.float32)
                dq = getattr(obj, 'dq', None)
                if dq is None or dq.shape != counts.shape:
                    dq = np.zeros_like(counts, dtype=np.int16)
                flux_list.append(counts)
                err_list.append(errors)
                dq_list.append(dq)
                n_fibers = counts.shape[0]
                fiber_ids.extend(np.arange(n_fibers))
                benchs.extend([meta.get('bench', '')]*n_fibers)
                sides.extend([meta.get('side', '')]*n_fibers)
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
                fits.Column(name='BENCH', format='A10', array=np.array(benchs)),
                fits.Column(name='SIDE', format='A10', array=np.array(sides)),
                fits.Column(name='CHANNEL', format='A10', array=np.array(channels)),
                fits.Column(name='EXTNUM', format='K', array=np.array(extnums)),
            ]
            table_hdu = fits.BinTableHDU.from_columns(cols, name=f'TABLE_{channel}')
            table_hdu.header['EXTNAME'] = f'TABLE_{channel}'
            hdul.append(table_hdu)
        hdul.writeto(output_file, overwrite=True)
        print(f"RSS file written to: {output_file}")
        
        return
