import os
from astropy.nddata import CCDData
from astropy.stats import mad_std

import ccdproc as ccdp
import matplotlib.pyplot as plt
import numpy as np
import argparse


class BiasLlamas:
    def __init__(self, input_data) -> None:
        
        if isinstance(input_data, str):
            if not os.path.isdir(input_data):
                raise ValueError(f"The directory {input_data} does not exist.")
            self.bias_path = input_data
            self.files = [f for f in os.listdir(input_data) if f.endswith('.fits')]
            
        elif isinstance(input_data, list):
            self.files = input_data
        
        else:
            raise TypeError("Input must be a string (directory path) or a list of files.")
        
        if not self.files:
            raise ValueError("No .fits files found in the provided directory or list.")
        
        return
    
    def master_bias(self):
        
        bias_images = ccdp.ImageFileCollection(self.bias_path)
        calibrated_biases = bias_images.files_filtered(imagetyp='bias', include_path=True)
        
        combined_bias = ccdp.combine(calibrated_biases,
                             method='average',
                             sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                             sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                             mem_limit=350e6
                            )

        combined_bias.meta['combined'] = True

        combined_bias.write(self.bias_path / 'combined_bias.fit')

        if __name__ == "__main__":

            parser = argparse.ArgumentParser(description="Create a master bias frame from a directory of bias images or a list of files.")
            parser.add_argument('input_data', type=str, help="Directory path containing bias images or a list of bias image files.")
            
            args = parser.parse_args()
            
            bias_llamas = BiasLlamas(args.input_data)
            bias_llamas.master_bias()