
"""Module for extracting data from LLAMAS observations.

This module provides functionality for extracting data from LLAMAS (Large Lens Array 
Multi-Object Spectrograph) observations. It includes classes and functions for performing 
optimal and boxcar extractions, saving and loading extraction results, and parallel 
processing using Ray.

Classes:
    ExtractLlamas: A class for extracting data from LLAMAS observations using optimal 
        or boxcar methods.
    ExtractLlamasRay: A Ray remote class for parallel processing of LLAMAS extractions.

Functions:
    save_extractions: Save multiple extraction objects to a single file.
    load_extractions: Load a batch of extraction objects from a file.
    parse_args: Parse command-line arguments for input pkl files.

Example:
    This module can be run as a script to process LLAMAS pkl files using parallel 
    processing with Ray::

        python extractLlamas.py *.pkl
"""
from   astropy.io import fits
import scipy
import numpy as np
import os
from datetime import datetime
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
import pickle, cloudpickle
import logging
import argparse, glob
import ray, multiprocessing, psutil
from llamas_pyjamas.Utils.rayManager import init_ray, shutdown_ray
import traceback
from typing import Tuple
import json
import pkg_resources
from pathlib import Path

####################################################################################

from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, LUT_DIR
from llamas_pyjamas.Trace.traceLlamas import TraceLlamas

logger = logging.getLogger(__name__)

# Detector noise defaults used for per-fibre variance (F4) when the FITS header
# does not carry EGAIN/RDNOISE. The Poisson term dominates, so these only set
# the read-noise floor; override via header keywords when available.
DEFAULT_GAIN = 1.0        # e-/ADU
DEFAULT_READNOISE = 2.5   # e-


class ExtractLlamas:
    """A class used to extract data from LLAMAS observations.

    This class handles the extraction of spectral data from LLAMAS observations using 
    either optimal or boxcar extraction methods.

    Attributes:
        trace (TraceLlamas): An instance of the TraceLlamas class containing trace information.
        bench (str): The bench identifier from the trace.
        side (str): The side identifier from the trace.
        channel (str): The channel identifier from the trace.
        fitsfile (str): The FITS file associated with the trace.
        counts (np.ndarray): An array to store the extracted counts.
        hdr (dict): Header information from the FITS file.
        frame (np.ndarray): The data frame from the FITS file.
        x (np.ndarray): An array representing the x-axis.
        xshift (np.ndarray): An array to store x-axis shifts.
        wave (np.ndarray): An array to store wavelength information.
        ximage (np.ndarray): An array representing the x-axis image.
        relative_throughput (np.ndarray): Array storing relative throughput for each fiber.
        fiberid (np.ndarray): Array storing fiber IDs.
    """

    def __init__(self,trace: TraceLlamas, hdu_data: np.ndarray, hdr: dict,optimal=True,
                 method=None) -> None:
        """Initialize the ExtractLlamas object.

        Args:
            trace (TraceLlamas): An instance of the TraceLlamas class containing trace information.
            hdu_data (np.ndarray): The HDU data array from the FITS file.
            hdr (dict): Header information from the FITS file.
            optimal (bool, optional): Legacy switch — True = profile-weighted mean,
                False = boxcar. Ignored when ``method`` is given.
            method (str, optional): 'horne' | 'boxcar' | 'optimal'. 'horne' is the
                variance-weighted, flux-conserving, mask-aware estimator
                (Horne 1986) built on the bspline profile images.

        Returns:
            None
        """

        if (trace is None or hdu_data is None or hdr is None):
            # Instantiate a blank object that can be used for a deep copy
            self.trace      = None
            self.bench      = None
            self.side       = None
            self.channel    = None
            self.fitsfile   = None     
            self.hdr        = None
            self.frame      = None
            self.fiberid    = None
            self.relative_throughput = None
            self.x          = None
            self.ximage     = None  
            self.xshift     = None
            self.wave       = None
            self.counts     = None
            self.counts_err = None
            self.sky        = None
            self.sensfunc   = None
            self.flux       = None
            self.flux_err   = None

        else:

            self.trace = trace
            self.bench = trace.bench
            self.side = trace.side
            self.channel = trace.channel
            self.fitsfile = self.trace.fitsfile
            ##put in a check here for hdu against trace attributes when I have more brain capacity        
            
            self.hdr    = hdr
            self.frame  = hdu_data.astype(float)
            self.fiberid = np.arange(trace.nfibers)#np.zeros(shape=(trace.nfibers))
            self.relative_throughput = np.zeros(shape=(trace.nfibers))

            self.x      = np.arange(trace.naxis1)
            self.ximage = np.outer(np.ones(trace.naxis2),np.arange(trace.naxis1))
            # xshift and wave will be populated only after an arc solution
            self.xshift = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.wave   = np.zeros(shape=(trace.nfibers,trace.naxis1))

            self.counts     = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.counts_err = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.sky        = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.sensfunc   = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.flux       = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.flux_err   = np.zeros(shape=(trace.nfibers,trace.naxis1))
            
            # Get detector properties from header if available
            # self.gain = hdr.get('EGAIN', 1.0)  # e-/ADU, default to 1.0
            # self.readnoise = hdr.get('RDNOISE', 3.0) 

            if method is None:
                method = 'optimal' if optimal else 'boxcar'
            method = str(method).lower()
            # 'optimal' is the user-facing name for the Horne estimator; the
            # pre-2026-07 profile-weighted mean survives as 'legacy' for
            # comparison only.
            if method == 'optimal':
                method = 'horne'
            print(f'Extraction method: {method}')
            print(f'bench {self.bench} self.side {self.side} channel {self.channel}')

            # Read noise [DN] for Horne variance weighting; modest errors here
            # only perturb the weights, not the flux normalisation.
            _rn2 = float(os.environ.get('LLAMAS_READ_NOISE', '3.5')) ** 2
            _nx = trace.naxis1
            
            benchside = str(self.bench) + str(self.side)
            with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'r') as f:
                LUT = json.load(f)
                self.LUT = LUT 
            
            
            # Safely handle missing entries in the LUT
            try:
                if 'dead_fibers' in self.LUT and benchside in self.LUT['dead_fibers']:
                    self.dead_fibers = self.LUT['dead_fibers'][benchside]
                    print(f'Dead fibers: {self.dead_fibers}')
                else:
                    self.dead_fibers = []  # Default to empty list if entry doesn't exist
                    logger.info(f"No dead fibers entry found in LUT for bench={self.bench}, side={self.side}")
            except Exception as e:
                self.dead_fibers = []  # Default to empty list if any error occurs
                logger.warning(f"Error accessing dead fibers in LUT: {e}")


            for ifiber in range(trace.nfibers):
                extracted = np.zeros(self.trace.naxis1)
                # print fiber, and list of dead fibers, print also bench and color
                logger.info(self.dead_fibers)
                logger.info(f'Extracting fiber # {ifiber} of {trace.nfibers} for bench {benchside} channel {self.channel}')
                if ifiber in self.dead_fibers:
                    logger.info(f"Skipping dead fiber # {ifiber}")
                    self.counts[ifiber,:] = extracted
                    continue

                if method == 'horne':
                    # Horne (1986) optimal extraction: variance-weighted,
                    # flux-conserving, mask-aware.
                    #   F = sum(P*f/V) / sum(P^2/V),  Var(F) = 1/sum(P^2/V)
                    # with the bspline profile P renormalised to sum to 1 per
                    # column over the VALID pixels — so missing/bad pixels
                    # renormalise the profile instead of silently biasing the
                    # flux (the old profile-weighted mean kept dropped pixels'
                    # weight in the denominator).
                    ys, xs = np.where(self.trace.fiberimg == ifiber)
                    if ys.size == 0:
                        logger.warning("No profile pixels for fiber #{}".format(ifiber))
                        continue
                    P = self.trace.profimg[ys, xs]
                    fpix = self.frame[ys, xs]
                    good = np.isfinite(fpix) & np.isfinite(P) & (P > 0)
                    ys, xs, P, fpix = ys[good], xs[good], P[good], fpix[good]
                    if xs.size == 0:
                        continue
                    # per-column profile normalisation over valid pixels
                    colP = np.zeros(_nx)
                    np.add.at(colP, xs, P)
                    Pn = P / np.where(colP[xs] > 0, colP[xs], 1.0)

                    # Pass 1: profile-weighted flux with constant variance,
                    # used only to build the MODEL-based variance. Weighting by
                    # the raw data instead (V = RN^2 + f) anti-correlates the
                    # weights with the noise and costs ~10% S/N.
                    num0 = np.zeros(_nx); den0 = np.zeros(_nx)
                    np.add.at(num0, xs, Pn * fpix)
                    np.add.at(den0, xs, Pn * Pn)
                    F0 = np.where(den0 > 0, num0 / np.where(den0 > 0, den0, 1.0), 0.0)
                    # Smooth the model along the dispersion axis before it
                    # enters the variance: an unsmoothed F0 makes the weights
                    # track the frame's own noise, decorrelating repeat
                    # exposures (measured: 4B stability 0.96 -> 0.80). Real
                    # spectral structure is preserved at the 9-px scale.
                    _k = 9
                    F0s = np.convolve(np.clip(F0, 0, None), np.ones(_k) / _k, mode='same')

                    # Pass 2: Horne with model variance V = RN^2 + max(F0s*P, 0)
                    V = _rn2 + np.clip(F0s[xs] * Pn, 0, None)
                    w = Pn / V
                    num = np.zeros(_nx)
                    den = np.zeros(_nx)
                    np.add.at(num, xs, w * fpix)
                    np.add.at(den, xs, w * Pn)
                    ok = den > 0
                    self.counts[ifiber, :] = np.where(ok, num / np.where(ok, den, 1.0), 0.0)
                    self.counts_err[ifiber, :] = np.where(ok, 1.0 / np.sqrt(np.where(ok, den, 1.0)), 0.0)

                elif method == 'legacy':
                    # LEGACY profile-weighted mean (NOT Horne): kept for
                    # comparison only. Not flux conserving; dropped pixels bias
                    # the flux low because the denominator keeps their weight.
                    x_spec,f_spec,weights = self.isolateProfile(ifiber)
                    if x_spec is None:
                        continue

                    extracted = np.zeros(self.trace.naxis1)
                    for i in range(self.trace.naxis1):
                        thisx = (x_spec == i)
                        if np.nansum(thisx) > 0:
                            extracted[i] = np.nansum(f_spec[thisx]*weights[thisx])/np.nansum(weights[thisx])
                        #handles case where there are no elements
                        else:
                            extracted[i] = 0.0

                    self.counts[ifiber,:] = extracted
                
                elif method == 'boxcar':
                    # Boxcar extraction with a trace-following aperture and
                    # FRACTIONAL pixel weights at the aperture edges.  An
                    # integer-rounded window (the previous implementation)
                    # jumps by a whole pixel row whenever the trace crosses a
                    # half-pixel boundary, imprinting discontinuities along
                    # the spectrum; weighting the edge pixels by their
                    # geometric overlap keeps the aperture continuous.
                    logger.info("..Boxcar extracting fiber #{}".format(ifiber))
                    extracted = np.zeros(self.trace.naxis1)
                    tracey = self.trace.traces[ifiber,:]
                    # Aperture half-width [px]. The legacy window was 9 px total
                    # (half=4.5), but at the ~6.9 px fibre pitch that reaches the
                    # neighbouring fibres' cores; 2.5 px (validated on-sky
                    # 2026-07: frame-to-frame flux stability 0.98-0.99) keeps the
                    # aperture on this fibre. Overridable via
                    # LLAMAS_BOXCAR_HALFWIDTH (set by reduce.py from the
                    # boxcar_halfwidth config key).
                    half = float(os.environ.get('LLAMAS_BOXCAR_HALFWIDTH', '2.5'))
                    ny = self.frame.shape[0]
                    for i in range(self.trace.naxis1):
                        yc = tracey[i]
                        if not np.isfinite(yc):
                            continue
                        lo, hi = yc - half, yc + half
                        # pixel j (centre convention) spans [j-0.5, j+0.5)
                        j0 = int(np.floor(lo + 0.5))          # first pixel with any overlap
                        j1 = int(np.floor(hi + 0.5 - 1e-9))   # last pixel with any overlap
                        if j0 < 0 or j1 >= ny:
                            continue   # aperture falls off the detector
                        total = 0.0
                        for jj in range(j0, j1 + 1):
                            w = min(hi, jj + 0.5) - max(lo, jj - 0.5)
                            if w > 0:
                                total += self.frame[jj, i] * min(w, 1.0)
                        extracted[i] = total

                    self.counts[ifiber,:] = extracted
            self.old_count_shape = self.counts.shape
            logger.info(f'Benchside {benchside} counts shape {self.counts.shape}')
            # NOTE: dead fibres are NOT inserted into counts here. Every per-fibre
            # array (counts, wave, xshift, sky, throughput, errors, ...) stays
            # LIVE-indexed and mutually aligned, so per-fibre operations
            # (arcTransfer, skyModel) pair the correct rows. Previously only
            # counts was padded to fibremap indexing while wave/xshift/sky stayed
            # live, so counts[i] and wave[i] described different fibres after the
            # first dead fibre (a silent misalignment). The fibermap expansion is
            # done once, explicitly, at RSS generation via
            # llamas_pyjamas.Utils.deadfibers (dead_fibers holds the fibremap
            # positions). self.dead_fibers is preserved for that step.

            # F4 (Pass 1): per-fibre uncertainty from photon + read noise.
            # Populates `errors`, which llamasRSS writes to the ERROR extension
            # and which fibreFlat/skySubtract propagate. Previously unset, so the
            # ERROR extension was all zeros and no S/N was derivable from products.
            # Detector gain (e-/ADU) and read noise (e-) are read from the header
            # when available; the Poisson term dominates either way.
            def _hdr_num(keys, default):
                for k in keys:
                    v = self.hdr.get(k)
                    if v is not None:
                        try:
                            fv = float(v)
                            if fv > 0:
                                return fv
                        except (TypeError, ValueError):
                            pass
                return default
            gain = _hdr_num(('EGAIN', 'GAIN', 'GAIN1', 'CCDGAIN'), DEFAULT_GAIN)
            readnoise = _hdr_num(('RDNOISE', 'RDNOISE1', 'READNOIS', 'RON'),
                                 DEFAULT_READNOISE)
            aperture_pix = float(getattr(self.trace, 'extraction_aperture', 9.0))
            counts_e = np.clip(self.counts, 0.0, None) * gain
            var_adu = (counts_e + aperture_pix * (readnoise ** 2)) / (gain ** 2)
            self.counts_err = np.sqrt(var_adu).astype(np.float32)
            # At extraction, flux == counts (no flux cal), so the error on the
            # written FLUX/COUNTS is the same array.
            self.errors = self.counts_err.copy()
            logger.info(f'Computed per-fibre errors (gain={gain:.3f} e-/ADU, '
                        f'readnoise={readnoise:.2f} e-, aperture={aperture_pix:.0f} '
                        f'pix); median error={np.median(self.counts_err):.3f}')

                    
                
                
                    

    def isolateProfile(self,ifiber, boxcar=False)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Isolate the profile for a given fiber.

        Args:
            ifiber (int): The fiber index to isolate the profile for.
            boxcar (bool, optional): If True, use boxcar extraction. If False, use profile 
                extraction. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - x_spec (np.ndarray or None): The x-coordinates of the spectrum for the given fiber.
                - f_spec (np.ndarray or None): The flux values of the spectrum for the given fiber.
                - weights (np.ndarray or None): The weights for the spectrum extraction, either 
                  boxcar or profile-based.

        Note:
            If no profile is found for the given fiber, the function will return (None, None, None) 
            and log a warning.
        """

        #profile  = self.trace.profimg[ifiber]
        inprofile = self.trace.fiberimg == ifiber
        profile = self.trace.profimg[self.trace.fiberimg == ifiber]
            
        #inprof = np.where(profile > 0)
        if inprofile.size == 0:
            logger.warning("No profile for fiber #{}".format(ifiber))
            return None,None,None
        
        x_spec = self.ximage[inprofile]
        f_spec = self.frame[inprofile]
        if boxcar == True:
            weights = np.where(inprofile, 1, 0)[inprofile]#[weights]
            # weights = np.ones_like(f_spec)

        elif boxcar == False:
            weights = self.trace.profimg[inprofile]#self.trace.profimg[inprofile]
        
        return x_spec,f_spec,weights


    # Set to False to include full 2D detector images in pickled files (for QA/troubleshooting).
    # Default True strips ~7 GB of per-pixel arrays that are not needed after extraction.
    _slim_pickle = True

    def __getstate__(self):
        """Custom pickle state: optionally strip large 2D detector images.

        When _slim_pickle is True (default), removes frame, ximage, and the trace's
        2D images (data, fiberimg, profimg, bpmask), which together account for ~7 GB
        per extraction batch.  Per-fiber arrays (counts, sky, wave, xshift,
        relative_throughput, sensfunc, flux, flux_err, etc.) are always preserved.

        To keep the full images for QA, set before saving:
            ExtractLlamas._slim_pickle = False
        """
        import copy
        state = self.__dict__.copy()
        if ExtractLlamas._slim_pickle:
            # Strip large per-pixel 2D arrays from the extraction object itself
            state['frame']  = None
            state['ximage'] = None
            # Strip trace.data (raw detector image copy in the trace) but keep
            # fiberimg and profimg — they are needed by generate_pixel_flat_extension
            # when the flat extraction pkl is reloaded for flat field generation.
            if state.get('trace') is not None:
                slim_trace = copy.copy(state['trace'])
                slim_trace.data = None
                state['trace'] = slim_trace
        return state

    def saveExtraction(self, save_dir: str)-> None:
        """Save the current extraction object to a file in the specified directory.

        This method constructs the output file path using the object's attributes
        (channel, side, bench) and saves the object using cloudpickle.

        Args:
            save_dir (str): The directory where the extraction file will be saved.
                Note that this argument is overridden and the file is always saved 
                in the 'output' directory relative to the script's location.

        Returns:
            None
        """

        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        outfile = f'LLAMASExtract_{self.channel}_{self.side}_{self.bench}.pkl'
        outpath = os.path.join(save_dir, outfile)
        logger.info(f'outpath {outpath}')
        with open(outpath,'wb') as fp:
            cloudpickle.dump(self, fp)
        return

    def loadExtraction(infile: str)-> object:
        """Load an object from a pickle file.

        Args:
            infile (str): The path to the input file containing the pickled object.

        Returns:
            object: The object loaded from the pickle file.
        """

        with open(infile,'r+b') as fp:
            object = pickle.load(fp)
        return(object)


def save_extractions(extraction_list, primary_header=None, savefile=None, save_dir=None, prefix='LLAMASExtract_batch')-> str:
    """Save multiple extraction objects to single file.

    Args:
        extraction_list (list): List of extraction objects to save.
        savefile (str, optional): Specific filename to use. If None, a timestamped 
            filename will be generated. Defaults to None.
        save_dir (str, optional): Directory to save the file. If None, uses the 
            default output directory. Defaults to None.
        prefix (str, optional): Prefix for the filename when savefile is None. 
            Defaults to 'LLAMASExtract_batch'.

    Returns:
        str: Path to the saved file.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create metadata for each extraction
    batch_data = {
        'primary_header': primary_header,
        'extractions': extraction_list,
        'metadata': [{
            'channel': ext.channel,
            'bench': ext.bench,
            'side': ext.side,
            'nfibers': ext.trace.nfibers
        } for ext in extraction_list]
    }
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = f'{prefix}_{timestamp}.pkl'
    if (savefile != None):
        outpath = os.path.join(save_dir, savefile)
    else:   
        outpath = os.path.join(save_dir, outfile)
    
    logger.info(f'Saving batch extraction to: {outpath}')
    with open(outpath, 'wb') as fp:
        cloudpickle.dump(batch_data, fp)
    return outpath

@staticmethod
def load_extractions(infile: str)-> Tuple[list, list]:
    """Load batch of extraction objects.

    Args:
        infile (str): Path to the file containing the batch extraction data.

    Returns:
        tuple[list, list]: A tuple containing:
            - extractions (list): List of extraction objects.
            - metadata (list): List of metadata dictionaries for each extraction.
    """
    with open(infile, 'rb') as fp:
        batch_data = cloudpickle.load(fp)
    
    logger.info(f"Loaded {len(batch_data['extractions'])} extractions")
    return batch_data['extractions'], batch_data['metadata']

def sort_extractions(input_pickle, output_pickle=None):
    """
    Sort extraction objects in canonical order matching idx_lookup from constants.py.

    The ordering follows: bench (1-4) -> side (A, B) -> channel (red, green, blue)
    This ensures consistent ordering across all extraction files for downstream processing.

    Args:
        input_pickle (str): Path to input pickle file with extractions
        output_pickle (str, optional): Path to output pickle file. If None, overwrites input file.

    Returns:
        str: Path to the sorted output pickle file

    Example:
        >>> sort_extractions('combined_flat_extractions.pkl', 'sorted_flat_extractions.pkl')
    """
    from llamas_pyjamas.constants import idx_lookup

    # Load the extraction file
    with open(input_pickle, 'rb') as fp:
        batch_data = cloudpickle.load(fp)

    extractions = batch_data['extractions']
    metadata = batch_data['metadata']
    primary_header = batch_data.get('primary_header', None)

    logger.info(f"Loaded {len(extractions)} extractions from {input_pickle}")

    # Create list of (sort_index, extraction, metadata) tuples
    ext_with_idx = []

    for ext, meta in zip(extractions, metadata):
        channel = meta['channel']
        bench = str(meta['bench'])
        side = meta['side']

        # Get sort index from lookup
        sort_key = (channel, bench, side)
        if sort_key not in idx_lookup:
            logger.warning(f"Extension {channel} {bench}{side} not in standard ordering - skipping")
            continue

        idx = idx_lookup[sort_key]
        ext_with_idx.append((idx, ext, meta))

    # Sort by index
    ext_with_idx.sort(key=lambda x: x[0])

    logger.info(f"Sorted {len(ext_with_idx)} extractions in canonical order")

    # Verify we have expected number of extensions
    if len(ext_with_idx) != 24:
        logger.warning(f"Expected 24 extensions, found {len(ext_with_idx)}")
        missing_indices = set(range(1, 25)) - set([x[0] for x in ext_with_idx])
        if missing_indices:
            logger.warning(f"Missing extension indices: {sorted(missing_indices)}")

    # Extract sorted extractions and metadata
    sorted_extractions = [x[1] for x in ext_with_idx]
    sorted_metadata = [x[2] for x in ext_with_idx]

    # Create sorted batch data
    sorted_batch_data = {
        'primary_header': primary_header,
        'extractions': sorted_extractions,
        'metadata': sorted_metadata
    }

    # Determine output path
    if output_pickle is None:
        output_pickle = input_pickle

    # Save sorted data
    with open(output_pickle, 'wb') as fp:
        cloudpickle.dump(sorted_batch_data, fp)

    logger.info(f"Saved sorted extractions to {output_pickle}")

    # Print summary of ordering
    print("\nSorted extraction order:")
    for i, (idx, ext, meta) in enumerate(ext_with_idx):
        print(f"  Extension {i}: idx={idx:2d} | {meta['channel']:5s} {meta['bench']}{meta['side']} | {meta['nfibers']} fibers")

    return output_pickle

@ray.remote
class ExtractLlamasRay(ExtractLlamas):
    """Ray remote class for parallel processing of LLAMAS extractions.

    ExtractLlamasRay is a subclass of ExtractLlamas that handles the extraction 
    process for LLAMAS data using Ray for parallel processing.

    Attributes:
        files (list): A list of files to be processed.
    """

    
    def __init__(self, files) -> None:
        """Initialize the ExtractLlamasRay instance.

        Args:
            files (list): A list of files to be processed.

        Returns:
            None
        """
        self.files = files
        pass
        
   
    def process_extractions(self, tracepkl: "TraceLlamas") -> None:
        """Process the extraction of LLAMAS traces from a given pickle file.

        Args:
            tracepkl (TraceLlamas): The path to the pickle file containing the LLAMAS trace data.

        Returns:
            None
        """

        with open(tracepkl, "rb") as tracer:
            trace = pickle.load(tracer)
        
        extraction = super.__init__(trace)
        extraction.saveExtraction()
        return

def parse_args()-> list:
    """Parse command-line arguments to process LLAMAS pkl files.

    This function sets up an argument parser to accept one or more file paths
    (with support for wildcards like *.pkl), expands the wildcards, validates
    the files, and returns a list of valid .pkl files.

    Returns:
        list: A list of valid .pkl file paths.

    Raises:
        ValueError: If no .pkl files are found.
    """

    parser = argparse.ArgumentParser(description='Process LLAMAS pkl files.')
    parser.add_argument('files', nargs='+', help='Path to input pkl files (supports wildcards like *.pkl)')
    args = parser.parse_args()
    
    # Expand wildcards and validate files
    pkl_files = []
    for pattern in args.files:
        matched_files = glob.glob(pattern)
        if not matched_files:
            print(f"Warning: No files found matching pattern {pattern}")
            continue
        pkl_files.extend([f for f in matched_files if f.endswith('.pkl')])
    
    if not pkl_files:
        raise ValueError("No .pkl files found!")
        
    return pkl_files


if __name__ == '__main__':
    # Example of how to run the extraction
    # ray.init(ignore_reinit_error=True)
   ##Need to edit this and the remote class so that it runs the extraction through ray not just multiple files at once.
    files = parse_args()
    
    NUMBER_OF_CORES = int(os.environ.get('LLAMAS_RAY_CPUS', multiprocessing.cpu_count()))
    init_ray(num_cpus=NUMBER_OF_CORES)

    print(f"\nStarting with {NUMBER_OF_CORES} cores available")
    print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    futures = []
    results = []
    
    extraction_processors = [ExtractLlamasRay.remote(file) for file in files]
    
    print(f"\nProcessing {len(files)} files with {NUMBER_OF_CORES} cores")
    for processor in extraction_processors:
        futures.append(processor.process_extractions.remote())
    
    #Monitor processing
    total_jobs = len(futures)
    completed = 0
        
    # Monitor processing
    print("\nProcessing Status:")
    while futures:
        
        # Print current CPU usage every 5 seconds
        if completed % 5 == 0:
            print(f"CPU Usage: {psutil.cpu_percent(percpu=True)}%")
            print(f"Progress: {completed}/{total_jobs} jobs complete")
        
        
        done_id, futures = ray.wait(futures)
        result = ray.get(done_id[0])
        results.append(result)
        completed += 1
        
    print(f"\nAll {total_jobs} jobs complete")
    print(f"Final CPU Usage: {psutil.cpu_percent(percpu=True)}%")

    shutdown_ray()   # standalone CLI: release Ray (scratch cleaned by atexit)
