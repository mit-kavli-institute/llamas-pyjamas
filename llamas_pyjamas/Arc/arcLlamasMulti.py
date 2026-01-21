from   astropy.io import fits
import scipy
import numpy as np
import llamas_pyjamas.Extract.extractLlamas as extract
from   matplotlib import pyplot as plt
from   scipy.ndimage import generic_filter
from   scipy.signal import correlate
from   astropy.table import Table
import astropy.units as u
import os
from   llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, LUT_DIR

from pypeit.core.wavecal.wvutils import xcorr_shift_stretch
from pypeit.core.wave import airtovac
from pypeit.core.arc import detect_peaks
from pypeit import bspline
from pypeit.core.fitting import iterfit, robust_fit # type: ignore
import warnings

# Ray multiprocessing imports
import ray
import multiprocessing
import time
import psutil
import logging
from typing import List, Dict, Tuple, Optional

# Setup logging for Ray processing
logger = logging.getLogger(__name__)

###############################################################################3

def nan_median_filter(input_spectrum):
    """Filter function to compute median while ignoring NaN values.

    Args:
        input_spectrum (np.ndarray): Input spectrum array.

    Returns:
        float: Median value of non-NaN elements, or NaN if no valid values exist.
    """
    values = input_spectrum[~np.isnan(input_spectrum)]  # Remove NaNs
    return np.median(values) if values.size else np.nan

def interpolateNaNs(input_spectrum):
    """Interpolate NaN values in spectrum using median filter.

    This function replaces NaN values in the input spectrum by applying a 
    median filter that ignores NaN values.

    Args:
        input_spectrum (np.ndarray): Input spectrum array that may contain NaN values.

    Returns:
        np.ndarray: Filtered spectrum with NaN values interpolated.
    """
    array_filtered = generic_filter(input_spectrum, nan_median_filter, size=3)
    return(array_filtered)

# Ray remote functions for parallel processing

@ray.remote
def process_fiber_shift(fits_ext: int, ifiber: int, refspec: np.ndarray, fiber_counts: np.ndarray, 
                       stretch_func: str = 'quadratic') -> Tuple[int, int, bool, np.ndarray]:
    """Process shift and stretch calculation for a single fiber with robust error handling.
    
    Args:
        fits_ext (int): FITS extension number
        ifiber (int): Fiber index
        refspec (np.ndarray): Reference spectrum
        fiber_counts (np.ndarray): Fiber counts data
        stretch_func (str): Stretch function type
        
    Returns:
        Tuple containing: (fits_ext, ifiber, success, xshift_result)
    """
    x = np.arange(2048)
    
    # FIRST ATTEMPT: Use existing interpolateNaNs approach (same as original)
    try:
        success, shift, stretch, stretch2, _, _, _ = \
            xcorr_shift_stretch(refspec, interpolateNaNs(fiber_counts), stretch_func=stretch_func)
            
        if success == 1:
            xshift_result = (x*stretch + x**2*stretch2) + shift
        else:
            # USE LOGGING WHEN SUCCESS != 1
            logger.warning(f"Arc shift failed for fiber {ifiber} in extension {fits_ext} (success={success})")
            xshift_result = x
            
        return fits_ext, ifiber, success, xshift_result
        
    except ValueError as e:
        if "array must not contain infs or NaNs" in str(e):
            logger.info(f"Applying data cleaning for fiber {ifiber} in extension {fits_ext} due to NaN/Inf values")
            
            # SECOND ATTEMPT: Enhanced cleaning for Ray processing
            try:
                cleaned_fiber_counts = _robust_clean_spectrum_ray(fiber_counts)
                cleaned_refspec = _robust_clean_spectrum_ray(refspec)
                
                success, shift, stretch, stretch2, _, _, _ = \
                    xcorr_shift_stretch(cleaned_refspec, cleaned_fiber_counts, stretch_func=stretch_func)
                    
                if success == 1:
                    xshift_result = (x*stretch + x**2*stretch2) + shift
                    logger.info(f"Fiber {ifiber} processed successfully after data cleaning")
                else:
                    # USE LOGGING WHEN CLEANED DATA ALSO FAILS
                    logger.warning(f"Arc shift failed for fiber {ifiber} even after cleaning (success={success})")
                    xshift_result = x
                
                return fits_ext, ifiber, success, xshift_result
                
            except Exception as e2:
                logger.error(f"Data cleaning failed for fiber {ifiber} in extension {fits_ext}: {str(e2)}")
                return fits_ext, ifiber, False, x
        else:
            logger.error(f"Unexpected error in fiber {ifiber} extension {fits_ext}: {str(e)}")
            raise e

@ray.remote
def process_fiber_throughput(fits_ext: int, ifiber: int, fiber_spec: np.ndarray, 
                           xshift_fiber: np.ndarray, reference_flux: float) -> Tuple[int, int, float]:
    """Process relative throughput calculation for a single fiber with logging.
    
    Args:
        fits_ext (int): FITS extension number
        ifiber (int): Fiber index
        fiber_spec (np.ndarray): Fiber spectrum data
        xshift_fiber (np.ndarray): X-shift data for fiber
        reference_flux (float): Reference flux for normalization
        
    Returns:
        Tuple containing: (fits_ext, ifiber, relative_throughput)
    """
    try:
        gd = (xshift_fiber > 150) & (xshift_fiber < 2048-150)
        flux = np.nansum(fiber_spec[gd])
        
        if reference_flux <= 0:
            logger.warning(f"Invalid reference flux ({reference_flux}) for fiber {ifiber} in extension {fits_ext}")
            relative_throughput = 0.0
        else:
            relative_throughput = flux / reference_flux
            
        # Log unusual throughput values for quality control
        if relative_throughput < 0.1 or relative_throughput > 2.0:
            logger.info(f"Unusual throughput value {relative_throughput:.3f} for fiber {ifiber} in extension {fits_ext}")
        
        return fits_ext, ifiber, relative_throughput
        
    except Exception as e:
        logger.error(f"Throughput calculation failed for fiber {ifiber} in extension {fits_ext}: {str(e)}")
        return fits_ext, ifiber, 0.0  # Return zero throughput as fallback

@ray.remote
def process_extension_wavelength_transfer(extension_idx: int, arcspec_data: Dict, 
                                        final_arcfit, channel: str) -> Tuple[int, Dict]:
    """Process wavelength solution transfer for an entire extension with logging.
    
    Args:
        extension_idx (int): Extension index
        arcspec_data (Dict): Arc spectrum data for the extension
        final_arcfit: Fitted arc solution
        channel (str): Channel type (red/green/blue)
        
    Returns:
        Tuple containing: (extension_idx, updated_wavelength_data)
    """
    try:
        nfibers = arcspec_data['nfibers']
        xshift = arcspec_data['xshift']
        
        # Calculate wavelength solution for all fibers in this extension
        wavelength_data = {}
        failed_fibers = []
        
        for ifiber in range(nfibers):
            try:
                x = xshift[ifiber, :]
                wavelength_solution = final_arcfit.eval(x)
                
                # Basic validation of wavelength solution
                if np.any(~np.isfinite(wavelength_solution)):
                    logger.warning(f"Non-finite wavelength values for fiber {ifiber} in extension {extension_idx}")
                    failed_fibers.append(ifiber)
                
                wavelength_data[ifiber] = wavelength_solution
                
            except Exception as e:
                logger.error(f"Wavelength transfer failed for fiber {ifiber} in extension {extension_idx}: {str(e)}")
                failed_fibers.append(ifiber)
                # Use default wavelength array as fallback
                wavelength_data[ifiber] = np.arange(len(x))
        
        if failed_fibers:
            logger.warning(f"Wavelength transfer failed for {len(failed_fibers)} fibers in extension {extension_idx} ({channel}): {failed_fibers}")
        else:
            logger.debug(f"Successfully transferred wavelength solution to {nfibers} fibers in extension {extension_idx} ({channel})")
        
        return extension_idx, wavelength_data
        
    except Exception as e:
        logger.error(f"Extension wavelength transfer failed for extension {extension_idx} ({channel}): {str(e)}")
        # Return empty result to indicate failure
        return extension_idx, {}

# Ray-specific data cleaning functions (enhanced error handling for Ray processing)

def _robust_clean_spectrum_ray(spectrum):
    """Enhanced spectrum cleaning for Ray processing - handles NaN and Inf values.
    
    This function is specifically designed for Ray workers to handle problematic
    spectrum data that causes xcorr_shift_stretch to fail.
    
    Args:
        spectrum (np.ndarray): Input spectrum that may contain NaN or Inf values
        
    Returns:
        np.ndarray: Cleaned spectrum with no NaN or Inf values
    """
    spectrum = np.asarray(spectrum, dtype=float)
    
    # Step 1: Replace Inf values with NaN first
    inf_mask = np.isinf(spectrum)
    if np.any(inf_mask):
        spectrum[inf_mask] = np.nan
        logger.debug(f"Replaced {np.sum(inf_mask)} Inf values with NaN")
    
    # Step 2: Enhanced NaN interpolation if any NaNs present
    if np.any(np.isnan(spectrum)):
        spectrum = _enhanced_interpolate_nans_ray(spectrum)
    
    # Step 3: Final validation - ensure no NaN/Inf remain
    invalid_mask = ~np.isfinite(spectrum)
    if np.any(invalid_mask):
        valid_values = spectrum[np.isfinite(spectrum)]
        if len(valid_values) > 0:
            fill_value = np.median(valid_values)
            logger.debug(f"Filled {np.sum(invalid_mask)} remaining invalid values with median: {fill_value}")
        else:
            fill_value = 0.0
            logger.warning("No valid values found in spectrum, filling with zeros")
        spectrum[invalid_mask] = fill_value
    
    return spectrum

def _enhanced_interpolate_nans_ray(spectrum):
    """Enhanced NaN interpolation for Ray processing with multiple fallback methods.
    
    Args:
        spectrum (np.ndarray): Input spectrum with NaN values
        
    Returns:
        np.ndarray: Spectrum with NaN values interpolated
    """
    from scipy import interpolate
    
    # Method 1: Try linear interpolation first
    valid_mask = np.isfinite(spectrum)
    valid_count = np.sum(valid_mask)
    
    if valid_count >= 2:
        # Linear interpolation for interior points
        valid_indices = np.where(valid_mask)[0]
        valid_values = spectrum[valid_mask]
        
        try:
            f = interpolate.interp1d(valid_indices, valid_values, 
                                   kind='linear', bounds_error=False, 
                                   fill_value='extrapolate')
            spectrum = f(np.arange(len(spectrum)))
            logger.debug(f"Applied linear interpolation to spectrum ({valid_count} valid points)")
        except Exception as e:
            logger.debug(f"Linear interpolation failed: {str(e)}, falling back to median filter")
    
    # Method 2: If still have NaNs, use median filter (original approach)
    if np.any(np.isnan(spectrum)):
        spectrum = generic_filter(spectrum, nan_median_filter, size=3)
        logger.debug("Applied median filter interpolation")
    
    # Method 3: If still have NaNs, forward/backward fill
    if np.any(np.isnan(spectrum)):
        spectrum = _forward_backward_fill_ray(spectrum)
        logger.debug("Applied forward/backward fill")
    
    return spectrum

def _forward_backward_fill_ray(spectrum):
    """Forward and backward fill for remaining NaN values.
    
    Args:
        spectrum (np.ndarray): Input spectrum with potential NaN values
        
    Returns:
        np.ndarray: Spectrum with NaN values filled
    """
    spectrum = spectrum.copy()
    
    # Forward fill
    mask = np.isnan(spectrum)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    spectrum[mask] = spectrum[idx[mask]]
    
    # Backward fill for any remaining NaNs at the beginning
    mask = np.isnan(spectrum)
    if np.any(mask):
        idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        spectrum[mask] = spectrum[idx[mask]]
    
    return spectrum

# Ray initialization and management functions

def _initialize_ray_if_needed():
    """Initialize Ray if not already running, with graceful fallback to serial processing.
    
    Returns:
        bool: True if Ray is available and initialized, False if fallback to serial
    """
    try:
        if not ray.is_initialized():
            NUMBER_OF_CORES = int(os.environ.get('LLAMAS_RAY_CPUS', multiprocessing.cpu_count()))
            print(f"🚀 Initializing Ray with {NUMBER_OF_CORES} CPU cores for parallel processing...")
            print(f"💾 Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
            
            ray.shutdown()  # Clear any existing Ray instances
            ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
            
            print(f"✅ Ray initialized successfully with {NUMBER_OF_CORES} cores")
            return True
        else:
            print("🔄 Ray already initialized - using existing Ray cluster")
            return True
            
    except Exception as e:
        print(f"⚠️  Ray initialization failed: {str(e)}")
        print("🔀 Falling back to serial processing...")
        return False

def _cleanup_ray():
    """Cleanup Ray resources if we initialized them."""
    try:
        if ray.is_initialized():
            ray.shutdown()
            print("🧹 Ray shutdown complete")
    except:
        pass  # Ignore cleanup errors

def reidentifyArc(shifted_arc, reference_linelist=os.path.join(CALIB_DIR,'llamas_ThAr_ref_arclines.fits')):
    """Re-identify arc lines using cross-correlation with reference spectrum.

    This function takes a shifted arc spectrum and re-identifies arc lines by 
    cross-correlating with a reference spectrum.

    Args:
        shifted_arc: Path to the shifted arc extraction pickle file.
        reference_linelist (str, optional): Path to the reference line list FITS file. 
            Defaults to the LLAMAS ThAr reference line list.

    Returns:
        tuple: A tuple containing the x and y arrays of the processed arc spectrum.
    """

    arcdict = extract.ExtractLlamas.loadExtraction(shifted_arc)
    arcspec = arcdict['extractions']
    metadata = arcdict['metadata']

    fits_ext = 18

    if (metadata[fits_ext]['channel'] == 'red'):
        reference_arc = os.path.join(CALIB_DIR,'llamasr_ThAr_reference.fits')
        linelist = Table.read(reference_linelist, 3)

    refarc = interpolateNaNs(Table.read(reference_arc)['ThAr'])
    refarc -= np.nanmedian(refarc)
    thisarc = interpolateNaNs(arcspec[fits_ext].counts[150])
    thisarc -= np.nanmedian(thisarc)

    success, shift, stretch, stretch2, _, _, _ = \
        xcorr_shift_stretch(thisarc, refarc, stretch_func='linear')
        
#    for i in range(100,200):
#        plt.plot(arcspec[fits_ext].xshift[i],arcspec[fits_ext].counts[i],'.',markersize=1)

    
    xin = arcspec[fits_ext].xshift.flatten()
    yin = arcspec[fits_ext].counts.flatten()
    indices = np.argsort(xin)
    xin = xin[indices]
    yin = yin[indices]
    yin[np.where(np.isnan(yin))] = 0
    # yin = interpolateNaNs(yin)
    plt.plot(xin, yin, '.', markersize=1, color='c')
    print(np.isnan(yin).any())

    sset = bspline.bspline(xin, bkspace=1.2)
    res, yfit = sset.fit(xin, yin, np.ones(len(xin)))
    xmodel = (np.arange(4096)/2).astype(float)
    ymodel = sset.value(xmodel)[0]
    plt.plot(xmodel, ymodel, color='b')

    for line in linelist['xpos_fiber150']:
        # plt.plot([line+bestlag,line+bestlag],[0,np.max(refarc)],'-',color='red', alpha=0.5)
        newx = line*stretch+shift
        plt.plot([newx,newx],[0,np.max(refarc)],'-',color='red', alpha=0.5)
    plt.show()

    return(xin, yin)


# This is a slow but criticla step to calculate the shift an stretch of each fiber
# relative to a reference fiber, which fiber #150 in spectrograph 4A, near the 
# center of the IFU.  

def shiftArcX_original(arc_extraction_pickle):
    """Calculate shift and stretch for each fiber relative to reference fiber (original serial version).

    This is a critical step to calculate the shift and stretch of each fiber
    relative to a reference fiber (fiber #150 in spectrograph 4A, near the 
    center of the IFU).

    Args:
        arc_extraction_pickle (str): Path to the arc extraction pickle file.

    Returns:
        None: The function modifies the arc extraction object and saves it with 
            '_shifted.pkl' suffix.
    """

    warnings.filterwarnings("ignore", category=UserWarning, module="astropy.stats.sigma_clipping")
    arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_pickle)
    arcspec = arcdict['extractions']
    metadata = arcdict['metadata']

    # We will use fiber #150 in spectrograph 4A as the reference (near the center of the IFU)
    # This corresponds to extension 18 (red), 19 (green) and 20 (blue) in the arc extraction object
    red_refspec   = interpolateNaNs(arcspec[18].counts[150])
    green_refspec = interpolateNaNs(arcspec[19].counts[150])
    blue_refspec  = interpolateNaNs(arcspec[20].counts[150])
    
    x = np.arange(2048)

    for fits_ext in range(len(arcspec)):

        if (metadata[fits_ext]['channel'] == 'red'):
            refspec = red_refspec
        elif (metadata[fits_ext]['channel'] == 'green'):
            refspec = green_refspec
        elif (metadata[fits_ext]['channel'] == 'blue'):
            refspec = blue_refspec

        for ifiber in range(0,metadata[fits_ext]['nfibers']):
            print(f"Shifting fiber number {ifiber} in extension {metadata[fits_ext]['bench']}{metadata[fits_ext]['side']} {metadata[fits_ext]['channel']}")
            func = 'quadratic'

            success, shift, stretch, stretch2, _, _, _ = \
                xcorr_shift_stretch(refspec, interpolateNaNs(arcspec[fits_ext].counts[ifiber]), \
                                    stretch_func=func)
            if (success == 1):
                arcspec[fits_ext].xshift[ifiber,:] = (x*stretch+x**2*stretch2)+shift
            else:
                print("....Warning: arc shift failed for this fiber!")
                arcspec[fits_ext].xshift[ifiber,:] = x

    # Re-save with new information populated
    sv = arc_extraction_pickle.replace('.pkl','_shifted.pkl')
    extract.save_extractions(arcspec, savefile=sv)

def shiftArcXRay(arc_extraction_pickle):
    """Ray-enabled version of shiftArcX for parallel processing.
    
    Calculate shift and stretch for each fiber relative to reference fiber using Ray multiprocessing.
    This function parallelizes the fiber-level cross-correlation calculations for significant speedup.
    
    Note: This function assumes Ray is already initialized.
    
    Args:
        arc_extraction_pickle (str): Path to the arc extraction pickle file.
        
    Returns:
        None: The function modifies the arc extraction object and saves it with '_shifted.pkl' suffix.
    """
    
    warnings.filterwarnings("ignore", category=UserWarning, module="astropy.stats.sigma_clipping")
    arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_pickle)
    arcspec = arcdict['extractions']
    metadata = arcdict['metadata']

    # We will use fiber #150 in spectrograph 4A as the reference (near the center of the IFU)
    red_refspec   = interpolateNaNs(arcspec[18].counts[150])
    green_refspec = interpolateNaNs(arcspec[19].counts[150])
    blue_refspec  = interpolateNaNs(arcspec[20].counts[150])
    
    # Prepare all fiber processing tasks
    tasks = []
    
    for fits_ext in range(len(arcspec)):
        if metadata[fits_ext]['channel'] == 'red':
            refspec = red_refspec
        elif metadata[fits_ext]['channel'] == 'green':
            refspec = green_refspec
        elif metadata[fits_ext]['channel'] == 'blue':
            refspec = blue_refspec

        for ifiber in range(0, metadata[fits_ext]['nfibers']):
            fiber_counts = arcspec[fits_ext].counts[ifiber]
            task = process_fiber_shift.remote(fits_ext, ifiber, refspec, fiber_counts, 'quadratic')
            tasks.append(task)

    print(f"Processing {len(tasks)} fiber shift calculations in parallel...")
    
    # Execute all tasks and collect results
    completed = 0
    total_tasks = len(tasks)
    
    while tasks:
        # Process completed tasks
        done_tasks, tasks = ray.wait(tasks, num_returns=min(50, len(tasks)))
        results = ray.get(done_tasks)
        
        # Apply results back to arcspec
        for fits_ext, ifiber, success, xshift_result in results:
            arcspec[fits_ext].xshift[ifiber, :] = xshift_result
            if not success:
                print(f"....Warning: arc shift failed for fiber {ifiber} in extension {fits_ext}")
        
        completed += len(results)
        if completed % 100 == 0 or completed == total_tasks:
            print(f"Completed {completed}/{total_tasks} fiber shift calculations")

    # Re-save with new information populated
    sv = arc_extraction_pickle.replace('.pkl','_shifted.pkl')
    extract.save_extractions(arcspec, savefile=sv)
    print(f"Saved shifted arc extraction to {sv}")

def fiberRelativeThroughput_original(flat_extraction_pickle, arc_extraction_pickle):
    """Calculate relative fiber throughput from flat field observations (original serial version).

    This function calculates the relative throughput of each fiber compared to 
    a reference fiber using flat field observations.

    Args:
        flat_extraction_pickle (str): Path to the flat field extraction pickle file.
        arc_extraction_pickle (str): Path to the arc extraction pickle file.

    Returns:
        None: The function modifies the arc extraction object and saves it with 
            '_shifted_tp.pkl' suffix.
    """

    flatdict = extract.ExtractLlamas.loadExtraction(flat_extraction_pickle)
    flatspec = flatdict['extractions']
    metadata = flatdict['metadata']

    arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_pickle)
    arcspec = arcdict['extractions']

    reference_fiber = 150

    for fits_ext in range(len(arcspec)):

        if (metadata[fits_ext]['channel'] == 'red'):
            ref_ext = 18
        elif (metadata[fits_ext]['channel'] == 'green'):
            ref_ext = 19
        elif (metadata[fits_ext]['channel'] == 'blue'):
            ref_ext = 20

        refspec = flatspec[ref_ext].counts[reference_fiber]
        gd = (arcspec[fits_ext].xshift[reference_fiber] > 150) & (arcspec[fits_ext].xshift[reference_fiber] < 2048-150)
        reference_flux = np.nansum(refspec[gd])

        for ifiber in range(0,metadata[fits_ext]['nfibers']):   
            spec = flatspec[fits_ext].counts[ifiber]
            gd = (arcspec[fits_ext].xshift[ifiber] > 150) & (arcspec[fits_ext].xshift[ifiber] < 2048-150)
            flux = np.nansum(spec[gd])
            arcspec[fits_ext].relative_throughput[ifiber] = flux/reference_flux
            print(f'{metadata[fits_ext]['bench']}{metadata[fits_ext]['side']} {metadata[fits_ext]['channel']} Fiber #{ifiber}:  Throughput = {flux/reference_flux:5.3f}')

    sv = arc_extraction_pickle.replace('.pkl','_shifted_tp.pkl')
    extract.save_extractions(arcspec, savefile=sv)

def fiberRelativeThroughputRay(flat_extraction_pickle, arc_extraction_pickle):
    """Ray-enabled version of fiberRelativeThroughput for parallel processing.
    
    Calculate relative fiber throughput from flat field observations using Ray multiprocessing.
    This function parallelizes the fiber-level throughput calculations.
    
    Note: This function assumes Ray is already initialized.
    
    Args:
        flat_extraction_pickle (str): Path to the flat field extraction pickle file.
        arc_extraction_pickle (str): Path to the arc extraction pickle file.
        
    Returns:
        None: The function modifies the arc extraction object and saves it with '_shifted_tp.pkl' suffix.
    """
    
    flatdict = extract.ExtractLlamas.loadExtraction(flat_extraction_pickle)
    flatspec = flatdict['extractions']
    metadata = flatdict['metadata']

    arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_pickle)
    arcspec = arcdict['extractions']

    reference_fiber = 150
    
    # Prepare all fiber processing tasks
    tasks = []
    
    for fits_ext in range(len(arcspec)):
        if metadata[fits_ext]['channel'] == 'red':
            ref_ext = 18
        elif metadata[fits_ext]['channel'] == 'green':
            ref_ext = 19
        elif metadata[fits_ext]['channel'] == 'blue':
            ref_ext = 20

        refspec = flatspec[ref_ext].counts[reference_fiber]
        gd = (arcspec[fits_ext].xshift[reference_fiber] > 150) & (arcspec[fits_ext].xshift[reference_fiber] < 2048-150)
        reference_flux = np.nansum(refspec[gd])

        for ifiber in range(0, metadata[fits_ext]['nfibers']):   
            fiber_spec = flatspec[fits_ext].counts[ifiber]
            xshift_fiber = arcspec[fits_ext].xshift[ifiber]
            task = process_fiber_throughput.remote(fits_ext, ifiber, fiber_spec, xshift_fiber, reference_flux)
            tasks.append(task)

    print(f"Processing {len(tasks)} fiber throughput calculations in parallel...")
    
    # Execute all tasks and collect results
    completed = 0
    total_tasks = len(tasks)
    
    while tasks:
        # Process completed tasks
        done_tasks, tasks = ray.wait(tasks, num_returns=min(50, len(tasks)))
        results = ray.get(done_tasks)
        
        # Apply results back to arcspec
        for fits_ext, ifiber, relative_throughput in results:
            arcspec[fits_ext].relative_throughput[ifiber] = relative_throughput
            print(f'{metadata[fits_ext]['bench']}{metadata[fits_ext]['side']} {metadata[fits_ext]['channel']} Fiber #{ifiber}:  Throughput = {relative_throughput:5.3f}')
        
        completed += len(results)
        if completed % 100 == 0 or completed == total_tasks:
            print(f"Completed {completed}/{total_tasks} fiber throughput calculations")

    sv = arc_extraction_pickle.replace('.pkl','_shifted_tp.pkl')
    extract.save_extractions(arcspec, savefile=sv)
    print(f"Saved throughput-corrected arc extraction to {sv}")


def arcSolve_original(arc_extraction_shifted_pickle, autoid=False, savefile='LLAMAS_reference_arc.pkl', savedir=OUTPUT_DIR):
    """Solve wavelength calibration from ThAr arc spectra (original serial version).

    This function fits wavelength solutions to ThAr arc spectra by identifying 
    arc lines and fitting polynomial wavelength solutions.

    Args:
        arc_extraction_shifted_pickle (str): Path to the shifted arc extraction pickle file.
        autoid (bool, optional): Whether to use automatic line identification. 
            Defaults to False.

    Returns:
        None: The function saves the wavelength solution to 'LLAMAS_reference_arc.pkl'.
    """
    
    print("Loading arc extraction")
    arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_shifted_pickle)
    arcspec_shifted = arcdict['extractions']

    print("Fitting wavelength solution")
    for channel in ['red', 'green', 'blue']:

        if (channel == 'red'):
            test_extension = 18
            line_table = Table.read(os.path.join(LUT_DIR, 'red_peaks.csv'))
        elif (channel == 'green'):
            test_extension = 19
            line_table = Table.read(os.path.join(LUT_DIR, 'green_peaks.csv'))
        elif (channel == 'blue'):
            test_extension = 20
            line_table = Table.read(os.path.join(LUT_DIR, 'blue_peaks.csv'))   
        metadata = arcdict['metadata'][test_extension]
        print(f"Processing {metadata['bench']}{metadata['side']} {metadata['channel']}")
        
        line_table = line_table[(line_table['Wavelength'] > 0)]
        initial_arcfit = robust_fit(line_table['Pixel'], (airtovac(line_table['Wavelength']*u.AA)).value, function='legendre', order=5, lower=5, upper=5, maxdev=5)
        print(f'Inital arcfit {initial_arcfit}')
        arc_fitx = np.array([])
        arc_fitw = np.array([])
        arc_fity = np.array([])

        nfib, npix = arcspec_shifted[test_extension].xshift.shape

        # Normalize out variations in fiber throughput and get ready to fit a bspline
        arcspec = arcspec_shifted[test_extension]
        for i in range(0,nfib):
            x = arcspec.xshift[i,:]
            y = arcspec.counts[i,:]/arcspec.relative_throughput[i]
            yoffset = np.nanmedian(y)
            arc_fitx = np.append(arc_fitx, x)
            arc_fitw = np.append(arc_fitw,initial_arcfit.eval(x))
            arc_fity = np.append(arc_fity, y-yoffset)
        
        # Sort before passing to bspline fit
        idx = np.argsort(arc_fitx)
        arc_fitx = arc_fitx[idx]
        arc_fity = arc_fity[idx]
        arc_fitw = arc_fitw[idx]

        saturated = (arc_fity > 60000)
        arc_fity[saturated] = 60000

        # Scatter plot of the raw pixels
        mask = ((saturated) | (np.isnan(arc_fity)))

        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 6),
            gridspec_kw={'height_ratios': [2, 1]})

        ax1.plot(arc_fitw[~mask], arc_fity[~mask], '.', markersize=0.5, color='k')
 
        # Fit the bspline model
        sset, outmask = iterfit(arc_fitx[~mask], arc_fity[~mask], maxiter=6, kwargs_bspline={'bkspace':0.5})

        xmodel = -60 + np.arange(8400)/4
        ymodel = sset.value(xmodel)[0]

        ax1.plot(initial_arcfit.eval(xmodel), ymodel, color='r')
       # ax1.plot(xmodel, ymodel, color='r')

        cont = np.quantile(ymodel, 0.05)

        peaks = detect_peaks(ymodel-cont, threshold=2, mph=50, mpd=2)
        pkht  = ymodel[peaks]

        for pk in peaks:
            ax1.plot([initial_arcfit.eval(xmodel[pk])], [ymodel[pk]*1.1], '+', color='c')
            # ax1.plot([xmodel[pk]], [ymodel[pk]*1.1], '+', color='c')  
    
        thar_lines = Table.read(os.path.join(LUT_DIR,'ThAr_MagE_lines.dat'), format='ascii.fixed_width')
        wv_minmax = {'blue': [3200.0,4947.0], 'green': [4570.0, 7100.0], 'red': [6570.0, 10038.0]}

        if (channel == 'green'):
            thar_lines = thar_lines[(thar_lines['wave'] > wv_minmax['green'][0]) & 
                                    (thar_lines['wave'] < wv_minmax['green'][1])]     
        elif (channel == 'blue'):
            thar_lines = thar_lines[(thar_lines['wave'] > wv_minmax['blue'][0]) & 
                                    (thar_lines['wave'] < wv_minmax['blue'][1])]
        elif (channel == 'red'): 
            thar_lines = thar_lines[(thar_lines['wave'] > wv_minmax['red'][0]) & 
                                    (thar_lines['wave'] < wv_minmax['red'][1])]
    
        final_fitx = np.array([])
        final_fitwv = np.array([])

        for pk in peaks:
            pkwv = initial_arcfit.eval(xmodel[pk])
            thar_match = thar_lines[np.where(np.abs(pkwv-thar_lines['wave']) < 0.5)]
            if (len(thar_match) == 1):
                ax1.plot([pkwv,pkwv],[-200,0],color='b', alpha=0.1)
                final_fitx = np.append(final_fitx, xmodel[pk])
                final_fitwv = np.append(final_fitwv, thar_match['wave'])
        plt.show()
        print(f"Found {len(final_fitx)} lines in the ThAr linelist")
        if (len(final_fitx) ==0):
            print(f"No lines found in ThAr linelist for this channel {channel}")
            continue
        final_arcfit = robust_fit(final_fitx, final_fitwv, function='legendre', order=5, lower=5, upper=5, maxdev=5)

        for thisline in final_fitwv:
            ax1.plot([thisline,thisline], [-200,0], color='blue', alpha=0.25)
    
        ax1.plot(initial_arcfit.eval(xmodel), ymodel, color='r')
        ax1.plot(initial_arcfit.eval(xmodel[peaks]), 1.1*ymodel[peaks], '+', color='c')
    
        ax2.scatter(final_fitx, final_fitwv-final_arcfit.eval(final_fitx), color='r', alpha=0.5)
        rms = np.std(final_fitwv-final_arcfit.eval(final_fitx))
        ax2.set_ylim([-2,2])
        ax2.set_title(f'RMS = {rms:.2f} A')
        ax2.plot([-50,2050], [0, 0], color='k', alpha=0.5)
        plt.show()

        # peak_table = Table([xmodel[peaks], ymodel[peaks]])

        print(f"Transferring arc solution to all fibers in LLAMAS {channel} channels...")
        for extension in range(len(arcspec_shifted)):
            if (arcdict['metadata'][extension]['channel'] == channel):
                for ifiber in range(0,arcdict['metadata'][extension]['nfibers']):
                    x = arcspec_shifted[extension].xshift[ifiber,:]
                    arcspec_shifted[extension].wave[ifiber,:] = final_arcfit.eval(x)

    # Save the wavelength solution to disk
    print("Saving wavelength solution to disk")
    extract.save_extractions(arcspec_shifted, savefile=savefile, savedir=savedir)
    return()

def arcSolveRay(arc_extraction_shifted_pickle, autoid=False, savefile='LLAMAS_reference_arc.pkl', savedir=OUTPUT_DIR):
    """Ray-enabled version of arcSolve for parallel processing.
    
    Solve wavelength calibration from ThAr arc spectra using Ray multiprocessing.
    This function parallelizes the wavelength solution transfer to different extensions.
    
    Note: This function assumes Ray is already initialized.
    
    Args:
        arc_extraction_shifted_pickle (str): Path to the shifted arc extraction pickle file.
        autoid (bool, optional): Whether to use automatic line identification. Defaults to False.
        
    Returns:
        None: The function saves the wavelength solution to 'LLAMAS_reference_arc.pkl'.
    """
    
    print("Loading arc extraction")
    arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_shifted_pickle)
    arcspec_shifted = arcdict['extractions']

    print("Fitting wavelength solution")
    for channel in ['red', 'green', 'blue']:

        if (channel == 'red'):
            test_extension = 18
            line_table = Table.read(os.path.join(LUT_DIR, 'red_peaks.csv'))
        elif (channel == 'green'):
            test_extension = 19
            line_table = Table.read(os.path.join(LUT_DIR, 'green_peaks.csv'))
        elif (channel == 'blue'):
            test_extension = 20
            line_table = Table.read(os.path.join(LUT_DIR, 'blue_peaks.csv'))   
        metadata = arcdict['metadata'][test_extension]
        print(f"Processing {metadata['bench']}{metadata['side']} {metadata['channel']}")
        
        line_table = line_table[(line_table['Wavelength'] > 0)]
        initial_arcfit = robust_fit(line_table['Pixel'], (airtovac(line_table['Wavelength']*u.AA)).value, function='legendre', order=5, lower=5, upper=5, maxdev=5)
        print(f'Inital arcfit {initial_arcfit}')
        arc_fitx = np.array([])
        arc_fitw = np.array([])
        arc_fity = np.array([])

        nfib, npix = arcspec_shifted[test_extension].xshift.shape

        # Normalize out variations in fiber throughput and get ready to fit a bspline
        arcspec = arcspec_shifted[test_extension]
        for i in range(0,nfib):
            x = arcspec.xshift[i,:]
            y = arcspec.counts[i,:]/arcspec.relative_throughput[i]
            yoffset = np.nanmedian(y)
            arc_fitx = np.append(arc_fitx, x)
            arc_fitw = np.append(arc_fitw,initial_arcfit.eval(x))
            arc_fity = np.append(arc_fity, y-yoffset)
        
        # Sort before passing to bspline fit
        idx = np.argsort(arc_fitx)
        arc_fitx = arc_fitx[idx]
        arc_fity = arc_fity[idx]
        arc_fitw = arc_fitw[idx]

        saturated = (arc_fity > 60000)
        arc_fity[saturated] = 60000

        # Scatter plot of the raw pixels
        mask = ((saturated) | (np.isnan(arc_fity)))

        fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 6),
            gridspec_kw={'height_ratios': [2, 1]})

        ax1.plot(arc_fitw[~mask], arc_fity[~mask], '.', markersize=0.5, color='k')
 
        # Fit the bspline model
        sset, outmask = iterfit(arc_fitx[~mask], arc_fity[~mask], maxiter=6, kwargs_bspline={'bkspace':0.5})

        xmodel = -60 + np.arange(8400)/4
        ymodel = sset.value(xmodel)[0]

        ax1.plot(initial_arcfit.eval(xmodel), ymodel, color='r')
       # ax1.plot(xmodel, ymodel, color='r')

        cont = np.quantile(ymodel, 0.05)

        peaks = detect_peaks(ymodel-cont, threshold=2, mph=50, mpd=2)
        pkht  = ymodel[peaks]

        for pk in peaks:
            ax1.plot([initial_arcfit.eval(xmodel[pk])], [ymodel[pk]*1.1], '+', color='c')
            # ax1.plot([xmodel[pk]], [ymodel[pk]*1.1], '+', color='c')  
    
        thar_lines = Table.read(os.path.join(LUT_DIR,'ThAr_MagE_lines.dat'), format='ascii.fixed_width')
        wv_minmax = {'blue': [3200.0,4947.0], 'green': [4570.0, 7100.0], 'red': [6570.0, 10038.0]}

        if (channel == 'green'):
            thar_lines = thar_lines[(thar_lines['wave'] > wv_minmax['green'][0]) & 
                                    (thar_lines['wave'] < wv_minmax['green'][1])]     
        elif (channel == 'blue'):
            thar_lines = thar_lines[(thar_lines['wave'] > wv_minmax['blue'][0]) & 
                                    (thar_lines['wave'] < wv_minmax['blue'][1])]
        elif (channel == 'red'): 
            thar_lines = thar_lines[(thar_lines['wave'] > wv_minmax['red'][0]) & 
                                    (thar_lines['wave'] < wv_minmax['red'][1])]
    
        final_fitx = np.array([])
        final_fitwv = np.array([])

        for pk in peaks:
            pkwv = initial_arcfit.eval(xmodel[pk])
            thar_match = thar_lines[np.where(np.abs(pkwv-thar_lines['wave']) < 0.5)]
            if (len(thar_match) == 1):
                ax1.plot([pkwv,pkwv],[-200,0],color='b', alpha=0.1)
                final_fitx = np.append(final_fitx, xmodel[pk])
                final_fitwv = np.append(final_fitwv, thar_match['wave'])
        plt.show()
        print(f"Found {len(final_fitx)} lines in the ThAr linelist")
        if (len(final_fitx) ==0):
            print(f"No lines found in ThAr linelist for this channel {channel}")
            continue
        final_arcfit = robust_fit(final_fitx, final_fitwv, function='legendre', order=5, lower=5, upper=5, maxdev=5)

        for thisline in final_fitwv:
            ax1.plot([thisline,thisline], [-200,0], color='blue', alpha=0.25)
    
        ax1.plot(initial_arcfit.eval(xmodel), ymodel, color='r')
        ax1.plot(initial_arcfit.eval(xmodel[peaks]), 1.1*ymodel[peaks], '+', color='c')
    
        ax2.scatter(final_fitx, final_fitwv-final_arcfit.eval(final_fitx), color='r', alpha=0.5)
        rms = np.std(final_fitwv-final_arcfit.eval(final_fitx))
        ax2.set_ylim([-2,2])
        ax2.set_title(f'RMS = {rms:.2f} A')
        ax2.plot([-50,2050], [0, 0], color='k', alpha=0.5)
        plt.show()

        # peak_table = Table([xmodel[peaks], ymodel[peaks]])

        print(f"Transferring arc solution to all fibers in LLAMAS {channel} channels using Ray...")
        
        # Prepare extension data for Ray processing
        extension_tasks = []
        for extension in range(len(arcspec_shifted)):
            if (arcdict['metadata'][extension]['channel'] == channel):
                extension_data = {
                    'nfibers': arcdict['metadata'][extension]['nfibers'],
                    'xshift': arcspec_shifted[extension].xshift
                }
                task = process_extension_wavelength_transfer.remote(extension, extension_data, final_arcfit, channel)
                extension_tasks.append(task)

        # Process all extensions for this channel in parallel
        if extension_tasks:
            print(f"Processing {len(extension_tasks)} extensions for {channel} channel in parallel...")
            results = ray.get(extension_tasks)
            
            # Apply results back to arcspec_shifted
            for extension_idx, wavelength_data in results:
                for ifiber, wave_solution in wavelength_data.items():
                    arcspec_shifted[extension_idx].wave[ifiber, :] = wave_solution

    # Save the wavelength solution to disk
    print("Saving wavelength solution to disk")
    extract.save_extractions(arcspec_shifted, savefile=savefile, savedir=savedir)
    return()

# Smart wrapper functions that use Ray by default with fallback to serial

def shiftArcX(arc_extraction_pickle, use_ray=True):
    """Calculate shift and stretch for each fiber relative to reference fiber.
    
    This function automatically uses Ray multiprocessing for significant speedup.
    Falls back to serial processing if Ray initialization fails.

    Args:
        arc_extraction_pickle (str): Path to the arc extraction pickle file.
        use_ray (bool): Whether to use Ray multiprocessing. Defaults to True.

    Returns:
        None: The function modifies the arc extraction object and saves it with 
            '_shifted.pkl' suffix.
    """
    if use_ray:
        # Try Ray processing first
        ray_available = _initialize_ray_if_needed()
        if ray_available:
            try:
                print("🔥 Using Ray multiprocessing for arc shift calculation...")
                return shiftArcXRay(arc_extraction_pickle)
            except Exception as e:
                print(f"⚠️  Ray processing failed: {str(e)}")
                print("🔀 Falling back to serial processing...")
                use_ray = False
    
    if not use_ray:
        print("🐌 Using serial processing for arc shift calculation...")
        return shiftArcX_original(arc_extraction_pickle)

def fiberRelativeThroughput(flat_extraction_pickle, arc_extraction_pickle, use_ray=True):
    """Calculate relative fiber throughput from flat field observations.
    
    This function automatically uses Ray multiprocessing for significant speedup.
    Falls back to serial processing if Ray initialization fails.

    Args:
        flat_extraction_pickle (str): Path to the flat field extraction pickle file.
        arc_extraction_pickle (str): Path to the arc extraction pickle file.
        use_ray (bool): Whether to use Ray multiprocessing. Defaults to True.

    Returns:
        None: The function modifies the arc extraction object and saves it with 
            '_shifted_tp.pkl' suffix.
    """
    if use_ray:
        # Try Ray processing first
        ray_available = _initialize_ray_if_needed()
        if ray_available:
            try:
                print("🔥 Using Ray multiprocessing for fiber throughput calculation...")
                return fiberRelativeThroughputRay(flat_extraction_pickle, arc_extraction_pickle)
            except Exception as e:
                print(f"⚠️  Ray processing failed: {str(e)}")
                print("🔀 Falling back to serial processing...")
                use_ray = False
    
    if not use_ray:
        print("🐌 Using serial processing for fiber throughput calculation...")
        return fiberRelativeThroughput_original(flat_extraction_pickle, arc_extraction_pickle)

def arcSolve(arc_extraction_shifted_pickle, autoid=False, use_ray=True, savefile='LLAMAS_reference_arc.pkl', savedir=OUTPUT_DIR):
    """Solve wavelength calibration from ThAr arc spectra.
    
    This function automatically uses Ray multiprocessing for significant speedup.
    Falls back to serial processing if Ray initialization fails.

    Args:
        arc_extraction_shifted_pickle (str): Path to the shifted arc extraction pickle file.
        autoid (bool, optional): Whether to use automatic line identification. Defaults to False.
        use_ray (bool): Whether to use Ray multiprocessing. Defaults to True.

    Returns:
        None: The function saves the wavelength solution to 'LLAMAS_reference_arc.pkl'.
    """
    if use_ray:
        # Try Ray processing first
        ray_available = _initialize_ray_if_needed()
        if ray_available:
            try:
                print("🔥 Using Ray multiprocessing for arc wavelength solution...")
                return arcSolveRay(arc_extraction_shifted_pickle, autoid, savedir=savedir, savefile=savefile)
            except Exception as e:
                print(f"⚠️  Ray processing failed: {str(e)}")
                print("🔀 Falling back to serial processing...")
                use_ray = False
    
    if not use_ray:
        print("🐌 Using serial processing for arc wavelength solution...")
        return arcSolve_original(arc_extraction_shifted_pickle, autoid, savedir=savedir, savefile=savefile)

def arcTransfer(scidict, arcdict):
    """Transfer wavelength calibration from arc to science spectra.

    This function transfers the wavelength solution, x-shift information, and 
    relative throughput data from arc calibration spectra to science spectra.

    Args:
        scidict (dict): Dictionary containing science extraction data.
        arcdict (dict): Dictionary containing arc extraction data with wavelength solution.

    Returns:
        dict: Updated science dictionary with transferred calibration data.
    """
    from llamas_pyjamas.constants import idx_lookup

    scispec = scidict['extractions']
    arcspec = arcdict['extractions']

    # Loop over the extensions
    for fits_ext in range(len(scispec)):
        # Get channel, bench, and side from metadata
        channel = scidict['metadata'][fits_ext]['channel']
        bench = str(scidict['metadata'][fits_ext]['bench'])
        side = scidict['metadata'][fits_ext]['side']
        
        key = (channel, bench, side)

        # Use the lookup table to get the correct arc extension index
        arc_idx = idx_lookup[key] -1

        sci_meta_channel, sci_meta_bench, sci_meta_side = scidict['metadata'][fits_ext]['channel'], str(scidict['metadata'][fits_ext]['bench']), scidict['metadata'][fits_ext]['side']
        arc_meta_channel, arc_meta_bench, arc_meta_side = arcdict['metadata'][arc_idx]['channel'], str(arcdict['metadata'][arc_idx]['bench']), arcdict['metadata'][arc_idx]['side']

        if (sci_meta_channel != arc_meta_channel) or (sci_meta_bench != arc_meta_bench) or (sci_meta_side != arc_meta_side):
            print(f"Error: Metadata mismatch between science and arc for extension {fits_ext}")
            print(f"Science metadata: Channel={sci_meta_channel}, Bench={sci_meta_bench}, Side={sci_meta_side}")
            print(f"Arc metadata: Channel={arc_meta_channel}, Bench={arc_meta_bench}, Side={arc_meta_side}")
            continue

        
        # Get number of fibers in both science and arc spectra
        sci_nfibers = scidict['metadata'][fits_ext]['nfibers']
        arc_nfibers = arcdict['metadata'][arc_idx]['nfibers']
        if sci_nfibers != arc_nfibers:
            print(f"Warning: Number of fibers mismatch for {key} - Science: {sci_nfibers}, Arc: {arc_nfibers}")
            ### add in comparison of metadata here as I need to check the index matching is correct with the new fix

        # Use the minimum number of fibers to avoid index errors
        min_nfibers = min(sci_nfibers, arc_nfibers)
        
        if sci_nfibers != arc_nfibers:
            print(f"Warning: Number of fibers mismatch for {key} - Science: {sci_nfibers}, Arc: {arc_nfibers}")
            print(f"Using the first {min_nfibers} fibers for calibration transfer")
        
        # Loop over the fibers (only up to the minimum number present in both)
        for ifiber in range(min_nfibers):
            x = scispec[fits_ext].xshift[ifiber,:]
            scispec[fits_ext].wave[ifiber,:] = arcspec[arc_idx].wave[ifiber,:]
            scispec[fits_ext].xshift[ifiber,:] = arcspec[arc_idx].xshift[ifiber,:]
            scispec[fits_ext].relative_throughput[ifiber] = arcspec[arc_idx].relative_throughput[ifiber]

    return(scidict)

def run_wavelength_solution_ray(arc_filename: str, flat_filename: str, 
                               data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR) -> str:
    """
    Execute the complete wavelength solution pipeline using Ray multiprocessing.
    
    This is the main orchestration function that runs the entire wavelength solution
    process with Ray parallelization, following the exact workflow provided by the user.
    
    Args:
        arc_filename (str): Path to the arc FITS file
        flat_filename (str): Path to the flat field FITS file  
        data_dir (str): Directory containing input files
        output_dir (str): Directory for output files
        
    Returns:
        str: Path to the final wavelength solution file
    """
    
    # Initialize Ray with core management
    NUMBER_OF_CORES = int(os.environ.get('LLAMAS_RAY_CPUS', multiprocessing.cpu_count()))
    
    print(f"\n=== Starting LLAMAS Wavelength Solution Pipeline with Ray ===")
    print(f"Using {NUMBER_OF_CORES} CPU cores for parallel processing")
    print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # Initialize Ray
    ray.shutdown()  # Clear any existing Ray instances
    ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    
    try:
        start_time = time.time()
        
        # Step 1: Setup file paths following user's workflow
        arc_path = os.path.join(data_dir, arc_filename)
        flat_path = os.path.join(data_dir, flat_filename)
        
        print(f"Arc file: {arc_path}")
        print(f"Flat file: {flat_path}")
        
        # Generate extraction pickle filename  
        arc_picklename = os.path.join(output_dir, os.path.basename(arc_filename).replace('_mef.fits', '_extract.pkl'))
        flat_picklename = os.path.join(output_dir, os.path.basename(flat_filename).replace('_mef.fits', '_extract.pkl'))
        
        print(f"Arc extraction pickle: {arc_picklename}")
        print(f"Flat extraction pickle: {flat_picklename}")
        
        # Check if extraction files exist (assuming they've been created by extract step)
        if not os.path.exists(arc_picklename):
            raise FileNotFoundError(f"Arc extraction file not found: {arc_picklename}")
        if not os.path.exists(flat_picklename):
            raise FileNotFoundError(f"Flat extraction file not found: {flat_picklename}")
            
        # Step 2: Ray-enabled arc shift calculation
        print("\n=== Step 1: Arc Shift Calculation (Ray Parallel) ===")
        shiftArcXRay(arc_picklename)
        
        shift_picklename = arc_picklename.replace('_extract.pkl', '_extract_shifted.pkl')
        print(f"Shifted arc pickle: {shift_picklename}")
        
        # Step 3: Ray-enabled fiber relative throughput calculation
        print("\n=== Step 2: Fiber Relative Throughput (Ray Parallel) ===")
        fiberRelativeThroughputRay(flat_picklename, shift_picklename)
        
        tp_picklename = shift_picklename.replace('.pkl','_shifted_tp.pkl')
        print(f"Throughput-corrected pickle: {tp_picklename}")
        
        # Step 4: Ray-enabled arc wavelength solution
        print("\n=== Step 3: Arc Wavelength Solution (Ray Parallel) ===")
        arcSolveRay(tp_picklename)
        
        final_solution_path = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        print(f"\n=== Wavelength Solution Pipeline Complete ===")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Final wavelength solution saved to: {final_solution_path}")
        print(f"Final CPU Usage: {psutil.cpu_percent(percpu=True)}%")
        
        return final_solution_path
        
    except Exception as e:
        print(f"Error in wavelength solution pipeline: {str(e)}")
        raise
    finally:
        # Always shutdown Ray
        ray.shutdown()
        print("Ray shutdown complete")

def run_wavelength_solution_workflow(arc_filename: str, flat_filename: str, 
                                   data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR,
                                   use_ray: bool = True) -> str:
    """
    Convenience function that runs either Ray-enabled or standard wavelength solution workflow.
    
    This function matches the exact user workflow provided:
    ```
    arc.shiftArcX(arc_picklename)
    shift_picklename = arc_picklename.replace('_extract.pkl', '_extract_shifted.pkl')
    arc.fiberRelativeThroughput(flat_picklename, shift_picklename)
    tp = shift_picklename.replace('.pkl','_shifted_tp.pkl')
    arc.arcSolve(tp)
    ```
    
    Args:
        arc_filename (str): Path to the arc FITS file  
        flat_filename (str): Path to the flat field FITS file
        data_dir (str): Directory containing input files
        output_dir (str): Directory for output files
        use_ray (bool): Whether to use Ray multiprocessing (default: True)
        
    Returns:
        str: Path to the final wavelength solution file
    """
    
    if use_ray:
        return run_wavelength_solution_ray(arc_filename, flat_filename, data_dir, output_dir)
    else:
        # Standard workflow using original functions
        arc_path = os.path.join(data_dir, arc_filename)
        arc_picklename = os.path.join(output_dir, os.path.basename(arc_filename).replace('_mef.fits', '_extract.pkl'))
        flat_picklename = os.path.join(output_dir, os.path.basename(flat_filename).replace('_mef.fits', '_extract.pkl'))
        
        # Original workflow
        shiftArcX(arc_picklename)
        shift_picklename = arc_picklename.replace('_extract.pkl', '_extract_shifted.pkl')
        fiberRelativeThroughput(flat_picklename, shift_picklename) 
        tp = shift_picklename.replace('.pkl','_shifted_tp.pkl')
        arcSolve(tp)
        
        return os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')