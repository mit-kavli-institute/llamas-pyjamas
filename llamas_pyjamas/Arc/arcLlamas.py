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

###############################################################################3

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

def shiftArcX(arc_extraction_pickle):
    """Calculate shift and stretch for each fiber relative to reference fiber.

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

def fiberRelativeThroughput(flat_extraction_pickle, arc_extraction_pickle):
    """Calculate relative fiber throughput from flat field observations.

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


def arcSolve(arc_extraction_shifted_pickle, autoid=False):
    """Solve wavelength calibration from ThAr arc spectra.

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
    extract.save_extractions(arcspec_shifted, savefile='LLAMAS_reference_arc.pkl')
    return()

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