import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.Extract.extractLlamas import save_extractions
import llamas_pyjamas.Arc.arcLlamas as arc
from llamas_pyjamas.QA import plot_ds9
from llamas_pyjamas.config import OUTPUT_DIR, CALIB_DIR
from pypeit.core.fitting import iterfit, robust_fit
from pypeit.core.wavecal.wvutils import arc_lines_from_spec
from astropy.io import fits
from astropy.table import Table
import os
import warnings

def refineSkyX(science_extraction_file, channels=None, ref_fiber=150, fiber_half_width=10,
               sigdetect=5.0, fwhm=4.0, match_tol=5.0, min_lines=4, poly_order=1):
    """Refine per-fiber xshift using sky line centroids detected on the science frame.

    Builds a high-S/N sky template from the reference fiber ± fiber_half_width fibers
    for each camera, detects sky line peaks in xshift space, then fits a low-order
    polynomial correction (shift + stretch at order=1) per fiber that maps its current
    xshift onto the template coordinate system.  Both xshift and wave are updated
    consistently so the correction flows into the sky subtraction routines that follow.

    Parameters
    ----------
    science_extraction_file : str
        Path to the science extraction pickle (post-arcTransfer, with xshift and wave
        already populated).
    channels : list of str, optional
        Channels to process, e.g. ['green'].  Default: all three.
    ref_fiber : int
        Central fiber used to build the per-camera sky template (default 150).
    fiber_half_width : int
        Number of fibers either side of ref_fiber included in the template (default 10,
        giving 21 fibers total).
    sigdetect : float
        Detection threshold in sigma for arc_lines_from_spec (default 5.0).
    fwhm : float
        Expected line FWHM in pixels (default 4.0).
    match_tol : float
        Maximum xshift distance (pixels) allowed when matching fiber peaks to template
        peaks (default 5.0).
    min_lines : int
        Minimum matched lines required to attempt a fit; fiber falls back to no
        correction otherwise (default 4).
    poly_order : int
        Polynomial order for the xshift correction fit.  1 = shift + stretch (default).

    Returns
    -------
    str
        Path to the output pickle with corrected xshift and wave (*_skyX.pkl).
    """

    REF_EXT = {'red': 18, 'green': 19, 'blue': 20}
    all_channels = channels if channels is not None else ['red', 'green', 'blue']

    scidict  = ExtractLlamas.loadExtraction(science_extraction_file)
    scispec  = scidict['extractions']
    metadata = scidict['metadata']
    hdr      = scidict['primary_header']

    x = np.arange(2048, dtype=float)

    for channel in all_channels:

        ref_ext  = REF_EXT[channel]
        # Reference fiber lookup: xshift → wave for back-computing corrected wave
        xshift_ref = scispec[ref_ext].xshift[ref_fiber, :]
        wave_ref   = scispec[ref_ext].wave[ref_fiber, :]
        sort_idx   = np.argsort(xshift_ref)
        xshift_ref_s = xshift_ref[sort_idx]
        wave_ref_s   = wave_ref[sort_idx]

        for fits_ext in range(len(scispec)):
            if metadata[fits_ext]['channel'] != channel:
                continue

            nfibers = metadata[fits_ext]['nfibers']
            bench   = metadata[fits_ext]['bench']
            side    = metadata[fits_ext]['side']

            # ---- Build template from ref_fiber ± fiber_half_width ----
            t_lo  = max(0, ref_fiber - fiber_half_width)
            t_hi  = min(nfibers, ref_fiber + fiber_half_width + 1)
            template_xshift = np.array([])
            template_counts = np.array([])

            for tf in range(t_lo, t_hi):
                xs = scispec[fits_ext].xshift[tf, :]
                ct = np.nan_to_num(scispec[fits_ext].counts[tf, :], nan=0.0,
                                   posinf=0.0, neginf=0.0)
                template_xshift = np.append(template_xshift, xs)
                template_counts = np.append(template_counts, ct)

            # Sort by xshift and bin onto a uniform grid via median in each bin
            sort_t  = np.argsort(template_xshift)
            txs     = template_xshift[sort_t]
            tct     = template_counts[sort_t]
            # Uniform xshift grid spanning the template range
            xs_min, xs_max = txs[0], txs[-1]
            n_grid  = 2048
            grid    = np.linspace(xs_min, xs_max, n_grid)
            binned  = np.interp(grid, txs, tct)

            # Detect sky line peaks in the template
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All pixels rejected")
                warnings.filterwarnings("ignore", message=".*invalid values.*")
                warnings.filterwarnings("ignore", category=UserWarning,
                                        module="astropy.stats")
                tmpl_tcent, _, _, _, _ = arc_lines_from_spec(
                    binned, sigdetect=sigdetect, fwhm=fwhm)

            if len(tmpl_tcent) == 0:
                print(f"  {bench}{side} {channel}: no sky lines found in template — skipping")
                continue

            # Convert template centroids from grid indices to xshift values
            ref_xshift_peaks = np.interp(tmpl_tcent, np.arange(n_grid, dtype=float), grid)
            print(f"  {bench}{side} {channel}: {len(ref_xshift_peaks)} template sky lines, "
                  f"processing {nfibers} fibers")

            n_refined  = 0
            n_fallback = 0

            for ifiber in range(nfibers):
                spec        = np.nan_to_num(scispec[fits_ext].counts[ifiber, :],
                                            nan=0.0, posinf=0.0, neginf=0.0)
                quad_xshift = scispec[fits_ext].xshift[ifiber, :]

                # Detect peaks in this fiber
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="All pixels rejected")
                        warnings.filterwarnings("ignore", message=".*invalid values.*")
                        warnings.filterwarnings("ignore", category=UserWarning,
                                                module="astropy.stats")
                        fib_tcent, fib_ecent, _, _, _ = arc_lines_from_spec(
                            spec, sigdetect=sigdetect, fwhm=fwhm)
                except Exception as e:
                    n_fallback += 1
                    continue

                if len(fib_tcent) == 0:
                    n_fallback += 1
                    continue

                # Convert fiber centroids to xshift values
                fib_xshift_peaks = np.interp(fib_tcent, x, quad_xshift)

                # Match fiber peaks to template peaks in xshift space
                matched_fib  = []
                matched_ref  = []
                matched_wts  = []

                for xsh_ref_pk in ref_xshift_peaks:
                    dists   = np.abs(fib_xshift_peaks - xsh_ref_pk)
                    nearest = np.argmin(dists)
                    if dists[nearest] < match_tol:
                        matched_fib.append(fib_xshift_peaks[nearest])
                        matched_ref.append(xsh_ref_pk)
                        ecent = max(fib_ecent[nearest], 1e-4)
                        matched_wts.append(1.0 / ecent)

                if len(matched_fib) < min_lines:
                    n_fallback += 1
                    continue

                mf  = np.array(matched_fib)
                mr  = np.array(matched_ref)
                mw  = np.array(matched_wts)

                try:
                    correction = robust_fit(mf, mr, function='legendre', order=poly_order,
                                            weights=mw, lower=5, upper=5)
                    scispec[fits_ext].xshift[ifiber, :] = correction.eval(quad_xshift)
                    # Recompute wave consistently from the corrected xshift
                    scispec[fits_ext].wave[ifiber, :] = np.interp(
                        scispec[fits_ext].xshift[ifiber, :], xshift_ref_s, wave_ref_s)
                    n_refined += 1
                except Exception as e:
                    n_fallback += 1

            print(f"    refined {n_refined}, fallback {n_fallback}")

    outputfile = science_extraction_file.replace('_extractions.pkl', '_skyX_extractions.pkl')
    print(f"Saving sky-line-refined extraction to {outputfile}")
    save_extractions(scispec, primary_header=hdr, savefile=outputfile)
    return outputfile


def skyModel_1d(science_extraction_file, color, sky_extraction_file=None, show_plots=False):
    """
    Create a 1D sky model from the sky extraction.

    Parameters:
    science_extraction (ExtractLlamas): Extracted science data.
    sky_extraction (ExtractLlamas): Extracted data from a blank sky reference field

    Note that as a default, the code assumes that the user estimates the sky from 
    the science field itself. If a separate sky field is provided, it will be used instead.

    Either way, the user must generate a mask to exclude sources from the sky estimation.

    Returns:
    sky_model (np.ndarray): 1D sky model.
    """

    if sky_extraction_file is None:
        sky_extraction_file = science_extraction_file 

    print("Loading science extraction from ", science_extraction_file)
    science_dict = ExtractLlamas.loadExtraction(science_extraction_file)
    science = science_dict['extractions']
    science_metadata = science_dict['metadata']
    hdr = science_dict['primary_header']

    print("Loading sky extraction and arc from ", sky_extraction_file)
    if (sky_extraction_file == science_extraction_file):
        sky_dict = science_dict
    else:
        sky_dict = ExtractLlamas.loadExtraction(sky_extraction_file)

    arc_dict = ExtractLlamas.loadExtraction(os.path.join(OUTPUT_DIR, 'LLAMAS_reference_arc.pkl'))
    sky_wvcal = arc.arcTransfer(sky_dict, arc_dict)
    
    sky = sky_wvcal['extractions']
    sky_metadata = sky_wvcal['metadata']

    # Find the brightest fibers in the std star spectrum



    ########## CORE ROUTINE ##########################

    if (color == None):
        allcolors = ['red','green','blue']
    else:
        allcolors = [color]

    # Build list of unique cameras (channel, bench, side) filtered to requested colors
    seen = set()
    cameras = []
    for i in range(len(sky)):
        ch = sky_metadata[i]['channel']
        if ch not in allcolors:
            continue
        key = (ch, str(sky_metadata[i]['bench']), sky_metadata[i]['side'])
        if key not in seen:
            seen.add(key)
            cameras.append((key, i))  # store first extension index for this camera

    for (channel, bench, side), _ in cameras:

        extension = np.array([])
        fiber = np.array([])
        counts = np.array([])

        print(f"Generating sky model for camera {channel} bench {bench}{side}")

        # Collect all fibers from this camera only
        for i in range(len(sky)):
            if (sky_metadata[i]['channel'] == channel and
                    str(sky_metadata[i]['bench']) == bench and
                    sky_metadata[i]['side'] == side):
                for thisfiber in range(sky_metadata[i]['nfibers']):
                    fiber = np.append(fiber, thisfiber)
                    extension = np.append(extension, i)
                    tt = np.sum(sky[i].counts[thisfiber]/sky[i].relative_throughput[thisfiber])
                    counts = np.append(counts, tt)

        # Sort in descending order of brightness
        idx = np.argsort(-counts)
        extension = extension[idx].astype(int)
        fiber = fiber[idx].astype(int)
        counts = counts[idx]

        # Use the middle third of fibers by brightness as sky fibers (avoid brightest
        # which may have object flux, and dimmest which are noisy)
        n_fibers = len(counts)
        sky_start = n_fibers // 3
        sky_end   = 2 * n_fibers // 3

        sky_fitx = np.array([])
        sky_fity = np.array([])

        for i in range(sky_start, sky_end):
            tp = sky[extension[i]].relative_throughput[fiber[i]]
            if not np.isfinite(tp) or tp <= 0:
                tp = 1.0
            sky_fitx = np.append(sky_fitx, sky[extension[i]].xshift[fiber[i],:])
            sky_fity = np.append(sky_fity, sky[extension[i]].counts[fiber[i],:] / tp)

        # Re-sort in order of increasing wavelength
        idx = np.argsort(sky_fitx)
        sky_fitx = sky_fitx[idx]
        sky_fity = sky_fity[idx]

        # Filter out bad pixels before the fit
        gd = (~np.isnan(sky_fity))
        sky_fitx = sky_fitx[gd]
        sky_fity = sky_fity[gd]

        xshift_min, xshift_max = sky_fitx.min(), sky_fitx.max()
        print(f"  xshift range: {xshift_min:.2f} to {xshift_max:.2f}")
        if xshift_max - xshift_min < 1.0:
            print(f"  ERROR: Degenerate xshift for camera {channel} bench {bench}{side} — arcTransfer likely skipped this extension due to a metadata mismatch. Exiting.")
            import sys; sys.exit(1)

        print(f"  Fitting sky with {len(sky_fitx)} points from {sky_end - sky_start} fibers")
        sset, outmask = iterfit(sky_fitx, sky_fity, maxiter=6, kwargs_bspline={'bkspace':0.5})

        if show_plots:
            plt.plot(sky_fitx, sky_fity, '.', markersize=0.1, label='data', color='k')
            y = sset.value(sky_fitx)[0]
            plt.plot(sky_fitx, y, color='r')
            plt.ylim(0,1000)
            plt.title(f'Sky fit: {channel} {bench}{side}')
            plt.show()

        # Apply sky model to all fibers in this camera.
        # The bspline was fit in throughput-corrected units (sky photons), so scale
        # back by each fiber's throughput to match raw counts in obj.counts.
        print(f"  Applying sky model to {n_fibers} fibers")
        for i in range(n_fibers):
            skymodel = sset.value(sky[extension[i]].xshift[fiber[i],:])[0]
            tp = science[extension[i]].relative_throughput[fiber[i]]
            if not np.isfinite(tp) or tp <= 0:
                tp = 1.0
            science[extension[i]].sky[fiber[i],:] = skymodel * tp

    #########################################


    # Plot some QA - show a few fibers from the middle of the last camera processed
    try:
        if show_plots:
            qa_indices = range(sky_start, min(sky_start + 5, sky_end))
            for i in qa_indices:
                plt.plot(sky[extension[i]].wave[fiber[i],:], sky[extension[i]].counts[fiber[i],:], '.', markersize=0.5, label='data', color='k')
                plt.plot(sky[extension[i]].wave[fiber[i],:], science[extension[i]].sky[fiber[i],:], color='r')
                plt.ylim(0, np.nanmax(sky[extension[i]].counts[fiber[i],:])*1.2)
                plt.title('Ext '+str(extension[i])+' Fiber '+str(fiber[i]))
                plt.xlabel('Wavelength (Angstroms)')
                plt.ylabel('Counts')
                plt.show()

            """ 
                skymodel = sset.value(sky_fitx)[0]
                #plt.plot(sky_fitx, sky_fity-skymodel, '.', markersize=0.1, label='std', color='k')
                #plt.plot(sky_fitx, np.sqrt(skymodel + (2.5**2)/1.2), color='r')
                #plt.plot(sky_fitx, -1.0 * np.sqrt(skymodel + (2.5**2)/1.2), color='r') 

                plt.plot(sky_fitx, sky_fity, '.', markersize=0.1, label='std', color='k')
                plt.plot(sky_fitx, skymodel, color='r')

                plt.show()
            """
    except: 
        print("Error in plotting")

    # Save the extraction object with sky attribute populated back out to disk

    outputfile = science_extraction_file.replace('_extractions.pkl','_sky1d_extractions.pkl')
    print("Saving science extraction with sky model to ", outputfile)
    save_extractions(science, primary_header=hdr, savefile=outputfile)

    return outputfile
