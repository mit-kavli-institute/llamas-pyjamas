import logging
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.Extract.extractLlamas import save_extractions
import llamas_pyjamas.Arc.arcLlamas as arc
import llamas_pyjamas.Sky.skySelect as skySelect
from llamas_pyjamas.QA import plot_ds9
from llamas_pyjamas.config import OUTPUT_DIR, CALIB_DIR, LUT_DIR
from pypeit.core.fitting import iterfit, robust_fit
from pypeit.core.wavecal.wvutils import arc_lines_from_spec
from astropy.io import fits
from astropy.table import Table
import os
import warnings

logger = logging.getLogger(__name__)

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


# Per-fibre sky provenance flags (F3). Recorded on each extraction object as a
# `sky_quality` int array and propagated to the RSS MASK by llamasRSS.
SKY_OK = 0
SKY_FALLBACK = 1      # sky model substituted from a channel-global fallback
SKY_NONE = 2          # no sky model available (placeholder/missing camera)


def _apply_sky_model(sset, xshift_min, xshift_max, sky_hi, sky_lo,
                     extension, fiber, n_fibers, sky, science,
                     channel, bench, side, quality_flag=SKY_OK):
    """Evaluate a fitted sky bspline on every fibre of a camera, bounding the
    output (F2) and tagging each fibre's sky provenance (F3).

    Returns the number of fibres whose output needed clipping.
    """
    n_clipped = 0
    for i in range(n_fibers):
        x_eval = np.clip(sky[extension[i]].xshift[fiber[i], :],
                         xshift_min, xshift_max)
        skymodel = sset.value(x_eval)[0]
        bad = (~np.isfinite(skymodel) | (skymodel < sky_lo)
               | (skymodel > sky_hi))
        if np.any(bad):
            n_clipped += 1
            skymodel = np.clip(np.nan_to_num(skymodel, nan=0.0, posinf=sky_hi,
                                             neginf=sky_lo), sky_lo, sky_hi)
        tp = science[extension[i]].relative_throughput[fiber[i]]
        if not np.isfinite(tp) or tp <= 0:
            tp = 1.0
        sci = science[extension[i]]
        sci.sky[fiber[i], :] = skymodel * tp
        if quality_flag != SKY_OK:
            if getattr(sci, 'sky_quality', None) is None:
                sci.sky_quality = np.zeros(sci.sky.shape[0], dtype=np.int16)
            sci.sky_quality[fiber[i]] = quality_flag
    return n_clipped


def skyModel_1d(science_extraction_file, color, sky_extraction_file=None, show_plots=False,
                selection_method='dimmest', n_sky_fibres=20, sky_map=None,
                arc_soln=None):
    """
    Create a 1D sky model from the sky extraction.

    Parameters:
    science_extraction (ExtractLlamas): Extracted science data.
    sky_extraction (ExtractLlamas): Extracted data from a blank sky reference field

    Note that as a default, the code assumes that the user estimates the sky from
    the science field itself. If a separate sky field is provided, it will be used instead.

    The fibres that build the model are chosen per camera by ``selection_method``
    (see :mod:`llamas_pyjamas.Sky.skySelect`):

    * ``'dimmest'``      (default) the ``n_sky_fibres`` faintest fibres.
    * ``'middle-third'`` legacy central-third-by-brightness behaviour.
    * ``'skymap'``       fibres falling in the user sky region ``sky_map``
                         (a :class:`~llamas_pyjamas.Sky.skySelect.SkyMap`).
    * ``'frame'``        all good fibres — use when ``sky_extraction_file`` is a
                         dedicated blank-sky exposure.

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
        # The science extraction reaching this function is ALWAYS already
        # wavelength-calibrated (correct_wavelengths ran upstream). The former
        # unconditional arcTransfer against the packaged LLAMAS_reference_arc
        # here silently OVERWROTE any refined solution (refine_arc) with the
        # baseline quadratic on every run — do not re-transfer.
        sky_dict = science_dict
        sky_wvcal = sky_dict
    else:
        # Dedicated blank-sky frame: calibrate it with the SAME solution the
        # science frames used (arc_soln: path or loaded dict), falling back to
        # the packaged reference arc only when none is supplied.
        sky_dict = ExtractLlamas.loadExtraction(sky_extraction_file)
        if isinstance(arc_soln, dict):
            arc_dict = arc_soln
        else:
            if isinstance(arc_soln, str) and os.path.exists(arc_soln):
                arc_path = arc_soln
            else:
                arc_path = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
            print(f"skyModel_1d: calibrating sky frame with {os.path.basename(arc_path)}")
            arc_dict = ExtractLlamas.loadExtraction(arc_path)
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

    # F3: accumulate per-channel sky points from successfully-fit cameras so a
    # camera with "no finite sky fibres" can borrow a channel-global model in a
    # second pass instead of being left with a silent zero sky.
    channel_global_pts = {}   # channel -> [(x_array, y_array, sky_hi), ...]
    deferred_cameras = []     # cameras needing the global fallback

    for (channel, bench, side), _ in cameras:

        extension = np.array([])
        fiber = np.array([])
        counts = np.array([])
        cam_dead_fibers = []      # fibremap positions of this camera's dead fibres

        print(f"Generating sky model for camera {channel} bench {bench}{side}")

        # Collect all fibers from this camera only
        for i in range(len(sky)):
            if (sky_metadata[i]['channel'] == channel and
                    str(sky_metadata[i]['bench']) == bench and
                    sky_metadata[i]['side'] == side):
                cam_dead_fibers = list(getattr(sky[i], 'dead_fibers', []) or [])
                for thisfiber in range(sky_metadata[i]['nfibers']):
                    fiber = np.append(fiber, thisfiber)
                    extension = np.append(extension, i)
                    tt = np.sum(sky[i].counts[thisfiber]/sky[i].relative_throughput[thisfiber])
                    counts = np.append(counts, tt)

        # Cast index arrays for fibre lookups.
        extension = extension.astype(int)
        fiber = fiber.astype(int)
        n_fibers = len(counts)

        # Skip missing/placeholder cameras. When a camera is absent the pipeline
        # inserts a placeholder extension whose extracted counts are all zero
        # (set in guiExtract); that is the canonical "no real data" test used
        # across the pipeline.
        cam_exts = np.unique(extension)
        is_placeholder = all(np.allclose(np.nan_to_num(sky[e].counts), 0.0)
                             for e in cam_exts)
        finite_any = np.isfinite(counts)
        if is_placeholder:
            # Genuinely missing data — no sky can be modelled. Leave zero but tag
            # the fibres so downstream products are not mistaken for subtracted.
            print(f"  Skipping sky model for camera {channel} {bench}{side}: "
                  f"missing/placeholder extension.")
            # Info, not warning: placeholder (missing-hardware) cameras are a
            # permanent, expected condition and would otherwise repeat on the
            # curated terminal for every frame.
            logger.info("skyModel_1d: skipping camera %s %s%s "
                        "(missing/placeholder extension)", channel, bench, side)
            for i in range(n_fibers):
                sci = science[extension[i]]
                if getattr(sci, 'sky_quality', None) is None:
                    sci.sky_quality = np.zeros(sci.sky.shape[0], dtype=np.int16)
                sci.sky_quality[fiber[i]] = SKY_NONE
            continue
        if finite_any.sum() == 0:
            # Real data but no usable sky fibres on this benchside. Defer to a
            # channel-global fallback (F3) instead of leaving a silent zero sky.
            print(f"  Camera {channel} {bench}{side}: no finite sky fibres — "
                  f"deferring to channel-global fallback.")
            logger.warning("skyModel_1d: %s %s%s has no finite sky fibres; "
                           "deferring to channel-global fallback", channel, bench, side)
            deferred_cameras.append(dict(channel=channel, bench=bench, side=side,
                                         extension=extension.copy(),
                                         fiber=fiber.copy(), n_fibers=n_fibers))
            continue

        # Preferred sky candidates: finite AND positive (keeps 'dimmest' from
        # picking dead/zero fibres).
        usable = finite_any & (counts > 0)

        # Choose the sky fibres for this camera via the configured method.
        # 'skymap' maps each fibre's spatial position into the user sky map;
        # every other method ranks/selects on the white-light brightness.
        in_region = None
        if selection_method == 'skymap' and sky_map is not None:
            benchsides_cam = np.array([f"{bench}{side}"] * n_fibers)
            # `fiber` is a LIVE row index (0..n-1); FiberMap_LUT (via
            # fibres_in_sky_region) needs the physical fibremap position, which
            # differs after the first dead fibre. Map live -> physical.
            from llamas_pyjamas.Utils.deadfibers import live_fibre_ids
            phys_fiber = np.asarray(live_fibre_ids(n_fibers, cam_dead_fibers),
                                    dtype=int)
            in_region = skySelect.fibres_in_sky_region(benchsides_cam, phys_fiber,
                                                       sky_map)
        # Per-fibre slit position (trace-y at mid column) for 'stratified'.
        fiber_y = None
        if selection_method == 'stratified':
            fiber_y = np.full(n_fibers, np.nan)
            for k in range(n_fibers):
                tr = getattr(sky[extension[k]], 'trace', None)
                traces = getattr(tr, 'traces', None) if tr is not None else None
                if traces is not None and fiber[k] < traces.shape[0]:
                    fiber_y[k] = traces[fiber[k], traces.shape[1] // 2]
            if not np.isfinite(fiber_y).any():
                fiber_y = None  # no trace info -> select_sky_fibres falls back to quantile
        sel_mask = skySelect.select_sky_fibres(
            counts, usable, method=selection_method,
            n_fibres=n_sky_fibres, in_sky_region=in_region, fiber_y=fiber_y)

        # Min-fibre floor for 'dimmest'/'quantile': low-signal cameras (e.g. faint
        # blue) may have very few positive fibres, leaving a fit built from 1-2
        # fibres. Broaden to the middle-third of finite fibres for a sturdier fit.
        if (selection_method in ('dimmest', 'quantile', 'stratified')
                and sel_mask.sum() < skySelect.MIN_SKY_FIT_FIBRES
                and finite_any.sum() >= skySelect.MIN_SKY_FIBRES):
            broadened = skySelect.select_sky_fibres(counts, finite_any, method='middle-third')
            if broadened.sum() > sel_mask.sum():
                print(f"  Low-signal camera {channel} {bench}{side}: broadening dimmest "
                      f"({int(sel_mask.sum())}) -> middle-third ({int(broadened.sum())} fibres)")
                logger.info("skyModel_1d: %s %s%s only %d positive sky fibres; broadening "
                            "to middle-third (%d)", channel, bench, side,
                            int(sel_mask.sum()), int(broadened.sum()))
                sel_mask = broadened

        sel_idx = np.where(sel_mask)[0]

        # F3: a per-camera fit from only 1-2 faint fibres produces a near-zero,
        # unreliable sky (e.g. faint blue benchsides that selected a single
        # ~zero-count fibre). Defer those to the channel-global fallback, which
        # pools many fibres across the colour, rather than writing a zero sky.
        MIN_CAMERA_SKY_FIBRES = 3
        if len(sel_idx) < MIN_CAMERA_SKY_FIBRES:
            print(f"  Camera {channel} {bench}{side}: only {len(sel_idx)} usable "
                  f"sky fibre(s) — deferring to channel-global fallback.")
            logger.warning("skyModel_1d: %s %s%s only %d usable sky fibre(s); "
                           "deferring to channel-global fallback", channel, bench,
                           side, len(sel_idx))
            deferred_cameras.append(dict(channel=channel, bench=bench, side=side,
                                         extension=extension.copy(),
                                         fiber=fiber.copy(), n_fibers=n_fibers))
            continue

        sky_fitx = np.array([])
        sky_fity = np.array([])
        sky_fitf = np.array([])   # source-fibre id per point (for coverage trimming)

        for i in sel_idx:
            tp = sky[extension[i]].relative_throughput[fiber[i]]
            if not np.isfinite(tp) or tp <= 0:
                tp = 1.0
            xr = sky[extension[i]].xshift[fiber[i], :]
            sky_fitx = np.append(sky_fitx, xr)
            sky_fity = np.append(sky_fity, sky[extension[i]].counts[fiber[i], :] / tp)
            sky_fitf = np.append(sky_fitf, np.full(xr.size, i))

        # Re-sort in order of increasing wavelength
        idx = np.argsort(sky_fitx)
        sky_fitx = sky_fitx[idx]
        sky_fity = sky_fity[idx]
        sky_fitf = sky_fitf[idx]

        # Filter out bad pixels before the fit
        gd = (~np.isnan(sky_fity))
        sky_fitx = sky_fitx[gd]
        sky_fity = sky_fity[gd]
        sky_fitf = sky_fitf[gd]

        # Fit-domain trimming: restrict the fit to the xshift range covered by at
        # least MIN_CAMERA_SKY_FIBRES distinct fibres. A fibre with a deviant
        # wavelength solution (e.g. green 3B's ~-71 px edge fibres) extends the
        # pooled domain into a region only IT covers; the bspline knots there are
        # singular ("NaN in cholesky_band") and pypeit then returns an sset that
        # evaluates to zero EVERYWHERE, silently zeroing the camera's sky. Unlike
        # a per-fibre offset cut, coverage trimming keeps every fibre over the
        # well-sampled range (the legit fibre-to-fibre xshift spread is ~23 px,
        # so no offset threshold cleanly separates bogus from real).
        if sky_fitx.size:
            _nbin = 256
            _lo, _hi = np.nanmin(sky_fitx), np.nanmax(sky_fitx)
            if _hi > _lo:
                _edges = np.linspace(_lo, _hi, _nbin + 1)
                _bi = np.clip(np.digitize(sky_fitx, _edges) - 1, 0, _nbin - 1)
                _cov = np.array([len(np.unique(sky_fitf[_bi == k])) for k in range(_nbin)])
                _ok_bins = np.where(_cov >= MIN_CAMERA_SKY_FIBRES)[0]
                if _ok_bins.size:
                    _dlo, _dhi = _edges[_ok_bins.min()], _edges[_ok_bins.max() + 1]
                    _keep = (sky_fitx >= _dlo) & (sky_fitx <= _dhi)
                    if _keep.sum() < sky_fitx.size:
                        print(f"  Trimmed sky-fit domain to [{_dlo:.0f},{_dhi:.0f}] "
                              f"(>= {MIN_CAMERA_SKY_FIBRES} fibres/bin); dropped "
                              f"{sky_fitx.size - int(_keep.sum())} sparse edge pixels")
                        sky_fitx = sky_fitx[_keep]; sky_fity = sky_fity[_keep]; sky_fitf = sky_fitf[_keep]

        if sky_fitx.size == 0:
            # F3: defer to channel-global fallback instead of a silent zero sky.
            print(f"  Camera {channel} {bench}{side}: no good sky pixels — "
                  f"deferring to channel-global fallback.")
            logger.warning("skyModel_1d: %s %s%s no good sky pixels; deferring "
                           "to channel-global fallback", channel, bench, side)
            deferred_cameras.append(dict(channel=channel, bench=bench, side=side,
                                         extension=extension.copy(),
                                         fiber=fiber.copy(), n_fibers=n_fibers))
            continue

        xshift_min, xshift_max = sky_fitx.min(), sky_fitx.max()
        print(f"  xshift range: {xshift_min:.2f} to {xshift_max:.2f}")
        if xshift_max - xshift_min < 1.0:
            # F3: degenerate xshift (e.g. arcTransfer metadata mismatch) — defer
            # to the channel-global fallback rather than leaving a zero sky.
            print(f"  WARNING: Degenerate xshift for camera {channel} bench {bench}{side} — "
                  "deferring to channel-global fallback.")
            logger.warning("skyModel_1d: degenerate xshift for %s %s%s; deferring "
                           "to channel-global fallback", channel, bench, side)
            deferred_cameras.append(dict(channel=channel, bench=bench, side=side,
                                         extension=extension.copy(),
                                         fiber=fiber.copy(), n_fibers=n_fibers))
            continue

        # Cross-fibre consensus rejection (before the bspline). A REAL sky line
        # is high in every fibre; a per-fibre artifact (hot pixel, surviving
        # cosmic ray, flat feature) is high in ONE. The bspline's own rejection
        # cannot tell them apart because its upper limit is deliberately lenient
        # (kept high so real sky lines are not clipped), so a 1-fibre spike gets
        # fit as if it were sky and its narrow model bump over-subtracts that one
        # camera (green 3A/3B/4B, 2026-07). Here each point is compared to the
        # running cross-fibre median at its xshift (window ~ a couple of fibres'
        # worth of the xshift-sorted pool): points far ABOVE the consensus are
        # dropped, real lines (all fibres high => high median) survive.
        if len(sel_idx) >= 8 and sky_fitx.size > 4 * len(sel_idx):
            from scipy.ndimage import median_filter
            # Window ~ one fibre's worth of the xshift-sorted pool, so the
            # running median tracks even a sharp line's PEAK (a wider window
            # spans the line shoulders and would clip real bright peaks).
            # Threshold 8*MAD: single-fibre artifacts sit tens of MAD above the
            # consensus, real bright lines only a few, so this catches the
            # former without touching the latter.
            win = int(max(len(sel_idx), 15))
            base = median_filter(sky_fity, size=win, mode='nearest')
            mad = 1.4826 * median_filter(np.abs(sky_fity - base), size=win, mode='nearest')
            mad = np.maximum(mad, 0.05 * np.nanmedian(np.abs(base)) + 1.0)
            keep_c = (sky_fity - base) <= 8.0 * mad
            n_rej = int((~keep_c).sum())
            if 0 < n_rej < 0.2 * sky_fity.size:   # sanity: never drop a big fraction
                sky_fitx = sky_fitx[keep_c]; sky_fity = sky_fity[keep_c]
                sky_fitf = sky_fitf[keep_c]
                print(f"  Consensus rejection: dropped {n_rej} single-fibre outlier "
                      f"pixels ({channel} {bench}{side})")

        print(f"  Fitting sky with {len(sky_fitx)} points from {len(sel_idx)} fibers "
              f"(method='{selection_method}')")
        # Rejection: sky EMISSION lines are real positive signal, not outliers.
        # The pypeit default (upper=lower=5) clips line peaks — a bright blue line
        # is ~7 sigma above the continuum scatter — so the bspline fits only the
        # continuum and the lines are left ~90% unsubtracted. Use a high upper so
        # true lines are kept, and a moderate lower to still reject dead/negative
        # pixels. (bkspace 0.5 px is fine enough to represent the line profile.)
        sky_reject_upper = 30.0
        sky_reject_lower = 5.0
        sset, outmask = iterfit(sky_fitx, sky_fity, maxiter=6,
                                upper=sky_reject_upper, lower=sky_reject_lower,
                                kwargs_bspline={'bkspace': 0.5})

        # Guard: a singular fit (e.g. "NaN in cholesky_band") returns an sset
        # that evaluates to ~zero everywhere while the data are healthy. Never
        # accept such a model silently — defer to the channel-global fallback.
        _probe = np.nanpercentile(sky_fitx, [10, 30, 50, 70, 90])
        _mvals = sset.value(np.sort(_probe))[0]
        _data_med = float(np.nanmedian(sky_fity))
        if (not np.isfinite(_mvals).any()) or \
           (np.nanmax(_mvals) <= 0) or \
           (_data_med > 10 and float(np.nanmedian(_mvals)) < 0.05 * _data_med):
            print(f"  WARNING: degenerate sky-model fit for {channel} {bench}{side} "
                  f"(model median {float(np.nanmedian(_mvals)):.2f} vs data median "
                  f"{_data_med:.2f}) — deferring to channel-global fallback.")
            logger.warning("skyModel_1d: %s %s%s degenerate sky-model fit "
                           "(model~0, data median %.1f); deferring to "
                           "channel-global fallback", channel, bench, side, _data_med)
            deferred_cameras.append(dict(channel=channel, bench=bench, side=side,
                                         extension=extension.copy(),
                                         fiber=fiber.copy(), n_fibers=n_fibers))
            continue

        # F2: bound the sky model OUTPUT. Clipping the input xshift (below) is not
        # sufficient — the bspline can still return catastrophic values (~1e8–1e11)
        # that poison FLUX = COUNTS - SKY and the cubes. Sky photons are >= 0 and
        # cannot greatly exceed the brightest sky pixel actually fit; bound to a
        # generous multiple of that so true OH/[O I] lines are never clipped.
        _sky_hi = float(np.nanpercentile(sky_fity, 99.9)) * 3.0
        if not np.isfinite(_sky_hi) or _sky_hi <= 0:
            _sky_hi = float(np.nanmax(sky_fity)) if sky_fity.size else 0.0
        _sky_lo = 0.0
        _n_clipped_fibers = 0

        if show_plots:
            plt.plot(sky_fitx, sky_fity, '.', markersize=0.1, label='data', color='k')
            y = sset.value(sky_fitx)[0]
            plt.plot(sky_fitx, y, color='r')
            plt.ylim(0,1000)
            plt.title(f'Sky fit: {channel} {bench}{side}')
            plt.show()

        # Record this camera's sky points for the channel-global fallback (F3).
        channel_global_pts.setdefault(channel, []).append(
            (sky_fitx.copy(), sky_fity.copy(), _sky_hi))

        # Apply sky model to all fibers in this camera (output bounded by F2).
        print(f"  Applying sky model to {n_fibers} fibers")
        _n_clipped_fibers = _apply_sky_model(
            sset, xshift_min, xshift_max, _sky_hi, _sky_lo,
            extension, fiber, n_fibers, sky, science, channel, bench, side)
        if _n_clipped_fibers:
            # Info, not warning: the output clamp is a safety net that bounds the
            # bspline's edge/extrapolation overshoot; it fires on nearly every
            # camera (every fibre has spectrum-edge pixels) and does not degrade
            # the fitted sky over the well-sampled range. Logged for diagnosis;
            # surfaced as a per-run tally below rather than one line per camera.
            logger.info("skyModel_1d: %s %s%s clipped catastrophic sky-model "
                        "output in %d/%d fibres (cap=%.1f)", channel, bench,
                        side, _n_clipped_fibers, n_fibers, _sky_hi)

    # ── F3 pass 2: channel-global fallback for deferred cameras ──
    for cam in deferred_cameras:
        channel = cam['channel']; bench = cam['bench']; side = cam['side']
        pts = channel_global_pts.get(channel)
        if not pts:
            logger.warning("skyModel_1d: %s %s%s no channel-global sky available "
                           "(no other %s benchside modelled); leaving zero sky, "
                           "flagged SKY_NONE", channel, bench, side, channel)
            for i in range(cam['n_fibers']):
                sci = science[cam['extension'][i]]
                if getattr(sci, 'sky_quality', None) is None:
                    sci.sky_quality = np.zeros(sci.sky.shape[0], dtype=np.int16)
                sci.sky_quality[cam['fiber'][i]] = SKY_NONE
            continue
        gx = np.concatenate([p[0] for p in pts])
        gy = np.concatenate([p[1] for p in pts])
        g_hi = float(np.nanmax([p[2] for p in pts]))
        order = np.argsort(gx)
        gx, gy = gx[order], gy[order]
        good = ~np.isnan(gy)
        gx, gy = gx[good], gy[good]
        if gx.size < 10 or (gx.max() - gx.min()) < 1.0:
            logger.warning("skyModel_1d: %s %s%s channel-global sky degenerate; "
                           "leaving zero sky, flagged SKY_NONE", channel, bench, side)
            continue
        gset, _ = iterfit(gx, gy, maxiter=6, kwargs_bspline={'bkspace': 0.5})
        n_clip = _apply_sky_model(
            gset, gx.min(), gx.max(), g_hi, 0.0,
            cam['extension'], cam['fiber'], cam['n_fibers'], sky, science,
            channel, bench, side, quality_flag=SKY_FALLBACK)
        print(f"  Channel-global fallback sky applied to {channel} {bench}{side} "
              f"({cam['n_fibers']} fibres)")
        logger.warning("skyModel_1d: %s %s%s used channel-global fallback sky "
                       "(%d fibres, %d output-clipped)", channel, bench, side,
                       cam['n_fibers'], n_clip)

    #########################################


    # Plot some QA - show a few fibers from the middle of the last camera processed
    try:
        if show_plots:
            qa_indices = sel_idx[:5]
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
