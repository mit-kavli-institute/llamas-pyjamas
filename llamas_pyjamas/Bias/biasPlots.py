"""
llamas_pyjamas.Bias.biasPlots
==============================
Diagnostic plot generation for LLAMAS bias quality checks.

All functions accept a BiasCheckReport (or raw data arrays) and return a
matplotlib Figure that can be saved or displayed interactively.

Functions
---------
plot_bias_level_heatmap       -- 24-detector grid of bias levels
plot_interfibre_residuals     -- bar chart of residual medians and stds
plot_bias_vs_interfibre       -- scatter: bias median vs inter-fibre median
plot_spatial_residual_images  -- 1×3 panel (Raw | Bias | Residual) with mask overlay
plot_bias_check_dashboard     -- 2×2 multi-panel summary
"""

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from llamas_pyjamas.Bias.biasChecking import BiasCheckReport, DetectorBiasStats

logger = logging.getLogger(__name__)

# Bench/side/color ordering for the 24-detector heatmap
_BENCHES = ['1', '2', '3', '4']
_SIDES   = ['A', 'B']
_COLORS  = ['red', 'green', 'blue']


def _cam_label(stats: DetectorBiasStats) -> str:
    return f"{stats.bench}{stats.side}\n{stats.color[:1].upper()}"


def _build_grid(report: BiasCheckReport, attr: str):
    """
    Build a 4×6 grid (benches × side+color) of a DetectorBiasStats attribute.
    Returns (grid, row_labels, col_labels).
    """
    # col order: A-red A-green A-blue B-red B-green B-blue
    cols = [(s, c) for s in _SIDES for c in _COLORS]
    col_labels = [f"{s}-{c[:1].upper()}" for s, c in cols]
    row_labels = [f"Bench {b}" for b in _BENCHES]

    grid = np.full((len(_BENCHES), len(cols)), np.nan)
    for stats in report.detector_stats:
        try:
            row = _BENCHES.index(str(stats.bench))
            col = cols.index((stats.side.upper(), stats.color.lower()))
            grid[row, col] = getattr(stats, attr, np.nan)
        except (ValueError, AttributeError):
            pass
    return grid, row_labels, col_labels


# ---------------------------------------------------------------------------
# 1. Bias level heatmap
# ---------------------------------------------------------------------------

def plot_bias_level_heatmap(report: BiasCheckReport) -> plt.Figure:
    """
    24-detector heatmap of the inter-fibre bias median level.

    Parameters
    ----------
    report : BiasCheckReport

    Returns
    -------
    matplotlib.figure.Figure
    """
    grid, row_labels, col_labels = _build_grid(report, 'interfibre_bias_median')

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, aspect='auto', cmap='viridis',
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    plt.colorbar(im, ax=ax, label='Inter-fibre bias median (DN)')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title('Bias Level Heatmap — 24 Detectors')

    # Annotate cells
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=6, color='white')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Inter-fibre residuals bar chart
# ---------------------------------------------------------------------------

def plot_interfibre_residuals(report: BiasCheckReport) -> plt.Figure:
    """
    Bar chart of residual_median ± residual_std for each detector.

    Parameters
    ----------
    report : BiasCheckReport

    Returns
    -------
    matplotlib.figure.Figure
    """
    stats_list = report.detector_stats
    if not stats_list:
        logger.warning("plot_interfibre_residuals: no detector stats to plot")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    labels   = [f"{s.bench}{s.side}\n{s.color[:1].upper()}" for s in stats_list]
    medians  = [s.residual_median for s in stats_list]
    stds     = [s.residual_std    for s in stats_list]
    colors   = ['tomato' if s.warning_flags else 'steelblue' for s in stats_list]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 5))
    ax.bar(x, medians, yerr=stds, capsize=3, color=colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline( report.thresholds.max_residual_median, color='red',
                linewidth=0.8, linestyle='--', label='threshold')
    ax.axhline(-report.thresholds.max_residual_median, color='red',
                linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Residual median ± std (DN)')
    ax.set_title('Inter-fibre Bias Residuals per Detector')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Bias vs inter-fibre scatter
# ---------------------------------------------------------------------------

def plot_bias_vs_interfibre(report: BiasCheckReport) -> plt.Figure:
    """
    Scatter plot: inter-fibre bias median vs inter-fibre science median.

    Parameters
    ----------
    report : BiasCheckReport

    Returns
    -------
    matplotlib.figure.Figure
    """
    stats_list = report.detector_stats
    bias_meds  = [s.interfibre_bias_median     for s in stats_list]
    sci_meds   = [s.interfibre_science_median  for s in stats_list]
    colors_pt  = ['tomato' if s.warning_flags else 'steelblue' for s in stats_list]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(bias_meds, sci_meds, c=colors_pt, alpha=0.8, edgecolors='k', s=60)

    # 1:1 line
    all_vals = bias_meds + sci_meds
    finite   = [v for v in all_vals if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='1:1')

    for stats in stats_list:
        if np.isfinite(stats.interfibre_bias_median) and np.isfinite(stats.interfibre_science_median):
            ax.annotate(
                f"{stats.bench}{stats.side}-{stats.color[:1].upper()}",
                (stats.interfibre_bias_median, stats.interfibre_science_median),
                fontsize=6, alpha=0.7,
            )

    ax.set_xlabel('Bias inter-fibre median (DN)')
    ax.set_ylabel('Science inter-fibre median (DN)')
    ax.set_title('Bias vs Science Inter-fibre Levels')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Spatial residual images with mask overlay
# ---------------------------------------------------------------------------

def plot_spatial_residual_images(frame: np.ndarray,
                                  bias: np.ndarray,
                                  mask: np.ndarray,
                                  title: str = '') -> plt.Figure:
    """
    1×3 panel showing Raw frame | Bias frame | Residual, with the
    inter-fibre mask boundary overlaid as a contour on each panel.

    Parameters
    ----------
    frame : numpy.ndarray
        Raw 2-D science or flat frame.
    bias : numpy.ndarray
        Bias frame (same shape).
    mask : numpy.ndarray
        Boolean inter-fibre gap mask (True = gap pixel).
    title : str
        Optional suptitle for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    residual = frame.astype(float) - bias.astype(float)
    panels   = [frame, bias, residual]
    titles   = ['Raw Frame', 'Bias Frame', 'Residual (Raw - Bias)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, data, ttl in zip(axes, panels, titles):
        vmin, vmax = np.nanpercentile(data, [1, 99])
        ax.imshow(data, origin='lower', aspect='auto', cmap='gray',
                  vmin=vmin, vmax=vmax)
        # Overlay mask boundary
        ax.contour(mask.astype(float), levels=[0.5], colors='cyan',
                   linewidths=0.5, alpha=0.7)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Dashboard (2×2 multi-panel)
# ---------------------------------------------------------------------------

def plot_bias_check_dashboard(report: BiasCheckReport,
                               title: str = 'Bias QA Dashboard') -> plt.Figure:
    """
    2×2 multi-panel figure combining the four plot types.

    Parameters
    ----------
    report : BiasCheckReport
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

    # --- Panel 1 (top-left): heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    grid, row_labels, col_labels = _build_grid(report, 'interfibre_bias_median')
    im = ax1.imshow(grid, aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax1, label='Bias median (DN)', fraction=0.046, pad=0.04)
    ax1.set_xticks(range(len(col_labels)))
    ax1.set_xticklabels(col_labels, fontsize=7)
    ax1.set_yticks(range(len(row_labels)))
    ax1.set_yticklabels(row_labels, fontsize=7)
    ax1.set_title('Bias Level Heatmap', fontsize=9)

    # --- Panel 2 (top-right): scatter ---
    ax2 = fig.add_subplot(gs[0, 1])
    stats_list = report.detector_stats
    bm = [s.interfibre_bias_median    for s in stats_list]
    sm = [s.interfibre_science_median for s in stats_list]
    cp = ['tomato' if s.warning_flags else 'steelblue' for s in stats_list]
    ax2.scatter(bm, sm, c=cp, alpha=0.8, edgecolors='k', s=40)
    finite = [v for v in bm + sm if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        ax2.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8)
    ax2.set_xlabel('Bias inter-fibre median (DN)', fontsize=8)
    ax2.set_ylabel('Science inter-fibre median (DN)', fontsize=8)
    ax2.set_title('Bias vs Science', fontsize=9)

    # --- Panel 3 (bottom-left): residual medians ---
    ax3 = fig.add_subplot(gs[1, 0])
    labels  = [f"{s.bench}{s.side}-{s.color[:1].upper()}" for s in stats_list]
    medians = [s.residual_median for s in stats_list]
    stds    = [s.residual_std    for s in stats_list]
    xp      = np.arange(len(labels))
    bar_colors = ['tomato' if s.warning_flags else 'steelblue' for s in stats_list]
    ax3.bar(xp, medians, yerr=stds, capsize=2, color=bar_colors, alpha=0.8)
    ax3.axhline(0, color='black', linewidth=0.6)
    ax3.axhline( report.thresholds.max_residual_median, color='red',
                 linewidth=0.6, linestyle='--')
    ax3.axhline(-report.thresholds.max_residual_median, color='red',
                 linewidth=0.6, linestyle='--')
    ax3.set_xticks(xp)
    ax3.set_xticklabels(labels, rotation=90, fontsize=5)
    ax3.set_ylabel('Residual (DN)', fontsize=8)
    ax3.set_title('Inter-fibre Residuals', fontsize=9)

    # --- Panel 4 (bottom-right): test-region medians ---
    ax4 = fig.add_subplot(gs[1, 1])
    test_meds = [s.test_region_median for s in stats_list]
    ax4.bar(xp, test_meds, color='mediumseagreen', alpha=0.8)
    ax4.axhline(0, color='black', linewidth=0.6)
    ax4.set_xticks(xp)
    ax4.set_xticklabels(labels, rotation=90, fontsize=5)
    ax4.set_ylabel('Rows 30-50 residual (DN)', fontsize=8)
    ax4.set_title('Test-region Residuals (rows 30-50)', fontsize=9)

    fig.suptitle(f"{title}\n{report.summary}", fontsize=10)
    return fig
