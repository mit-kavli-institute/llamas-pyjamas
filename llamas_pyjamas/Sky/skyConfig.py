"""
llamas_pyjamas.Sky.skyConfig
============================
Configuration for the LLAMAS sky-subtraction framework.

The framework runs *after* the fibre-to-fibre flat as a per-colour, FITS-level
stage.  It builds on the base 1-D sky model already produced by
``Sky.skyLlamas.skyModel_1d`` (carried in the ``SKY`` extension of the RSS/FF
files) and refines it with standard IFU practice:

    source masking  ->  per-fibre OH-line scaling  ->  PCA residual cleaning

All tunables live here so the algorithm modules stay free of magic numbers.

Public API
----------
SkySubtractConfig -- dataclass of every tunable, with ``from_pipeline_config``
                     to build one from the pipeline ``config`` dict.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SkySubtractConfig:
    """Tunables for :func:`Sky.skySubtract.subtract_sky_rss`.

    Attributes
    ----------
    method : str
        ``'scaled'`` applies per-fibre OH scaling only; ``'pca'`` additionally
        runs the ZAP-style residual cleaning.  Default ``'pca'``.

    # --- source masking (skyMask) ---
    mask_method : str
        ``'whitelight'`` selects sky fibres from a white-light brightness proxy
        (and ``FIBER_TYPE`` when present).  ``'none'`` treats all fibres as sky.
    sky_fiber_percentile : float
        Fibres fainter than this white-light percentile are candidate sky
        fibres (object fibres are the bright tail).  Default 60.0.
    bright_reject_percentile : float
        The faintest fibres below this percentile are dropped from the sky set
        (too noisy / dead).  Default 10.0.

    # --- per-fibre OH scaling (skyScale) ---
    oh_sigdetect : float
        Detection threshold (sigma) for OH-line peaks via ``arc_lines_from_spec``.
    oh_fwhm : float
        Expected OH line FWHM in pixels.
    scale_window_pix : int
        Half-width (pixels) of the window around each OH line used to fit the
        per-fibre sky scale.
    scale_min : float
        Lower clip on the fitted per-fibre scale correction factor.
    scale_max : float
        Upper clip on the fitted per-fibre scale correction factor.
    min_oh_lines : int
        Minimum OH lines required to attempt a per-fibre scale fit; otherwise
        the fibre keeps the base model (scale = 1, no correction).

    # --- PCA residual cleaning (skyResidual) ---
    pca_ncomp : int
        Number of leading eigen-components projected out of every fibre.
    pca_max_basis_fibers : int
        Cap on the number of sky-masked fibres used to build the eigenbasis
        (for speed/memory); 0 means use all.

    # --- QA ---
    qa_plots : bool
        Write diagnostic plots.
    qa_dir : Optional[str]
        Directory for QA output; ``None`` => alongside the SKYSUB file.
    """

    method: str = "pca"

    # sky-fibre selection (shared concept with the base model skyModel_1d).
    # 'dimmest'/'frame' -> faint-population (white-light percentile) basis;
    # 'middle-third' -> central third by brightness; 'skymap' -> fibres in the
    # user sky map.  See llamas_pyjamas.Sky.skySelect.
    selection_method: str = "dimmest"
    sky_map_file: Optional[str] = None

    # source masking
    mask_method: str = "whitelight"
    sky_fiber_percentile: float = 60.0
    bright_reject_percentile: float = 10.0

    # per-fibre OH scaling
    oh_sigdetect: float = 5.0
    oh_fwhm: float = 4.0
    scale_window_pix: int = 6
    scale_min: float = 0.5
    scale_max: float = 1.5
    min_oh_lines: int = 3

    # PCA residual cleaning
    pca_ncomp: int = 20
    pca_max_basis_fibers: int = 0

    # QA
    qa_plots: bool = False
    qa_dir: Optional[str] = None

    def __post_init__(self):
        from llamas_pyjamas.Sky.skySelect import VALID_METHODS
        if self.method not in ("scaled", "pca"):
            raise ValueError(f"Unknown sky method {self.method!r}; "
                             "expected 'scaled' or 'pca'")
        if self.mask_method not in ("whitelight", "none"):
            raise ValueError(f"Unknown mask_method {self.mask_method!r}; "
                             "expected 'whitelight' or 'none'")
        if self.selection_method not in VALID_METHODS:
            raise ValueError(f"Unknown selection_method {self.selection_method!r}; "
                             f"expected one of {VALID_METHODS}")

    @property
    def run_pca(self) -> bool:
        """True when the PCA residual-cleaning stage should run."""
        return self.method == "pca"

    @classmethod
    def from_pipeline_config(cls, config: dict) -> "SkySubtractConfig":
        """Build from the pipeline ``config`` dict (all keys optional).

        Recognised keys mirror the dataclass fields with a ``sky_`` prefix
        where it disambiguates them in the flat pipeline namespace.
        """
        config = config or {}

        def get(*keys, default):
            for k in keys:
                if k in config and config[k] is not None:
                    return config[k]
            return default

        return cls(
            method=str(get("sky_method", default=cls.method)).lower(),
            selection_method=str(get("sky_selection_method",
                                     default=cls.selection_method)).lower(),
            sky_map_file=get("sky_map_file", default=cls.sky_map_file),
            mask_method=str(get("sky_mask_method", default=cls.mask_method)).lower(),
            sky_fiber_percentile=float(get("sky_fiber_percentile",
                                           default=cls.sky_fiber_percentile)),
            bright_reject_percentile=float(get("sky_bright_reject_percentile",
                                               default=cls.bright_reject_percentile)),
            oh_sigdetect=float(get("sky_oh_sigdetect", default=cls.oh_sigdetect)),
            oh_fwhm=float(get("sky_oh_fwhm", default=cls.oh_fwhm)),
            scale_window_pix=int(get("sky_scale_window", default=cls.scale_window_pix)),
            scale_min=float(get("sky_scale_min", default=cls.scale_min)),
            scale_max=float(get("sky_scale_max", default=cls.scale_max)),
            min_oh_lines=int(get("sky_min_oh_lines", default=cls.min_oh_lines)),
            pca_ncomp=int(get("sky_pca_ncomp", default=cls.pca_ncomp)),
            pca_max_basis_fibers=int(get("sky_pca_max_basis_fibers",
                                         default=cls.pca_max_basis_fibers)),
            qa_plots=bool(get("sky_qa_plots", default=cls.qa_plots)),
            qa_dir=get("sky_qa_dir", default=cls.qa_dir),
        )

    def to_header_dict(self) -> dict:
        """Compact provenance keys for the SKYSUB FITS header."""
        return {
            "SKYMETH": (self.method, "Sky framework method"),
            "SKYSEL": (self.selection_method, "Sky-fibre selection method"),
            "SKYPCANC": (self.pca_ncomp if self.run_pca else 0,
                         "PCA residual components removed"),
            "SKYMASKP": (self.sky_fiber_percentile,
                         "Sky-fibre white-light percentile cut"),
        }

    def as_dict(self) -> dict:
        return asdict(self)
