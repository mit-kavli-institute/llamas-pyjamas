# Cube — 3D cube construction

Turns RSS files into IFU data cubes. There are **three** constructors here, which is the first
thing to understand about this directory.

## Which one runs

Selected by `cube_method` in the config, with `CRR_cube = true` overriding it:

| `cube_method` | Constructor | Notes |
|---|---|---|
| `simple` | `simple_cube_constructor.py` — `SimpleCubeConstructor` | **What the shipped config uses**, and the recommended path. Fast, MUSE-style interpolation |
| `crr` (or `CRR_cube = true`) | `crr_cube_constructor.py` — `CRRCubeConstructor` | Covariance-regularised reconstruction. Slower, experimental |
| `traditional` | `cubeConstruct.py` — `CubeConstructor` | Legacy. Also the fallback when a config omits `cube_method` |

> **Only `simple` propagates variance.** It writes `VAR` and `WEIGHT` extensions; the
> traditional and CRR constructors do not. Anything downstream that needs an uncertainty — flux
> calibration in particular — should require `VAR`/`WEIGHT` and refuse a cube without them,
> rather than assuming they are present.

Note the asymmetry in defaults: `example_config.txt` ships `cube_method = simple`, but
`construct_cube()`'s own parameter default is `'traditional'`. A config that omits the key
silently gets the legacy path.

Cube generation is **off by default** (`generate_cubes = False`) — the 2-D RSS products carry the
science and cubes are ~366 MB each. Build them on demand in the CubeViewer, or stack dithers
first (see [`docs/workflow/04-combining-dithers.md`](../../docs/workflow/04-combining-dithers.md)).

## Modules

| Module | Role |
|---|---|
| `simple_cube_constructor.py` | `SimpleCubeConstructor`. Grid methods: `oversampled` (default), `native_hex`, `nearest_hex`. Has a CLI |
| `cubeConstruct.py` | `CubeConstructor` — `construct_cube_from_rss()`, `load_rss_channels()`, `get_fiber_coordinates()`, `map_fiber_to_sky()` |
| `crr_cube_constructor.py` | `CRRCubeConstructor`, `CRRCubeConfig`, `CRRDataCube`, `RSSData` |
| `crr_kernels.py` | `double_gaussian_kernel()`, `wavelength_dependent_seeing()`, `build_kernel_matrix()` |
| `crr_weights.py` | `compute_crr_weights()`, `compute_shepard_weights()`, `build_variance_matrix()`, `regularized_svd_inverse()`, `flux_conservation_matrix()` |
| `crr_parallel.py` | `CRRWorker` (Ray actor), `parallel_cube_construction()` |
| `crr_cli.py` | CRR command-line entry point |
| `rss_to_crr_adapter.py` | `load_rss_as_crr_data()`, `combine_channels_for_crr()` |
| `datamodel.py` | Cube data-model definitions |

Fibre positions come from the RSS file's own `FIBERMAP` extension, matched against
`../LUT/LLAMAS_FiberMap_rev04.dat`.

The CRR method implements Liu et al. (2020), AJ 159, 22 — the paper is in
[`reference_material/papers/`](../../reference_material/papers/Liu_2020_AJ_159_22.pdf), and every
CRR cube carries `CUBEREF = 'Liu et al. (2020)'` in its header.

## Detailed docs

- [`SIMPLE_CUBE_README.md`](SIMPLE_CUBE_README.md) — the simple constructor's CLI and options
- [`framework.md`](framework.md) — how the traditional constructor works, step by step

## See also

API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.Cube.html>
