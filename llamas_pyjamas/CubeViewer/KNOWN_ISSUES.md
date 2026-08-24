# Bug report — CubeViewer aperture picking fails when more than one DS9 is running

**Reported by:** Sarah Hughes (slhughes@mit.edu)
**Date:** 2026-07-29
**Branch observed on:** `rs-dev` (no code changed — analysis only)
**Component:** `llamas_pyjamas/CubeViewer/` (XPA/DS9 transport + crosshair picker)
**Data:** `/Users/slh/Documents/Projects/Magellan_dev/LLAMAS/testing/sunburst/extractions`

---

## Summary

Opening the sunburst RSS files in the CubeViewer and sending the white-light image to DS9
works, but enabling **Pick** to select aperture fibres immediately fails with repeated
`Lost DS9: Un-parseable coordinates from DS9: ...`.

The reported console output contains **two unrelated defects that happen to appear together**:

| # | Defect | Severity |
|---|---|---|
| 1 | Crosshair polling breaks whenever ≥2 DS9 instances are running | **Blocking** — picking fails 100% of the time |
| 2 | `astropy VerifyWarning: Card is too long, comment will be truncated` | Cosmetic — but see §2.3 for a dangerous cousin |

Both are reproduced deterministically below, without needing DS9.

---

# 1. Blocking — aperture picking breaks with multiple DS9 instances

## 1.1 Symptom

```
WARNING llamas_pyjamas.CubeViewer.cubeViewPick: Lost DS9: Un-parseable coordinates from DS9:
'XPA$BEGIN DS9:ds9 7f000001:52307\n0.5 0.5\nXPA$END   DS9:ds9 7f000001:52307\n
 XPA$BEGIN DS9:ds9 7f000001:64510\n224.53958 272.76042\nXPA$END   DS9:ds9 7f000001:64510'
```

The message text *is* the diagnosis. The XPA reply carries `XPA$BEGIN` / `XPA$END` framing
wrapping **two** answers from **two** DS9 instances:

- `7f000001:52307` → `0.5 0.5` — the default crosshair position; this window was never clicked in
- `7f000001:64510` → `224.53958 272.76042` — the window actually being clicked

`7f000001` is the loopback address `127.0.0.1`, consistent with `XPA_METHOD` being unset
(inet transport). So two DS9 processes were live and both registered under the XPA name `ds9`.

## 1.2 Root cause

CubeViewer addresses DS9 by the bare XPA template `'ds9'`. It is hardcoded and unreachable
from the UI or CLI:

```python
# CubeViewer/cubeViewDS9.py:177
def __init__(self, target: str = 'ds9', timeout: float = DEFAULT_TIMEOUT) -> None:
    self.target = target

# CubeViewer/cubeViewLlamas.py:149,154
def __init__(self, path: Optional[str] = None, target: str = 'ds9') -> None:
    ...
    self.ds9 = DS9(target=target)
```

`main()` (`cubeViewLlamas.py:1100-1112`) parses only a positional path and never passes
`target=`, and there is no `--target` flag — so the target is always the literal string `'ds9'`.

When two DS9s match that template, `xpaget ds9 crosshair image` **multiplexes**: it returns
both replies concatenated, each framed with `XPA$BEGIN`/`XPA$END`. That raw string goes
straight into the parser:

```python
# CubeViewer/cubeViewDS9.py:345-347
def crosshair(self, system: str = 'image') -> Tuple[float, float]:
    """Return the current crosshair position in the given coordinate system."""
    return parse_coordinates(self.get(f'crosshair {system}'))

# CubeViewer/cubeViewDS9.py:141-155
def parse_coordinates(text: str) -> Tuple[float, float]:
    fields = text.split()
    if len(fields) < 2:
        raise DS9Error(f"Expected two coordinates from DS9, got {text!r}")
    try:
        return float(fields[0]), float(fields[1])
    except ValueError as exc:
        raise DS9Error(f"Un-parseable coordinates from DS9: {text!r}") from exc
```

`fields[0:2]` is `['XPA$BEGIN', 'DS9:ds9']` → `float()` raises `ValueError` → `DS9Error`.

The poll loop then tears the session down after three *consecutive* failures — ~300 ms at the
default 100 ms interval — which is why it dies the instant picking starts:

```python
# CubeViewer/cubeViewPick.py:77-88     (FAILURE_LIMIT = 3, DEFAULT_INTERVAL_MS = 100)
@pyqtSlot()
def _poll(self) -> None:
    try:
        x, y = self._ds9.crosshair('image')
    except DS9Error as exc:
        self._failures += 1
        if self._failures >= FAILURE_LIMIT:
            self.finish()
            self.failed.emit(str(exc))
        return
    self._failures = 0
    self.moved.emit(x, y)

# CubeViewer/cubeViewPick.py:284-288
@pyqtSlot(str)
def _on_failed(self, message: str) -> None:
    logger.warning('Lost DS9: %s', message)
    self.stop()
    self.lost.emit(message)
```

Because the failure is structural rather than transient, `FAILURE_LIMIT` never helps — every
poll fails. The three identical log lines in the report correspond to three **Pick** toggle
cycles, each burning its own three failures; `_on_ds9_lost` (`cubeViewLlamas.py:856`) unchecks
`pick_box` after each.

## 1.3 Verified reproduction (no DS9 required)

Feeding the exact captured string to the shipped parser:

```
fields[0:2] = ['XPA$BEGIN', 'DS9:ds9']
DS9Error -> Un-parseable coordinates from DS9: 'XPA$BEGIN DS9:ds9 7f000001:52307\n0.5 0.5\n...'

parse_coordinates('224.53958 272.76042') -> (224.53958, 272.76042)   # single instance is fine
```

## 1.4 Why no guard caught it

Every check in the transport tests for the **zero-match** case; none tests for **many-match**:

- `DS9._run` (`cubeViewDS9.py:209-217`) raises only on a non-zero exit code, or when `'match'`
  appears **on stderr**. A multiplexed reply is exit 0 with empty stderr — it sails through.
- `DS9.is_alive()` (`cubeViewDS9.py:247-249`) computes `int(xpaaccess -n ds9) > 0`. Two
  instances returns `2`, which reads as healthy. It is also never called from CubeViewer.
- `DS9.targets()` (`cubeViewDS9.py:222-232`) correctly enumerates instances but is **dead
  code** — its only caller is `Tests/test_cubeview_ds9.py:228`.
- The class docstring (`cubeViewDS9.py:162-165`) explicitly warns *"use an explicit
  `name:port` when several are running"*, but no code path ever supplies one.
- `Tests/test_cubeview_ds9.py:143` asserts the bare target as intended behaviour
  (`argv == ['ds9', 'crosshair', 'image']`), so the test suite currently pins the bug in place.

## 1.5 Second-order damage from the same root cause

`xpaset` with a bare template **broadcasts** to every matching instance:

- `set_fits` (`cubeViewDS9.py:313-343`) pipes the white-light FITS into *both* DS9s. This is
  why the stale instance still had a frame loaded, and it doubles the transfer of a multi-MB
  image on every wavelength change.
- `set('mode crosshair')` (`cubeViewPick.py:186`) flips *both* windows into crosshair mode.
- `set_regions` / `delete_region_group` paint aperture markers into both.

> **⚠️ Warning for whoever fixes this.** Stripping the `XPA$*` framing lines and taking the
> first coordinate pair would be **worse than the current failure**. It would silently return
> `0.5 0.5` from the wrong DS9 and select the wrong fibres with no error at all. The fix must
> disambiguate the *target*, not sanitise the *reply*.

## 1.6 Where the second DS9 came from (probable, not proven)

No DS9 was running by the time of this analysis, so the two instances could not be inspected
directly. However there is an unambiguous process leak in the pipeline:

```python
# QA/llamasQA.py:81-91      (plot_ds9, with the default samp=False)
process = subprocess.Popen(
    ['ds9', '-'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
process.communicate(input=fits_file.read())
```

A **brand-new DS9 process per call** — never reused, never reaped. `communicate()` returns as
soon as the pipes drain, but the GUI stays up indefinitely and nothing tracks `process`.

Reached from:
- `Image/WhiteLightModule.py:465-468` via `WhiteLight()`, which defaults to `ds9plot=True`
  (`WhiteLightModule.py:382`) — a plain `WhiteLight(...)` call spawns a DS9
- `QuickWhiteLightCube`, which runs it **per camera in a loop** (`WhiteLightModule.py:1334`)
- Re-exported package-wide via `QA/__init__.py` and imported by `Sky/skyLlamas.py:10` and
  `Image/processWhiteLight.py:45`

The sunburst run produced `*_whitelight.fits` products, so this path was exercised.

Additional latent spawn sites — all currently inert because **`pyds9` is not installed** in
`llamas_data_reduction_clean` and is not declared in `pyproject.toml`:

- `Trace/traceLlamas.py:313` — `pyds9.DS9(target='DS9:*', start=True, wait=10, verify=True)`
  inside `profileFit()`, guarded only by `if (True):` under a commented-out import. One new
  DS9 **per fibre** if ever reached — and a `NameError` as it stands today.
- `Flat/scattered2dLlamas.py:71` — same shape, same latent `NameError`.
- `Utils/reduxSetupGUI.py:564` — `pyds9.DS9()` with the default `start`, which *launches* a
  DS9 if none matches; on any exception it drops the handle (`self._ds9 = None`, line 570)
  without closing the old one, orphaning it.

## 1.7 Architectural note

The repo contains **three independent DS9 transports**:

| Mechanism | Location | Status |
|---|---|---|
| Raw `xpaget`/`xpaset` subprocess | `CubeViewer/cubeViewDS9.py` | The maintained path |
| `astropy.samp.SAMPIntegratedClient` | `GUI/obslog.py:115`, `QA/llamasQA.py:24` | Live |
| `pyds9` | `Utils/reduxSetupGUI.py`, `Trace/`, `Flat/` | Legacy / dead (not installed) |

Worth consolidating. Note also the SAMP client fallback at `obslog.py:112` /
`llamasQA.py:57` — `self.client_id = clients[-1]` binds to *the last registered SAMP client*,
whatever it is, if none advertises `samp.name == 'ds9'`.

## 1.8 Suggested fixes, in priority order

1. **Disambiguate the target — the real fix.** At viewer startup / `ElementPicker.start()`,
   call the existing `DS9.targets()` and:
   - 0 matches → the current "Is DS9 running?" error
   - 1 match → bind to its explicit `name:port` from `xpaget xpans`, rather than `ds9`, so a
     later-launched DS9 cannot hijack an in-progress session
   - \>1 match → raise, or prompt with the list, instead of proceeding. A `QInputDialog` over
     `targets()` output would suffice.
2. **Fail loudly on multiplexed replies.** In `DS9._run`, treat stdout containing `XPA$BEGIN`
   as an error naming the matched instances. Never let it reach a parser.
3. **Plumb a `--target` CLI flag** through `cubeViewLlamas.main()` → `CubeViewerWindow` →
   `DS9`, plus a `CUBEVIEWER_DS9_TARGET` env var, as a manual escape hatch.
4. **Stop leaking DS9 processes** — `QA/llamasQA.py:plot_ds9` should reuse a single instance
   via XPA, or at minimum `WhiteLight(ds9plot=...)` should default to `False`.
5. **Delete or guard the dead `pyds9` call sites** (`traceLlamas.py:313`,
   `scattered2dLlamas.py:71`) — they are latent `NameError`s inside hot loops.
6. **Fix the test that pins the bug** (`Tests/test_cubeview_ds9.py:143`) and add a regression
   test feeding the captured multiplexed string through `parse_coordinates`.

## 1.9 Immediate workaround

Quit every DS9 window, confirm none remain:

```bash
xpaget xpans      # should report: no 'xpaget' access points match template: xpans
ps aux | grep [d]s9
```

then start exactly one `ds9 &` and open the CubeViewer.

### Two environment notes worth passing on

- **`~/.local/bin` is not in `XPA_SEARCH_DIRS`.** `cubeViewDS9.py:45-52` lists `/usr/local/bin`,
  `/opt/homebrew/bin`, `/opt/local/bin`, `/usr/bin` and the SAOImageDS9 bundle — but this
  machine's self-built `xpaget`/`xpaset`/`xpans` live in `~/.local/bin`. They resolve today
  only via `$PATH` (`shutil.which`), so launching the GUI from Finder or a bundled app — which
  does not inherit the shell `$PATH` — would fail with "XPA client tools not found" instead.
  `CUBEVIEWER_XPA_DIR` is the documented override.
- **`targets()` mis-parses the no-instance case.** `xpaget xpans` returns
  `XPA$ERROR no 'xpaget' access points match template: xpans` on **stdout** with **exit 0**,
  so `_run` does not raise (its guard checks stderr) and `parse_xpans_line` splits it into ≥2
  fields, happily yielding a fake target `('XPA$ERROR', 'no')`. This must be fixed *before*
  `targets()` is relied on for fix #1.

---

# 2. Cosmetic — `VerifyWarning: Card is too long, comment will be truncated`

**Unrelated to §1.** Harmless in itself, but it masks a genuinely dangerous cousin (§2.3).

## 2.1 Root cause

A single over-long FITS card, written on every "send to DS9" of an RSS that has header
pointing but no refined WCS — i.e. the normal sunburst case:

```python
# CubeViewer/cubeViewRSS.py:415
meta['WCSFRAME'] = ('sky', 'Celestial WCS from header pointing (initial guess)')
```

It is copied key-by-key into the outgoing header:

```python
# CubeViewer/cubeViewLlamas.py:774-782
header = fits.Header()
_wcs_prefixes = ('CRPIX', 'CRVAL', 'CDELT', 'CTYPE', 'CUNIT', 'CD', 'PC',
                 'CROTA', 'LONPOLE', 'LATPOLE', 'RADESYS', 'WCSAXES', 'EQUINOX')
for key, value in meta.items():
    if key == 'contributions' or key.upper().startswith(_wcs_prefixes):
        continue
    header[key] = value
```

Note `_wcs_prefixes` contains `WCSAXES`, not `WCS` — so `WCSFRAME` is **not** filtered out.
astropy formats and warns when `DS9.set_fits` serialises the HDUList
(`cubeViewDS9.py:341-343`, `hdulist.writeto(buffer)`).

Card arithmetic: 8 (keyword) + 2 (`= `) + 20 (string value padded) + 3 (` / `) + 50 (comment)
= **83 > 80**. The value is a short string, so astropy cannot invoke the `CONTINUE`
convention and truncates the comment instead.

Confirmed against the installed astropy:

```
comment len  = 50
image len    = 80
image        = "WCSFRAME= 'sky     '           / Celestial WCS from header pointing (initial gue"
```

Three characters (`ss)`) are lost. Pixel data and WCS are unaffected — DS9 renders correctly.

Sibling branches are safe and do not warn: `cubeViewRSS.py:408` (69 cols),
`cubeViewRSS.py:417` (73 cols), `cubeViewCube.py:92-94`, and `hex_header_keys()`
(`Image/WhiteLightModule.py:81-99`).

## 2.2 Fix

Shorten the comment to ≤46 characters, e.g. `'Celestial WCS from header pointing'`.

## 2.3 Related pre-existing truncations in the written products

The sunburst files **on disk open with zero warnings** — the truncation already happened at
write time and the damage is baked in. Three cards were silently clipped by the pipeline:

| Keyword | Source | Comment chars lost |
|---|---|---|
| `SKYPED` | `Sky/skyPedestal.py:432` | `ed` (2) |
| `SKYDVORD` | `Sky/skyConfig.py:213-214` | `dth)` (4) |
| `LAMPONLY` | `Flat/fibreFlat.py:1372-1373` | **37 of 39** |

**`LAMPONLY` is the one to act on.** In
`sunburst/extractions/flat/fibre_flat_corrections.fits` it reads:

```
LAMPONLY= 'red_1_A,red_1_B,red_2_A,red_2_B,red_3_A,red_3_B,red_4_A,red_4_B' / be
```

The 63-character value leaves 2 columns for the comment. **If a ninth benchside is ever added
to the fallback list, the value itself will overflow and be truncated** — silently corrupting
the record of which benchsides used the lamp-only throughput fallback. That matters: this list
is precisely the inter-benchside absolute-scale provenance. It should use the OGIP long-string
`CONTINUE` convention, exactly as `SKYPROV` in the `SKYMASK` extension already does correctly.

Three further cards sit at exactly 80 columns — one extra character breaks them:

- `SKYDVGAT` — `Sky/skyConfig.py:215-216`
- `NSKYNONE` — `File/llamasRSS.py:628-629`
- `FLATSTAT` — `reduce.py:1299-1301`; its **value grows with the counts**, so e.g.
  `'240C/10S/5E'` would overflow

Note that astropy emits these via `warnings`, not `logging`, so they never reach the pipeline
logs — `grep -c "Card is too long"` is 0 in both sunburst run logs.

---

# 3. Verification recipes

## Bug 1 — without touching DS9

```python
from llamas_pyjamas.CubeViewer.cubeViewDS9 import parse_coordinates

parse_coordinates(
    "XPA$BEGIN DS9:ds9 7f000001:52307\n0.5 0.5\nXPA$END   DS9:ds9 7f000001:52307\n"
    "XPA$BEGIN DS9:ds9 7f000001:64510\n224.53958 272.76042\nXPA$END   DS9:ds9 7f000001:64510"
)
# -> DS9Error: Un-parseable coordinates from DS9: ...
```

**End-to-end:** start two `ds9 &`, confirm both appear in `xpaget xpans`, open the CubeViewer
on a sunburst RSS, send to DS9, enable **Pick** → reproduces. Quit one DS9, retry → picking
works.

## Bug 2

```python
import warnings
from astropy.io import fits

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    h = fits.Header()
    h['WCSFRAME'] = ('sky', 'Celestial WCS from header pointing (initial guess)')
    h.cards['WCSFRAME'].image
print([str(x.message) for x in w])
# -> ['Card is too long, comment will be truncated.']
```

---

# 4. Environment as observed

```
xpaget / xpaset / xpans   /Users/slh/.local/bin/        (self-built, 2026-07-28)
ds9                       /opt/local/bin/ds9            version 8.6 (MacPorts)
XPA_METHOD                unset  -> inet loopback, matching the 7f000001:PORT ids
pyds9                     NOT installed (conda llamas_data_reduction_clean, and system python)
conda env                 llamas_data_reduction_clean
```

At the time of analysis no DS9 process and no `xpans` name server were running, so the two
reported instances were live only during the user's session.

## Sunburst data inspected

`/Users/slh/Documents/Projects/Magellan_dev/LLAMAS/testing/sunburst/extractions` — 6 science
RSS files (3 frames × blue/green), 11 HDUs each, 2389 fibres × 2048 pixels:

```
0 PRIMARY   1 SKYSUB   2 ERROR   3 MASK   4 COUNTS   5 SKY
6 WAVE      7 FWHM     8 FIBERMAP (BinTable, 2389 rows)   9 SKYRESID   10 SKYMASK
```

The PRIMARY headers carry `RA = 150.19595833333332`, `DEC = -7.558027777777777`,
`OBJECT = 'GD108'` and no refined-WCS solution — which is exactly the branch condition
(`refined_wcs is None` and `ra/dec is not None`) that selects the over-long `WCSFRAME` comment
at `cubeViewRSS.py:415`. This confirms §2.1 is on the live code path for this dataset.
