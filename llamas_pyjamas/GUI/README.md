# GUI — QuickLook interface

The PyQt6 QuickLook GUI: browse a night's frames, extract on demand, and inspect white-light
images and spectra. It uses **DS9** as the image display, driven over XPA.

| File | Role |
|---|---|
| `obslog.py` | **Entry point.** `MainWindow`, `HeaderWindow`, `PlotWindow`, `ImageRegions` |
| `obslog_qt.py`, `header_qt.py` | Generated Qt classes for `obslog.ui` / `headerWidget.ui` |
| `obslog.ui`, `headerWidget.ui` | Qt Designer layouts |
| `guiExtract.py` | The extraction the GUI drives — `GUI_extract()`, `process_trace()`, `ExtractLlamasCube()`, `select_bias_for_extension()`, `compute_detector_background()`. Also imported by `reduce.py` |

## Running it

DS9 must already be running — every pick is a click in DS9.

```bash
cd llamas-pyjamas/llamas_pyjamas/GUI
conda activate myenv
ds9 &
python obslog.py
```

The GUI produces the same white-light images as the LLAMAS observing GUI, plus extracted spectra
for quick inspection. Striping in those images usually means the master bias dates from well
before the science frames — rebuild it with `../Scripts/update_bias_master.py`.

For post-reduction work — registration, combining dithers, science extraction — use the
**CubeViewer** (`python -m llamas_pyjamas.CubeViewer`), not this GUI. See
[`docs/workflow/`](../../docs/workflow/README.md).

## Background subtraction

[`../Tutorials/GUI_extract_background_subtraction.md`](../Tutorials/GUI_extract_background_subtraction.md)
covers how `compute_detector_background()` works and when to adjust it.

## Note

This package has no `__init__.py` and needs a display, so it is **excluded from the Sphinx API
build** — this README is its documentation.
