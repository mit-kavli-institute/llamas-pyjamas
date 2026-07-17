"""Unit tests for CubeViewer crosshair picking and the spectrum panel.

These need no DS9 and no display: a fake scene stands in for the data, and Qt runs under the
offscreen platform. What they pin down is the behaviour that keeps the GUI responsive —
selections are deduplicated on element identity, not position — and the panel's handling of
empty selections and outlier-dominated flux.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_cubeview_pick`).
"""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PyQt6.QtWidgets import QApplication

from llamas_pyjamas.CubeViewer.cubeViewPick import MARKER_TAG, ElementPicker
from llamas_pyjamas.CubeViewer.cubeViewScene import Spectrum, SpectralScene
from llamas_pyjamas.CubeViewer.cubeViewSpecPlot import SpectrumPanel

_app = QApplication.instance() or QApplication([])


class FakeScene(SpectralScene):
    """Two elements: pixels with x < 100 are 'A', x < 200 are 'B', beyond that nothing."""

    channels = ('green',)

    def __init__(self):
        self.spectra_calls = 0

    def _which(self, x):
        return 'A' if x < 100 else ('B' if x < 200 else None)

    def wavelength_range(self, channel=None):
        return (4000.0, 5000.0)

    def collapse(self, wave_min, wave_max, channels=None):
        raise NotImplementedError

    def element_at(self, x_pix, y_pix):
        return self._which(x_pix)

    def spectra_at(self, x_pix, y_pix):
        self.spectra_calls += 1
        return self.spectra_of([e for e in [self._which(x_pix)] if e is not None])

    def spectra_of(self, elements):
        return [Spectrum(wave=np.linspace(4000, 5000, 10), flux=np.ones(10),
                         channel='green', label=f'element {e}') for e in elements]

    def elements_within(self, x_pix, y_pix, radius_pix):
        self.spectra_calls += 1
        element = self._which(x_pix)
        return [] if element is None else [element]

    def region_for(self, elements):
        return '' if not elements else 'image\n' + '\n'.join(
            f'circle({i},2,3) # color=cyan' for i, _ in enumerate(elements))

    def marker_region(self, x_pix, y_pix):
        return '' if self._which(x_pix) is None else 'image; circle(1,2,3)'


class NullDS9:
    def set(self, *a, **k):
        pass


def test_picker_dedupes_on_element_not_position():
    scene = FakeScene()
    picker = ElementPicker(NullDS9(), scene)
    seen = []
    picker.selectionChanged.connect(seen.append)

    picker._on_moved(10, 10)
    picker._on_moved(20, 10)      # still element A
    picker._on_moved(30, 10)      # still element A
    assert len(seen) == 1, f'moving within one element must emit once, got {len(seen)}'
    assert scene.spectra_calls == 1, 'the expensive path must not run per poll'

    picker._on_moved(150, 10)     # now element B
    assert len(seen) == 2
    assert scene.spectra_calls == 2


def test_picker_emits_empty_when_leaving_the_field():
    scene = FakeScene()
    picker = ElementPicker(NullDS9(), scene)
    seen = []
    picker.selectionChanged.connect(seen.append)

    picker._on_moved(10, 10)
    picker._on_moved(500, 10)     # off the field
    assert seen[-1] == [], 'leaving the field must clear the selection'
    # and staying off it must not re-emit
    picker._on_moved(600, 10)
    assert len(seen) == 2


def test_picker_reselects_after_scene_swap():
    # Element ids are scene-specific, so a swap must not let a stale id suppress the next pick.
    scene = FakeScene()
    picker = ElementPicker(NullDS9(), scene)
    seen = []
    picker.selectionChanged.connect(seen.append)
    picker._on_moved(10, 10)
    assert len(seen) == 1

    picker.set_scene(FakeScene())
    picker._on_moved(10, 10)      # same position, same id 'A', but a different scene
    assert len(seen) == 2, 'a scene swap must invalidate the remembered element'


def test_accumulate_grows_and_toggles():
    scene = FakeScene()
    picker = ElementPicker(NullDS9(), scene)
    seen = []
    picker.selectionChanged.connect(seen.append)
    picker.set_accumulate(True)

    picker._on_moved(10, 10)       # element A
    assert picker._chosen == ['A']
    picker._on_moved(150, 10)      # element B
    assert picker._chosen == ['A', 'B']
    assert len(seen[-1]) == 1, 'members are summed into one spectrum per channel'
    assert seen[-1][0].weight == 2.0

    picker._on_moved(10, 10)       # revisiting A drops it
    assert picker._chosen == ['B']

    picker.clear_aperture()
    assert picker._chosen == []
    assert seen[-1] == []


def test_accumulate_off_restores_single_pick():
    scene = FakeScene()
    picker = ElementPicker(NullDS9(), scene)
    seen = []
    picker.selectionChanged.connect(seen.append)
    picker.set_accumulate(True)
    picker._on_moved(10, 10)
    picker._on_moved(150, 10)
    assert len(picker._chosen) == 2

    picker.set_accumulate(False)   # must drop the aperture, not keep summing it
    assert picker._chosen == []
    picker._on_moved(10, 10)
    assert seen[-1][0].label == 'element A'


def test_radius_change_forces_recompute():
    # The selection depends on the radius, so changing it must not be swallowed by the
    # position-based dedupe.
    scene = FakeScene()
    picker = ElementPicker(NullDS9(), scene)
    seen = []
    picker.selectionChanged.connect(seen.append)
    picker._on_moved(10, 10)
    assert len(seen) == 1
    picker.set_radius(5.0)
    picker._on_moved(10, 10)       # same position, new radius
    assert len(seen) == 2


def test_marker_tagging_covers_every_shape():
    # An aperture marker is one shape per fibre; tagging only the last would strand the rest
    # on the next `regions group delete`.
    region = 'image\ncircle(1,2,3) # color=cyan\ncircle(4,5,6) # color=cyan'
    tagged = ElementPicker._tagged(region)
    assert tagged.count(f'tag={{{MARKER_TAG}}}') == 2
    assert tagged.splitlines()[0] == 'image', 'the coordinate system line carries no tag'


def test_marker_tagging():
    tagged = ElementPicker._tagged('image; polygon(1,2,3,4) # color=cyan')
    assert f'tag={{{MARKER_TAG}}}' in tagged
    # A region with no comment section still gets one.
    assert '#' in ElementPicker._tagged('image; circle(1,2,3)')
    # Already-tagged regions are left alone.
    once = ElementPicker._tagged('image; circle(1,2,3) # tag={cubeview}')
    assert once.count('tag=') == 1


def test_panel_plots_and_clears():
    panel = SpectrumPanel()
    assert len(panel.axes.lines) == 0
    panel.set_spectra([Spectrum(wave=np.linspace(4000, 5000, 50), flux=np.ones(50),
                                channel='green', label='2B fibre 7')])
    assert len(panel.axes.lines) == 1
    assert panel.title.text() == '2B fibre 7'

    panel.set_spectra([])
    assert len(panel.axes.lines) == 0
    assert panel.title.text() == 'No selection'


def test_panel_autoscale_ignores_spikes():
    # A cosmic ray must not flatten the continuum into a line at zero.
    flux = np.ones(1000)
    flux[500] = 1e6
    panel = SpectrumPanel()
    panel.set_spectra([Spectrum(wave=np.linspace(4000, 5000, 1000), flux=flux,
                                channel='red', label='spike')])
    low, high = panel.axes.get_ylim()
    assert high < 100, f'a single 1e6 spike must not set the y range (got {high})'


def test_panel_channel_toggle():
    panel = SpectrumPanel()
    panel.set_spectra([
        Spectrum(wave=np.linspace(3500, 4800, 10), flux=np.ones(10), channel='blue', label='f'),
        Spectrum(wave=np.linspace(4800, 7000, 10), flux=np.ones(10), channel='green', label='f'),
    ])
    assert len(panel.axes.lines) == 2
    panel._visible['blue'].setChecked(False)
    assert len(panel.axes.lines) == 1
    panel._visible['blue'].setChecked(True)
    assert len(panel.axes.lines) == 2


def test_panel_wavelength_range_zooms_and_scales_within_it():
    # Setting the white-light window must zoom the plot to it, and the y range must follow
    # the samples inside that window -- scaling to the whole spectrum would flatten the slice.
    wave = np.linspace(4000, 7000, 3000)
    flux = np.ones(3000)
    flux[wave > 6000] = 500.0            # a bright region outside the window of interest
    panel = SpectrumPanel()
    panel.set_spectra([Spectrum(wave=wave, flux=flux, channel='green', label='f')])
    panel.set_wavelength_range(4000, 5000)

    assert panel.axes.get_xlim() == (4000.0, 5000.0)
    assert panel.axes.get_ylim()[1] < 10, 'y must scale to the window, not the whole spectrum'

    panel.clear_wavelength_range()
    assert panel.axes.get_ylim()[1] > 10, 'clearing restores full-extent scaling'


def test_panel_wavelength_range_survives_empty_window():
    panel = SpectrumPanel()
    panel.set_spectra([Spectrum(wave=np.linspace(4000, 5000, 100), flux=np.ones(100),
                                channel='green', label='f')])
    panel.set_wavelength_range(8000, 9000)   # no samples here
    assert panel.axes.get_xlim() == (8000.0, 9000.0), 'keep the requested range so the '\
                                                      'emptiness is visible'


def test_panel_survives_all_masked_spectrum():
    panel = SpectrumPanel()
    panel.set_spectra([Spectrum(wave=np.linspace(4000, 5000, 10), flux=np.full(10, np.nan),
                                channel='green', label='dead')])
    assert len(panel.axes.lines) == 0, 'an all-NaN spectrum plots nothing but must not raise'


if __name__ == "__main__":
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
