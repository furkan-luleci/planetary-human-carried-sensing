"""Unit tests of the blind identifier: peak picking on a device spectrum (pick_peaks) and one-to-one
matching of picked lines to the true natural frequencies (match_peaks).  They replace a placeholder
that was written before the identifier existed (2026-11-22)."""
import inspect

import numpy as np

from planetsense.identification.spectral import match_peaks, pick_peaks

KW = dict(fmin=1.0, fmax=30.0, prominence_db=4.0, background_hz=4.0)   # the study's frozen settings


def spectrum(peaks, floor=1.0):
    """A flat background plus Lorentzian lines (f0, height, half-width), on a 0.05 Hz grid."""
    f = np.arange(0.0, 40.0, 0.05)
    S = floor * np.ones_like(f)
    for f0, h, w in peaks:
        S = S + h / (1.0 + ((f - f0) / w) ** 2)
    return f, S


def test_finds_every_line_well_above_its_surroundings():
    f, S = spectrum([(3.3, 30.0, 0.05), (7.4, 20.0, 0.06), (12.6, 15.0, 0.08)])
    found = [p[0] for p in pick_peaks(f, S, **KW)]
    m = match_peaks(found, [3.3, 7.4, 12.6])
    assert (m["tp"], m["fp"], m["fn"]) == (3, 0, 0)
    assert max(abs(e) for _t, _q, e in m["pairs"]) < 0.03


def test_line_below_the_prominence_threshold_is_not_reported():
    f, S = spectrum([(5.2, 0.5, 0.05)])                 # 10 log10(1.5) = 1.8 dB above the background
    assert pick_peaks(f, S, **KW) == []


def test_harmonic_rejection_removes_gait_lines_and_keeps_others():
    stride = 0.9
    gait = [(k * stride, 25.0, 0.04) for k in range(1, 23)]          # gait lines up to 19.8 Hz
    f, S = spectrum(gait + [(7.4, 25.0, 0.04)])                     # 0.2 Hz from the nearest gait line
    without = [p[0] for p in pick_peaks(f, S, **KW)]
    kept = [p[0] for p in pick_peaks(f, S, harmonics_of=stride, harmonic_halfwidth=0.15, max_harmonic_hz=20.0, **KW)]
    assert len(without) > 10
    assert match_peaks(kept, [7.4])["tp"] == 1
    assert all(abs(x / stride - round(x / stride)) * stride > 0.15 for x in kept)


def test_matching_is_one_to_one_within_the_tolerance():
    # 1.78 and 1.86 Hz both lie within 0.15 Hz of 1.80 Hz: only the closer one counts.  7.00 Hz lies within
    # 3 % of 7.21 Hz (0.216 Hz); 12.80 Hz lies outside 3 % of 12.30 Hz (0.369 Hz).
    m = match_peaks([1.86, 1.78, 7.00, 12.80], [1.80, 7.21, 12.30])
    assert (m["tp"], m["fp"], m["fn"]) == (2, 2, 1)
    assert sorted(round(p[1], 2) for p in m["pairs"]) == [1.78, 7.0]


def test_identifier_sees_only_the_spectrum():
    # blind: the picking takes the spectrum and its own settings; true frequencies enter only the scoring
    params = inspect.signature(pick_peaks).parameters
    assert not any(n in params for n in ("truth", "modes", "true_frequencies", "f_true"))
