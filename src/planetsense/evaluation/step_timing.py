"""Gait timing estimated from a worn device's own accelerometer signal.

The earlier study removes the gait's harmonics by order tracking on the
exact gait phase (touchdowns of the prescribed walker).  A real watch or phone
does not know its wearer's touchdowns; it has to estimate its own gait clock.
This module is that estimate, using nothing but the device's triaxial signal:

1. events: peaks of the envelope of the high-passed acceleration magnitude
   (heel-strike and other limb transients), timed to a fraction of a sample by
   parabolic interpolation, doubles closer than half the event period merged
   (``detected_events``);
2. events per stride: the 3-axis signal repeats once per stride (arm and leg
   swing reverse at every step), so the number of events per stride is the
   candidate n in {1, 2, 4} whose lag n x (event interval) gives the highest
   autocorrelation, decided once per walker and device position from all its
   walks; near-ties go to the larger count (``stride_scores``,
   ``choose_events_per_stride``);
3. counting without slips: each event gets the whole count nearest to n times
   the phase of the stride-rate swing (Hilbert phase of the dominant direction
   in a band around the stride frequency), relative to the average offset of
   the neighbouring events, so an irregular step neither adds nor loses a count;
   the count is interpolated between events (``device_event_count``);
4. gait clock: step count = 2 x events / (events per stride), divided by the
   nominal step frequency (``device_gait_clock``), as the exact clock of
   records.gait_clock.

``true_stride_phase`` (from the simulation's touchdowns) is used only to
measure the estimate's accuracy.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

CANDIDATES = (1, 2, 4)           # detected events per stride
STRIDE_RANGE_S = (0.7, 3.5)      # plausible stride periods (s)


def device_input(rec, sensor, noise_ug=75.0):
    """What the device delivers over the whole record: its specific force (3 axes) plus
    white sensor noise of ``noise_ug`` ug/sqrt(Hz), seeded by record name and sensor."""
    from ..sensor.imu import ug_to_si, white_noise
    X = np.asarray(rec["f"], float)[:, list(rec["sensors"]).index(sensor), :].copy()
    rng = np.random.default_rng(zlib.crc32(f"{rec['name']}|{sensor}|step-timing".encode()))
    for a in range(3):
        X[:, a] += white_noise(len(X), float(rec["fs"]), ug_to_si(noise_ug), rng)
    return X


def pooled_stride(recs, sensor, noise_ug=75.0):
    """Events per stride and stride period of one device position, pooled over walks of
    the same walker (records and their twins); also each walk's own stride period
    (events per stride x its mean event interval)."""
    Tev, sc = [], []
    for r in recs:
        X = device_input(r, sensor, noise_ug)
        _, T = detected_events(X, float(r["fs"]))
        Tev.append(T)
        sc.append(stride_scores(X, float(r["fs"]), T))
    n_e, tot = choose_events_per_stride(sc)
    per = [n_e * T for T in Tev]
    return dict(n_e=int(n_e), stride_s=float(np.median(per)), per_walk_stride_s=[float(v) for v in per], scores=tot)


def envelope(X, fs, hp_hz=12.0, env_hz=8.0):
    """Envelope of the high-passed acceleration magnitude (limb transients)."""
    a = np.linalg.norm(X, axis=1)
    hp = sosfiltfilt(butter(4, hp_hz, "high", fs=fs, output="sos"), a)
    return sosfiltfilt(butter(2, env_hz, "low", fs=fs, output="sos"), np.abs(hp))


def event_period(e, fs, lo=0.25, hi=1.6, rel=0.6):
    """Period of the envelope's events: first autocorrelation peak within [lo, hi] s whose
    height reaches ``rel`` of the highest one."""
    x = e - e.mean()
    n = len(x)
    F = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(F * np.conj(F))[:n]
    ac /= ac[0]
    i0, i1 = int(lo * fs), min(int(hi * fs), n - 1)
    pk, _ = find_peaks(ac[i0:i1])
    if not len(pk):
        raise ValueError("no periodicity in the envelope")
    h = ac[i0:i1][pk]
    j = pk[np.nonzero(h >= rel * h.max())[0][0]]
    return (i0 + j) / fs


def detected_events(X, fs):
    """Event times (s, sub-sample) and the mean event interval (s).

    Envelope peaks at least 0.6 event periods apart; doubles closer than half a period
    merged (the stronger kept); peak times refined by a parabola through the three
    samples around each peak.  The mean interval counts a gap of about m periods as m
    intervals."""
    e = envelope(X, fs)
    Ts = event_period(e, fs)
    pk, _ = find_peaks(e, distance=max(1, int(0.6 * Ts * fs)))
    if len(pk) < 4:
        raise ValueError("too few events detected")
    keep = [0]
    for i in range(1, len(pk)):
        if (pk[i] - pk[keep[-1]]) / fs < 0.5 * Ts:
            if e[pk[i]] > e[pk[keep[-1]]]:
                keep[-1] = i
        else:
            keep.append(i)
    pk = pk[keep]
    pk = pk[(pk > 0) & (pk < len(e) - 1)]
    a, b, c = e[pk - 1], e[pk], e[pk + 1]
    den = a - 2 * b + c
    ok = np.abs(den) > 1e-12
    d = np.zeros(len(pk))
    d[ok] = 0.5 * (a[ok] - c[ok]) / den[ok]
    te = (pk + np.clip(d, -0.5, 0.5)) / fs
    dt = np.diff(te)
    n_int = np.maximum(1, np.round(dt / float(np.median(dt))))
    return te, float((te[-1] - te[0]) / n_int.sum())


def _vector_autocorr(X, fs, band=(0.3, 6.0)):
    """Autocorrelation (biased, normalised to 1 at lag 0) of the band-passed 3-axis
    signal, summed over the axes."""
    B = sosfiltfilt(butter(2, list(band), "bandpass", fs=fs, output="sos"), X - X.mean(0), axis=0)
    n = len(B)
    ac = np.zeros(n)
    for j in range(B.shape[1]):
        F = np.fft.rfft(B[:, j], 2 * n)
        ac += np.fft.irfft(F * np.conj(F))[:n]
    return ac / ac[0]


def stride_scores(X, fs, Tev):
    """Repetition of the vector signal after n events, n in CANDIDATES: the highest
    autocorrelation within 15 % of the lag n x Tev; NaN for lags outside
    STRIDE_RANGE_S."""
    ac = _vector_autocorr(X, fs)
    out = {}
    for n in CANDIDATES:
        L = n * Tev
        if not (STRIDE_RANGE_S[0] <= L <= STRIDE_RANGE_S[1]):
            out[n] = np.nan
            continue
        a, b = int(0.85 * L * fs), min(int(1.15 * L * fs) + 1, len(ac))
        out[n] = float(ac[a:b].max()) if b > a else np.nan
    return out


def choose_events_per_stride(score_list, tie=0.02):
    """Events per stride from the scores of several walks of the same walker and device
    position (mean over the walks); near-ties go to the larger count: a clock that runs
    slow by a whole factor only makes the harmonic comb denser (the lines that are not
    there are not detected), one that runs fast drops the odd stride harmonics."""
    tot = {}
    for n in CANDIDATES:
        v = np.array([s[n] for s in score_list], float)
        tot[n] = float(np.nanmean(v)) if np.isfinite(v).any() else -np.inf
    best = max(tot.values())
    return max(n for n in CANDIDATES if tot[n] >= best - tie), tot


def phase_from_fundamental(X, fs, f_stride, rel_band=0.35):
    """Stride phase (strides, continuous) from the Hilbert phase of the stride-rate
    oscillation along the signal's dominant direction in that band."""
    sos = butter(2, [(1 - rel_band) * f_stride, (1 + rel_band) * f_stride], "bandpass", fs=fs, output="sos")
    B = sosfiltfilt(sos, X - X.mean(0), axis=0)
    _, V = np.linalg.eigh(B.T @ B)
    s = B @ V[:, -1]
    ph = np.unwrap(np.angle(hilbert(s))) / (2 * np.pi)
    if ph[-1] < ph[0]:
        ph = -ph
    return ph


def device_event_count(X, fs, n_e, f_str, half=8):
    """Continuous event count psi_e(t): each detected event gets the whole count nearest
    to n_e times the stride-rate swing phase, relative to the average offset of its
    ``half`` neighbours on each side; one event per count, counts increasing; linear
    interpolation between events and extrapolation at the mean rate.
    Returns (psi_e, event times used, their counts)."""
    te, _ = detected_events(X, fs)
    ph = phase_from_fundamental(X, fs, f_str)
    t = np.arange(len(X)) / fs
    q = n_e * np.interp(te, t, ph)
    z = np.exp(2j * np.pi * q)
    c = np.array([np.angle(z[max(0, j - half):j + half + 1].mean()) for j in range(len(q))])
    c = np.unwrap(c) / (2 * np.pi)
    k = np.round(q - c)
    r = np.abs(q - c - k)
    sel = []
    for j in range(len(te)):
        if sel and k[j] <= k[sel[-1]]:
            if k[j] == k[sel[-1]] and r[j] < r[sel[-1]]:
                sel[-1] = j
            continue
        sel.append(j)
    te, k = te[sel], k[sel] - k[sel][0]
    psi = np.interp(t, te, k)
    rate = (k[-1] - k[0]) / (te[-1] - te[0])
    lo, hi = t < te[0], t > te[-1]
    psi[lo] = (t[lo] - te[0]) * rate
    psi[hi] = k[-1] + (t[hi] - te[-1]) * rate
    return psi, te, k


def device_gait_clock(psi_e, n_e, fp):
    """Gait clock (s) of the harmonic-removal step from the device's event count:
    step count 2 psi_e / n_e divided by the nominal step frequency ``fp``."""
    return 2.0 * np.asarray(psi_e, float) / float(n_e) / float(fp)


def true_stride_phase(rec):
    """Stride phase from the simulation's touchdown events (reference only)."""
    t = np.asarray(rec["t"], float)
    te = np.asarray(rec["events"], float).reshape(-1, 3)[:, 0]
    j = np.arange(len(te), dtype=float)
    psi = np.interp(t, te, j)
    lo, hi = t < te[0], t > te[-1]
    psi[lo] = (t[lo] - te[0]) / (te[1] - te[0])
    psi[hi] = j[-1] + (t[hi] - te[-1]) / (te[-1] - te[-2])
    return psi / 2.0


def timing_error(ph_dev, ph_ref, T_stride, nbins=40):
    """Error of a device stride phase against the reference over the same samples, in ms:
    (total rms after removing the median offset, rms after also removing the part that
    repeats every stride, largest absolute error).  A stride-periodic warp of the clock
    does not hinder harmonic removal (the gait stays periodic in the warped clock); the
    remainder does."""
    d = np.asarray(ph_dev, float) - np.asarray(ph_ref, float)
    d = d - np.median(d)
    tot = float(np.sqrt(np.mean(d ** 2)) * T_stride * 1e3)
    b = np.minimum(np.floor((np.asarray(ph_ref, float) % 1.0) * nbins).astype(int), nbins - 1)
    per = np.array([d[b == i].mean() if np.any(b == i) else 0.0 for i in range(nbins)])
    rnd = d - per[b]
    return tot, float(np.sqrt(np.mean(rnd ** 2)) * T_stride * 1e3), float(np.max(np.abs(d)) * T_stride * 1e3)
