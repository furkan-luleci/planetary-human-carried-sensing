"""Spectral estimation, transmission estimates and blind frequency recovery.

Three outcomes are kept apart:

  transmission   how much of the deck motion reaches the sensor (a property of
                 the mechanics; estimated without sensor noise)
  visibility     whether a structural peak stands out of the worn sensor's own
                 spectrum, given gait content and sensor noise
  recovery       whether an analyst who does NOT know the structure can read the
                 structural frequencies off the worn sensor alone

Nothing in this module knows the gravity of a record.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import csd, get_window, medfilt


def segment_csd(X, fs, nperseg, noverlap=None):
    """Cross-spectral matrix of the columns of X (n, m): (f, G[nf, m, m]),
    one-sided density, Hann window, constant detrend.  Returns also the number
    of averaged segments."""
    X = np.asarray(X, float)
    n, m = X.shape
    nperseg = int(min(nperseg, n))
    noverlap = nperseg // 2 if noverlap is None else noverlap
    G = None
    for i in range(m):
        for j in range(i, m):
            f, p = csd(X[:, i], X[:, j], fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                       detrend="constant", scaling="density")
            if G is None:
                G = np.zeros((len(f), m, m), complex)
            G[:, i, j] = p
            if j != i:
                G[:, j, i] = np.conj(p)
    nseg = max(1, (n - noverlap) // (nperseg - noverlap))
    return f, G, nseg


def ensemble_csd(records, fs, nperseg):
    """Average the cross-spectral matrix over several records (each (n_i, m)),
    weighting by the number of segments."""
    Gs, w, f = None, 0, None
    for X in records:
        if len(X) < nperseg:
            continue
        f, G, ns = segment_csd(X, fs, nperseg)
        Gs = G * ns if Gs is None else Gs + G * ns
        w += ns
    if Gs is None:
        raise ValueError("no record is long enough for nperseg=%d" % nperseg)
    return f, Gs / w, w


def sv1(G):
    """Largest singular value of the cross-spectral matrix at every frequency
    (the 'first singular value' spectrum of frequency-domain decomposition)."""
    return np.linalg.svd(G, compute_uv=False)[:, 0]


def h1(Gxy, Gxx):
    """H1 transfer estimate and coherence helper: H = Gxy / Gxx."""
    return Gxy / Gxx


def band_power(f, S, f0, half):
    m = np.abs(f - f0) <= half
    return float(np.trapezoid(S[m], f[m])) if m.sum() > 1 else float("nan")


def smooth_background(S, df, width_hz):
    """Running median of log S: a peak-free estimate of the local level."""
    k = int(round(width_hz / df)) | 1
    k = max(k, 3)
    pad = k // 2
    x = np.log(np.maximum(S, 1e-300))
    xp = np.concatenate([np.full(pad, np.median(x[:pad + 1])), x, np.full(pad, np.median(x[-pad - 1:]))])
    return np.exp(medfilt(xp, k)[pad:-pad])


def step_frequency(f, S, fmin=0.4, fmax=3.5):
    """Dominant low-frequency line of a worn-sensor spectrum (the step rate)."""
    m = (f >= fmin) & (f <= fmax)
    return float(f[m][np.argmax(S[m])]) if m.any() else float("nan")


def pick_peaks(f, S, fmin=2.5, fmax=30.0, prominence_db=6.0, background_hz=3.0,
               harmonics_of=None, harmonic_halfwidth=0.25, max_harmonic_hz=None, min_sep_hz=0.5):
    """Blind peak picking on a (first-singular-value) spectrum.

    A candidate is a local maximum that rises at least ``prominence_db`` above
    the running-median background.  Lines within ``harmonic_halfwidth`` of a
    multiple of ``harmonics_of`` (the step rate, itself estimated from the
    same signal) are discarded up to ``max_harmonic_hz``.  Returns a list of
    (frequency, height above background in dB), strongest first, at least
    ``min_sep_hz`` apart.
    """
    f = np.asarray(f, float)
    S = np.asarray(S, float)
    df = f[1] - f[0]
    bg = smooth_background(S, df, background_hz)
    exc = 10.0 * np.log10(np.maximum(S, 1e-300) / bg)
    cand = []
    for k in range(1, len(f) - 1):
        if not (fmin <= f[k] <= fmax):
            continue
        if S[k] > S[k - 1] and S[k] >= S[k + 1] and exc[k] >= prominence_db:
            if harmonics_of and (max_harmonic_hz is None or f[k] <= max_harmonic_hz):
                r = f[k] / harmonics_of
                if abs(r - round(r)) * harmonics_of <= harmonic_halfwidth and round(r) >= 1:
                    continue
            # parabolic refinement of the peak frequency on the log spectrum
            a, b, c = np.log(S[k - 1]), np.log(S[k]), np.log(S[k + 1])
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den != 0 else 0.0
            cand.append((float(f[k] + d * df), float(exc[k])))
    cand.sort(key=lambda p: -p[1])
    out = []
    for p in cand:
        if all(abs(p[0] - o[0]) >= min_sep_hz for o in out):
            out.append(p)
    return out


def match_peaks(found, truth, tol_rel=0.03, tol_abs=0.15):
    """One-to-one matching of identified to true frequencies (closest pairs
    first).  A pair counts if |f - f_true| <= max(tol_abs, tol_rel * f_true).
    Returns dict(tp, fp, fn, precision, recall, f1, pairs=[(f_true, f_found, err)])."""
    found = [float(x) for x in found]
    truth = [float(x) for x in truth]
    pairs = sorted(((abs(a - b), i, j) for i, a in enumerate(truth) for j, b in enumerate(found)))
    ut, uf, res = set(), set(), []
    for d, i, j in pairs:
        if i in ut or j in uf:
            continue
        if d <= max(tol_abs, tol_rel * truth[i]):
            ut.add(i)
            uf.add(j)
            res.append((truth[i], found[j], found[j] - truth[i]))
    tp, fp, fn = len(res), len(found) - len(res), len(truth) - len(res)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, pairs=sorted(res))
