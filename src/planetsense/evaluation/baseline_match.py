"""The Observability Index of the earlier study computed on this project's records
with the baseline's own processing (vendored verbatim in legacy_pipeline.py
and legacy_oi.py).

What this module adds is only the hand-over of the signals:

* records -> the baseline's "ensemble" format: per record, the triaxial
  signals of the left and right devices on the deck crossing (walking:
  centre of mass between the supports; the baseline itself discards the first
  1.5 s) or after the coupled lead-in (standing), with the gait clock and the
  nominal step frequency used for order-tracked gait-harmonic removal;
* the watch is the wrist pair (wrist_L, wrist_R -> baseline wristL, wristR);
  the phone is the trouser-pocket pair (thigh_L, thigh_R), passed in the
  baseline's phone channels (hipL, hipR), so it also gets that phone's noise
  seeds;
* the mean and linear trend of each axis are removed before processing: this
  project's accelerometers report specific force in sensor axes, which
  contains the steady part of gravity, absent from the baseline's signals;
* the matched-pair check of the baseline is replaced (``_same_walk_ours``):
  the baseline's walker is prescribed, so a record and its rigid twin share the
  gait clock exactly; here the walker responds to the deck, so each record
  keeps its own gait clock (same nominal step frequency, same record count);
* the zero-padding length is chosen for the same frequency spacing as the
  baseline (2^17 points at 1 kHz = 0.0076294 Hz; here 26214 points at 200 Hz =
  0.0076295 Hz).

Everything after that (harmonic detection and removal, cross-spectra, first
singular value, record and left/right averaging, smoothing, display grid,
storage at 0.1 dB, second smoothing, index, classes) is the baseline's code.
"""
from __future__ import annotations

import contextlib

import numpy as np
from ..simulation.records import FS_IMU, WALK_TRIM_S, analysis_window, crossing_window, gait_clock
from . import legacy_pipeline as LP
from .legacy_oi import obs_class

DEVICES = {"watch": ("wrist", {"wrist_L": "wristL", "wrist_R": "wristR"}),
           "phone": ("hip", {"thigh_L": "hipL", "thigh_R": "hipR"})}
NFFT = 26214
NOISE_UG = 75.0


def base_cfg(noise_ug: float = NOISE_UG) -> dict:
    c = dict(LP.BASE_CONFIG)
    c.update(sensors=["hipL", "hipR", "wristL", "wristR"], sens_noise_wrist=float(noise_ug),
             sens_noise_hip=float(noise_ug), sens_noise=None)
    return c


def ensemble(recs, standing: bool, fp: float | None = None, noise_ug: float = NOISE_UG, zero: bool = False,
             clocks=None, sensors=None) -> dict:
    """Baseline-format ensemble of the records (in the given order: record i is
    paired with twin i, and gets the noise seeds of the baseline's record i).
    ``zero=True`` keeps only the lengths (a rigid-floor standing twin: sensor
    noise alone).  ``clocks``: per record, a gait clock over the whole record to
    use instead of the exact one (records.gait_clock with ``fp``); ``sensors``:
    the baseline sensor names to process (default all four; one name gives a
    single-side device, so that each side can carry its own clock)."""
    cfg = base_cfg(noise_ug)
    if sensors is not None:
        cfg["sensors"] = list(sensors)
    passes = []
    for ir, r in enumerate(recs):
        m = analysis_window(r) if standing else crossing_window(r)
        trim = 0.0 if standing else WALK_TRIM_S
        i0 = int(trim * FS_IMU)
        X = np.asarray(r["f"], float)[m]
        names = list(r["sensors"])
        clean3 = {}
        for _dev, (_part, mp) in DEVICES.items():
            for ours, base in mp.items():
                A = X[:, names.index(ours), :].copy()
                if zero:
                    A[:] = 0.0
                else:
                    t = np.arange(len(A), dtype=float)
                    for a in range(3):                      # mean and linear trend of the analysed part
                        cf = np.polyfit(t[i0:], A[i0:, a], 1)
                        A[:, a] -= np.polyval(cf, t)
                clean3[base] = dict(X=A[:, 0], Y=A[:, 1], Z=A[:, 2])
        p = dict(clean3=clean3, clean={s: clean3[s]["Z"] for s in clean3}, fs=FS_IMU, standing=standing,
                 trim_s=trim, fp=None if standing else float(fp),
                 tau=None if standing else (gait_clock(r, fp)[m] if clocks is None
                                            else np.asarray(clocks[ir], float)[m]))
        passes.append(p)
    x0 = passes[0]["clean"][cfg["sensors"][0]]
    i0 = LP._trim_index(passes[0], len(x0))
    grid = LP.raw_psd(x0[i0:], FS_IMU, cfg)[0]
    return dict(cfg=cfg, passes=passes, standing=standing, grid=grid)


def _same_walk_ours(flex, twin):
    if len(flex["passes"]) != len(twin["passes"]):
        raise ValueError("matched pair: %d records, %d twins" % (len(flex["passes"]), len(twin["passes"])))
    for i, (pf, pb) in enumerate(zip(flex["passes"], twin["passes"])):
        if LP._has_comb(pf) != LP._has_comb(pb):
            raise ValueError("matched pair: record %d has a comb on one side only" % i)
        if LP._has_comb(pf) and not np.isclose(float(pf["fp"]), float(pb["fp"]), rtol=1e-12, atol=0.0):
            raise ValueError("matched pair: record %d nominal step frequency differs" % i)


@contextlib.contextmanager
def _pairing():
    orig = LP._same_walk
    LP._same_walk = _same_walk_ours
    try:
        yield
    finally:
        LP._same_walk = orig


def device_curves(ens: dict, twin: dict, nfft: int = NFFT) -> dict:
    """Display-grid curves of both devices, encoded as in the baseline explorer
    (0.1 dB steps, 110 dB range), with the device and background spectra."""
    cfg = ens["cfg"]
    fmin, fmax = cfg["psd_fmin"], cfg["psd_fmax"]
    g = np.linspace(fmin, fmax, LP.NF)
    out = dict(fmin=fmin, fmax=fmax)
    with _pairing():
        for dev, (part, _mp) in DEVICES.items():
            d = LP.ens_device_spectra(ens, part, nfft=nfft, comb_twin=twin if not ens["standing"] else None)
            b = LP.ens_device_spectra(twin, part, nfft=nfft, comb_from=ens if not twin["standing"] else None)
            f, P = ens["grid"], np.interp(ens["grid"], d["f"], d["ens_sv1"])          # = ens_part_sv1
            fr, Pr = twin["grid"], np.interp(twin["grid"], b["f"], b["ens_sv1"])
            Pd = LP.smooth_psd(P, f[1] - f[0], cfg["psd_smooth_hz"])
            Prd = LP.smooth_psd(Pr, fr[1] - fr[0], cfg["psd_smooth_hz"])
            c, pk = LP._enc(np.interp(g, f, Pd, left=np.nan, right=np.nan))
            rc, rpk = LP._enc(np.interp(g, fr, Prd, left=np.nan, right=np.nan))
            out[dev] = dict(c=c, pk=pk, rc=rc, rpk=rpk,
                            n_comb_lines=(int(np.mean([len(v) for v in d["comb_lines"].values()]))
                                          if d["comb_lines"] else 0))
    return out


def device_curves_sided(sides: dict, nfft: int = NFFT) -> dict:
    """As ``device_curves``, for devices whose two sides were processed as separate
    single-side ensembles (each side with its own gait clock): ``sides[dev]`` is a
    list of (ensemble, twin) pairs, one per side.  The first singular value of each
    side's record-averaged cross-spectral matrix is averaged over the sides, as the
    baseline does within one ensemble (Eq. 7); everything after is the baseline's."""
    out = None
    for dev, pairs in sides.items():
        ens0, twin0 = pairs[0]
        cfg = ens0["cfg"]
        if out is None:
            out = dict(fmin=cfg["psd_fmin"], fmax=cfg["psd_fmax"])
        g = np.linspace(cfg["psd_fmin"], cfg["psd_fmax"], LP.NF)
        part = DEVICES[dev][0]
        sd, sb, nl = [], [], []
        with _pairing():
            for ens, twin in pairs:
                d = LP.ens_device_spectra(ens, part, nfft=nfft, comb_twin=twin if not ens["standing"] else None)
                b = LP.ens_device_spectra(twin, part, nfft=nfft, comb_from=ens if not twin["standing"] else None)
                sd.append(d["ens_sv1"])
                sb.append(b["ens_sv1"])
                nl += [len(v) for v in d["comb_lines"].values()]
        f, P = ens0["grid"], np.interp(ens0["grid"], d["f"], sum(sd) / len(sd))
        fr, Pr = twin0["grid"], np.interp(twin0["grid"], b["f"], sum(sb) / len(sb))
        Pd = LP.smooth_psd(P, f[1] - f[0], cfg["psd_smooth_hz"])
        Prd = LP.smooth_psd(Pr, fr[1] - fr[0], cfg["psd_smooth_hz"])
        c, pk = LP._enc(np.interp(g, f, Pd, left=np.nan, right=np.nan))
        rc, rpk = LP._enc(np.interp(g, fr, Prd, left=np.nan, right=np.nan))
        out[dev] = dict(c=c, pk=pk, rc=rc, rpk=rpk, n_comb_lines=int(np.mean(nl)) if nl else 0)
    return out


def index_run(curves: dict, refs, bands, Tres: float) -> dict:
    """The baseline's curve_index_run on the encoded curves."""
    r = dict(curves, refs=[[str(lb), float(f)] for lb, f in refs], bands=[[float(a), float(b)] for a, b in bands],
             Tres=float(Tres))
    for dev in DEVICES:                  # the baseline writes its results into the device dicts: work on copies
        if isinstance(r.get(dev), dict):
            r[dev] = dict(r[dev])
    LP.curve_index_run(r, LP.NF)
    return r


def record_window_s(standing: bool, speed: float | None, cfg: dict | None = None):
    """(Trec, Tres) of the baseline explorer's record_window: walking L/v - 1.5 s,
    standing 60 s; Tres = min(Trec, 30 s)."""
    cfg = cfg or base_cfg()
    Trec = float(cfg["stand_T"]) if standing else float(cfg["deck_L"]) / float(speed) - WALK_TRIM_S
    return Trec, min(Trec, float(cfg["psd_seg_s"]))


def classes(pmode: dict, t1: float, t2: float) -> dict:
    return {m: obs_class(v["oi"], t1, t2) for m, v in pmode.items()}


def display(curves_dev: dict):
    """Decoded device and background curves on the display grid (for figures)."""
    g = np.linspace(0.3, 50.0, LP.NF)
    S = LP.display_decode(curves_dev["c"], curves_dev["pk"], LP.NF)
    B = LP.display_decode(curves_dev["rc"], curves_dev["rpk"], LP.NF) if curves_dev.get("rc") else np.zeros(LP.NF)
    return g, S, B
