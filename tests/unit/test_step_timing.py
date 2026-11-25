"""Device-estimated gait clock (Phase J, src/planetsense/evaluation/step_timing.py) and the
per-side processing path (src/planetsense/evaluation/baseline_match.py)."""
import numpy as np

import pytest

pytest.importorskip("planetsense.evaluation.legacy_pipeline", reason="needs the earlier study's code (see README)")
from planetsense.evaluation import legacy_pipeline as LP
from planetsense.evaluation import baseline_match as MM
from planetsense.evaluation import step_timing as ST

FS = 200.0


def _device(T=1.12, dur=30.0, phases=(0.0, 0.23, 0.5, 0.73), amps=(3.0, 1.6, 2.6, 1.4), warp=None, seed=1):
    """Synthetic worn-device signal: arm or leg swing once per stride (x), bounce twice
    per stride (z), short high-frequency transients at fixed gait phases, sensor noise.
    ``warp(t)`` gives the true stride phase (default t / T)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    ph = t / T if warp is None else warp(t)
    X = np.zeros((len(t), 3))
    X[:, 0] = 2.0 * np.sin(2 * np.pi * ph)
    X[:, 2] = 9.81 + 0.6 * np.sin(4 * np.pi * ph + 0.4)
    k0 = int(np.floor(ph[0])) - 1
    for k in range(k0, int(np.ceil(ph[-1])) + 1):
        for p, a in zip(phases, amps):
            te = np.interp(k + p, ph, t, left=np.nan, right=np.nan)
            if not np.isfinite(te):
                continue
            m = (t >= te) & (t < te + 0.08)
            tt = t[m] - te
            X[m, 2] += a * np.exp(-tt / 0.012) * np.sin(2 * np.pi * 28.0 * tt)
            X[m, 0] += 0.4 * a * np.exp(-tt / 0.012) * np.sin(2 * np.pi * 31.0 * tt)
    X += 75e-6 * 9.80665 * np.sqrt(FS / 2) * rng.standard_normal(X.shape)
    return t, ph, X


def _clock(X, with_period=False):
    te, Tev = ST.detected_events(X, FS)
    n_e, _ = ST.choose_events_per_stride([ST.stride_scores(X, FS, Tev)])
    psi, _, _ = ST.device_event_count(X, FS, n_e, 1.0 / (n_e * Tev))
    return (n_e, psi / n_e, n_e * Tev) if with_period else (n_e, psi / n_e)


def test_device_clock_tracks_a_regular_gait():
    """Four transients per stride, two strong and two weak: the weak ones may be left
    out (then two events make a stride); the stride period and the clock must be right."""
    t, ph, X = _device()
    n_e, ph_dev, Tstr = _clock(X, with_period=True)
    assert n_e in (2, 4) and abs(Tstr - 1.12) / 1.12 < 0.02
    m = (t > 3) & (t < 27)
    tot, rnd, mx = ST.timing_error(ph_dev[m], ph[m], 1.12)
    assert rnd < 1.0                    # ms; the part that does not repeat every stride
    assert tot < 15.0                   # ms; mostly the warp from unequally spaced events


def test_four_similar_events_per_stride():
    t, ph, X = _device(amps=(3.0, 2.6, 2.8, 2.4), seed=4)
    n_e, ph_dev, Tstr = _clock(X, with_period=True)
    assert n_e == 4 and abs(Tstr - 1.12) / 1.12 < 0.02
    m = (t > 3) & (t < 27)
    assert ST.timing_error(ph_dev[m], ph[m], 1.12)[1] < 1.0


def test_counting_survives_quick_steps():
    """Two quick steps (one stride in 0.75 of the usual time) must not leave the count a
    quarter stride off for the rest of the walk."""
    T = 1.12

    def warp(t):
        rate = np.where((t > 12.0) & (t < 12.0 + 0.75 * T), 1.0 / (0.75 * T), 1.0 / T)
        return np.concatenate([[0.0], np.cumsum(rate[1:] * np.diff(t))])
    t, ph, X = _device(T=T, warp=warp, seed=2)
    _, ph_dev = _clock(X)
    d = ph_dev - ph
    d -= np.median(d[(t > 3) & (t < 11)])
    late = (t > 16) & (t < 27)
    assert np.max(np.abs(d[late])) < 0.03     # strides


def test_two_events_per_stride():
    t, ph, X = _device(phases=(0.0, 0.5), amps=(2.5, 2.5), seed=3)
    n_e, ph_dev = _clock(X)
    assert n_e == 2
    m = (t > 3) & (t < 27)
    assert ST.timing_error(ph_dev[m], ph[m], 1.12)[1] < 1.0


def _synthetic():
    rng = np.random.default_rng(5)
    n = 2600
    t = np.arange(n) / FS
    fp = 1.8
    tau = t * (1 + 0.01 * np.sin(0.3 * t))
    passes, twins = [], []
    for _ in range(2):
        c3, r3 = {}, {}
        for s in ("wristL", "wristR"):
            gait = {ax: 0.8 * np.sin(2 * np.pi * 0.5 * fp * tau + k) + 0.3 * np.sin(2 * np.pi * 1.5 * fp * tau)
                    for k, ax in enumerate("XYZ")}
            deck = {ax: 0.02 * np.sin(2 * np.pi * 7.21 * t + rng.uniform(0, 6)) for ax in "XYZ"}
            c3[s] = {ax: gait[ax] + deck[ax] + 0.01 * rng.standard_normal(n) for ax in "XYZ"}
            r3[s] = {ax: gait[ax].copy() for ax in "XYZ"}
        common = dict(fs=FS, standing=False, trim_s=1.5, fp=fp, tau=tau.copy())
        passes.append(dict(clean3=c3, clean={s: c3[s]["Z"] for s in c3}, **common))
        twins.append(dict(clean3=r3, clean={s: r3[s]["Z"] for s in r3}, **common))
    return passes, twins


def test_each_side_alone_gives_the_two_sided_result():
    """Processing each side as its own ensemble and averaging the first singular values
    afterwards equals the baseline's two-sided processing (Eq. 7)."""
    passes, twins = _synthetic()

    def ens(ps, sensors):
        cfg = dict(LP.BASE_CONFIG, sensors=list(sensors))
        return dict(cfg=cfg, passes=[dict(p) for p in ps], standing=False, grid=np.linspace(0.3, 50, 200))
    with MM._pairing():
        E, B = ens(passes, ["wristL", "wristR"]), ens(twins, ["wristL", "wristR"])
        both = LP.ens_device_spectra(E, "wrist", comb_twin=B)["ens_sv1"]
        side = []
        for s in ("wristL", "wristR"):
            Es, Bs = ens(passes, [s]), ens(twins, [s])
            side.append(LP.ens_device_spectra(Es, "wrist", comb_twin=Bs)["ens_sv1"])
    np.testing.assert_allclose(0.5 * (side[0] + side[1]), both, rtol=1e-12, atol=0.0)
