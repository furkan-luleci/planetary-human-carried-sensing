import numpy as np

from planetsense.excitation.multisine import make_multisine
from planetsense.simulation.standing import make_standing


def test_repeatability_fixed_seed(modal_deck):
    """Brief 17.5: the same configuration and seed give bit-identical output;
    a different seed gives a different excitation."""
    def run(seed):
        sc = make_standing(g=3.71, deck=modal_deck)
        ms = make_multisine(9, T=4.0, rms=60.0, seed=seed)
        return sc.system.simulate(sc.y_eq.copy(), T=0.4, dt=2.5e-4, out_dt=1e-3, forcing=ms)
    a, b, c = run(7), run(7), run(8)
    for k in ("f", "w", "fn", "eta", "energy"):
        assert np.array_equal(a[k], b[k]), k
    assert not np.array_equal(a["f"], c["f"])


def test_multisine_is_time_grid_independent():
    """The excitation is defined in continuous time: sampling it on a finer
    grid reproduces the coarse samples exactly (needed for time-step studies)."""
    ms = make_multisine(3, T=4.0, rms=100.0, seed=1)
    coarse = ms.sample(1e-3, 2000)
    fine = ms.sample(2.5e-4, 8000)
    assert np.abs(fine[:, ::4] - coarse).max() < 1e-9
    direct = ms.sample(1.1e-3, 50)                     # non-commensurate grid -> direct sum
    t = 1.1e-3 * np.arange(50)
    ref = (ms.amp[0][None, :] * np.cos(2 * np.pi * np.outer(t, ms.k) / ms.T + ms.phase[0])).sum(axis=1)
    assert np.abs(direct[0] - ref).max() < 1e-9
    assert np.allclose(ms.rms(), 100.0)
    assert abs(coarse[0].std() / 100.0 - 1) < 0.05
