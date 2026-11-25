import numpy as np
import pytest

from planetsense.simulation.standing import make_standing
from tolerances import TOL

CASES = [(g, deck, pol) for g in (9.81, 3.71, 1.62) for deck in (False, True)
         for pol in ("gravity_scaled", "earth_fixed")]


@pytest.mark.parametrize("g,with_deck,policy", CASES)
def test_gravity_static_support(g, with_deck, policy, modal_deck):
    """Brief 17.5 / 7: for every gravity the static configuration is re-solved;
    the feet carry exactly m*g; the solved state is an equilibrium of the full
    equations; and that equilibrium is stable."""
    sc = make_standing(g=g, deck=modal_deck if with_deck else None, gain_policy=policy)
    s, y, info = sc.system, sc.y_eq, sc.info
    W = sc.human.total_mass * g
    assert abs(info["weight"] - 75.0 * g) < 1e-9
    assert abs(info["sum_fn"] - W) / W < TOL["static_weight_rel"]
    assert (info["fn"] > 0).all()                       # all eight points loaded
    assert np.abs(info["residual_root"]).max() < 1e-8   # no force is needed on the root
    dy = s.rhs(0.0, y, first=True)
    assert np.abs(dy[:s.nx]).max() < TOL["equilibrium_accel"]
    # centre of pressure under the centre of mass
    assert np.abs(info["cop"] - info["com"][:2]).max() < 1e-9
    A, _, _ = s.linearize(y)
    assert np.linalg.eigvals(A).real.max() < 0.0, "standing equilibrium is unstable"


@pytest.mark.parametrize("g", [9.81, 3.71, 1.62])
def test_quiet_standing_holds_weight_over_time(g, modal_deck):
    sc = make_standing(g=g, deck=modal_deck)
    s = sc.system
    out = s.simulate(sc.y_eq.copy(), T=2.0, dt=2.5e-4, out_dt=5e-3, fall_height=0.7)
    W = sc.info["weight"]
    assert not out["fell"]
    assert abs(out["fn"].sum(axis=1).mean() / W - 1) < TOL["standing_mean_force_rel"]
    assert out["n_slip"] == 0
    assert np.abs(out["com"] - out["com"][0]).max() < 1e-6       # stays put
    # the accelerometers keep reading the local gravity
    assert np.abs(np.linalg.norm(out["f"], axis=2) - g).max() < 1e-6


def test_weight_scales_with_gravity_mass_does_not(modal_deck):
    w = {}
    for g in (9.81, 3.71, 1.62):
        sc = make_standing(g=g, deck=None)
        w[g] = (sc.info["sum_fn"], sc.human.total_mass)
    assert abs(w[1.62][0] / w[9.81][0] - 1.62 / 9.81) < 1e-9
    assert abs(w[3.71][0] / w[9.81][0] - 3.71 / 9.81) < 1e-9
    assert w[1.62][1] == w[9.81][1] == w[3.71][1]
