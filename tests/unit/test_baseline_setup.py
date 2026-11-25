"""Test set-up of the earlier study (Section 2.5) as used by the trials: shaker
levels, the 5 N vertical floor, and the deck-alone lead-in."""
import json
import os

import numpy as np
import pytest

from planetsense.excitation.multisine import make_baseline_shakers, shaker_layout
from planetsense.structure.spine_beam import SpineBeamDeck

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_shaker_levels_and_floor():
    F = make_baseline_shakers(T=40.0, rms=800.0, floor_rms=5.0, seed=3).sample(1e-3, 40000, 0.0)
    rms = F.std(axis=1)
    horiz = np.r_[0:9:3, 1:9:3]
    np.testing.assert_allclose(rms[horiz], 800.0, rtol=1e-9)
    np.testing.assert_allclose(rms[2::3], np.hypot(800.0, 5.0), rtol=0.01)     # independent floor adds in power
    G = make_baseline_shakers(T=40.0, rms=0.0, floor_rms=5.0, seed=3).sample(1e-3, 40000, 0.0)   # static-load case
    assert np.abs(G[horiz]).max() == 0.0
    np.testing.assert_allclose(G[2::3].std(axis=1), 5.0, rtol=1e-9)
    # same seed, same forces (matched excitation across planets); continuous in time
    H = make_baseline_shakers(T=40.0, rms=800.0, floor_rms=5.0, seed=3).sample(1e-3, 40000, 0.0)
    assert np.array_equal(F, H)


def test_shaker_layout_is_the_baseline_diagonal():
    xy = shaker_layout(14.0, 3.0)
    np.testing.assert_allclose(xy, [[3.5, -1.35], [7.0, 0.0], [10.5, 1.35]])


def test_deck_lead_in_continues_seamlessly_into_the_coupled_simulation():
    """The deck state handed over by the lead-in equals a longer lead-in, when the
    coupled simulator runs on with the walker still off the deck."""
    from planetsense.human.body_v2 import load_leg
    from planetsense.simulation.records import DT, lead_in_seconds
    from planetsense.simulation.walking import WalkParams, make_walker
    deck = SpineBeamDeck().modal(f_max=150.0)
    assert lead_in_seconds(deck) == pytest.approx(20.0)
    pj = json.load(open(os.path.join(ROOT, "outputs", "gait", "earth_1.00.json")))
    forcing = make_baseline_shakers(T=30.0, rms=800.0, floor_rms=5.0, seed=5)
    sys_, y = make_walker(9.81, WalkParams(**pj["params"]), load_leg(), deck=deck, x0=-5.0, v0=0.93, y_walk=0.575,
                          shaker_xy=shaker_layout(14.0, 3.0), g_gain=pj.get("g_gain"))
    e0, ed0 = sys_.deck_lead_in(forcing, 5.0, DT)
    ie = 2 * sys_.nq
    y[ie:ie + sys_.nm], y[ie + sys_.nm:ie + 2 * sys_.nm] = e0, ed0
    out = sys_.simulate(y, T=0.5, dt=DT, out_dt=1e-3, forcing=forcing)

    class Shift:
        def sample(self, h, n, t0=0.0):
            return forcing.sample(h, n, t0 + 0.5)
    e1, _ = sys_.deck_lead_in(Shift(), 5.5, DT)
    assert np.abs(out["y_end"][ie:ie + sys_.nm] - e1).max() <= 1e-9 * np.abs(e1).max()
