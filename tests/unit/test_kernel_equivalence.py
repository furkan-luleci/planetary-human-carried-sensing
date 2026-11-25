import numpy as np
import pytest

from planetsense.simulation.standing import make_standing
from tolerances import TOL


@pytest.mark.parametrize("with_deck", [False, True])
def test_compiled_path_matches_numpy_reference(with_deck, modal_deck, rng):
    sc = make_standing(g=9.81, deck=modal_deck if with_deck else None)
    s, y = sc.system, sc.y_eq
    F = rng.normal(0, 100, 9) if with_deck else None
    scale = np.concatenate([np.full(36, 3e-3), np.full(36, 5e-2), np.full(s.nm, 1e-4),
                            np.full(s.nm, 1e-2), np.full(s.nc, 1e-3)])
    worst = 0.0
    for trial in range(60):
        yp = y.copy()
        yp[:s.nx] += rng.normal(0, 1, s.nx) * scale
        if trial % 3 == 0:
            yp[2] += 0.006                              # some contact points open
        first = bool(trial % 2)
        a0 = s.anchor.copy()
        s.use_kernel = False
        d_ref = s.rhs(0.0, yp, F, first=first)
        a_ref, s.anchor = s.anchor.copy(), a0.copy()
        s.use_kernel = True
        d_ker = s.rhs(0.0, yp, F, first=first)
        a_ker, s.anchor = s.anchor.copy(), a0.copy()
        worst = max(worst, np.abs(d_ref - d_ker).max() / np.abs(d_ref).max(),
                    np.abs(a_ref - a_ker).max())
    assert worst < TOL["kernel_rel"]
