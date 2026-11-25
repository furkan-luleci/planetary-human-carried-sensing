import os

import numpy as np
import pytest

from tolerances import TOL

NPZ = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "phase_0_baseline",
                   "out", "baseline_deck_MCK.npz")


@pytest.mark.skipif(not os.path.exists(NPZ), reason="baseline OpenSees matrices not exported "
                    "(run experiments/phase_0_baseline/reproduce_baseline.py)")
def test_deck_matrices_equal_baseline_opensees(fe_deck):
    """The NumPy deck must be the same structure as the earlier study's OpenSeesPy
    deck: mass, stiffness and Rayleigh damping matrices agree entry by entry."""
    z = np.load(NPZ)
    ib, im = [], []
    for n, row in zip(z["dof_nodes"], z["dof_map"]):
        for k, e in enumerate(row):
            if e >= 0:
                ib.append(int(e))
                im.append(fe_deck.dof_index(int(n) - 1, k))
    ib, im = np.array(ib), np.array(im)
    assert (im >= 0).all() and ib.size == fe_deck.free.size == z["M"].shape[0]
    for mine, theirs, tol in ((fe_deck.M, z["M"], TOL["deck_matrix_rel"]),
                              (fe_deck.K, z["K"], TOL["deck_matrix_rel"]),
                              (fe_deck.C, z["C"], 1e-8)):
        a, b = mine[np.ix_(im, im)], theirs[np.ix_(ib, ib)]
        assert np.abs(a - b).max() / np.abs(b).max() < tol
