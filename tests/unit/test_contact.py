import numpy as np

from planetsense.contact.compliant import (ContactParams, contact_forces, elastic_energy,
                                     update_anchor)


def test_contact_unilateral_no_attraction(rng):
    """Brief 17.5: the normal reaction is never attractive and vanishes when open."""
    cp = ContactParams()
    n = 20000
    gap = rng.uniform(-0.01, 0.01, n)
    rate = rng.uniform(-5.0, 5.0, n)                   # includes fast rebound
    s = rng.uniform(-0.01, 0.01, (n, 2))
    sr = rng.uniform(-1, 1, (n, 2))
    f, fn = contact_forces(cp, gap, rate, s, sr)
    assert (fn >= 0).all()
    assert (fn[gap >= 0] == 0).all()
    assert (np.abs(f[gap >= 0]).max(axis=1) == 0).all()          # no friction without normal force
    assert (np.hypot(f[:, 0], f[:, 1]) <= cp.mu * fn * (1 + 1e-12)).all()   # Coulomb cone
    # the clip is active exactly where the damping term would pull the foot down
    clipped = (gap < 0) & (1.0 - cp.c_n * rate < 0)
    assert clipped.any() and (fn[clipped] == 0).all()


def test_normal_law_is_continuous_at_touchdown():
    cp = ContactParams()
    gap = np.array([1e-12, 0.0, -1e-12])
    f, fn = contact_forces(cp, gap, np.full(3, -1.0), np.zeros((3, 2)), np.zeros((3, 2)))
    assert fn[0] == 0 and fn[1] == 0 and 0 <= fn[2] < 1e-9


def test_anchor_release_and_slip():
    cp = ContactParams()
    anchor = np.zeros((3, 2))
    rel = np.array([[0.01, 0.0], [1e-5, 0.0], [0.02, 0.0]])
    gap = np.array([0.001, -0.004, -0.004])
    fn = np.array([0.0, 90.0, 90.0])
    new, slip = update_anchor(cp, anchor, rel, gap, fn)
    assert np.allclose(new[0], rel[0]) and not slip[0]          # open: released
    assert np.allclose(new[1], 0.0) and not slip[1]             # sticking
    lim = cp.mu * 90.0 / cp.k_t
    assert slip[2] and abs(np.linalg.norm(rel[2] - new[2]) - lim) < 1e-15
    e = elastic_energy(cp, gap, rel - new, fn)
    assert e > 0
