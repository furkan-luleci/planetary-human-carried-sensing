import numpy as np
import pinocchio as pin
import pytest

from planetsense.human.model import build_human
from tolerances import TOL


def test_segment_masses_and_inertias():
    h = build_human(body_mass=75.0)
    assert abs(sum(s.mass for s in h.segments.values()) - 75.0) < 1e-10
    assert abs(h.total_mass - 75.0) < 1e-10
    for s in h.segments.values():
        assert np.allclose(s.inertia, s.inertia.T)
        assert np.linalg.eigvalsh(s.inertia).min() > 0
        w = np.sort(np.linalg.eigvalsh(s.inertia))
        assert w[0] + w[1] >= w[2] * (1 - 1e-12)      # triangle inequality of a physical inertia
    c = h.com(h.neutral())
    assert abs(c[1]) < 1e-12                           # left-right symmetry


def test_inertial_mass_invariant_to_gravity(rng):
    """Brief 17.5: weight changes with g, inertial mass and inertia do not."""
    models = [build_human(g=g) for g in (9.81, 3.71, 1.62)]
    for _ in range(10):
        q, v = rng.uniform(-0.6, 0.6, 36), rng.uniform(-2, 2, 36)
        M0 = pin.crba(models[0].model, models[0].data, q).copy()
        T0 = models[0].kinetic_energy(q, v)
        for h in models[1:]:
            M = pin.crba(h.model, h.data, q)
            assert np.abs(M - M0).max() <= TOL["mass_invariance_rel"] * np.abs(M0).max()
            assert abs(h.kinetic_energy(q, v) - T0) <= 1e-12 * abs(T0)
            assert h.total_mass == models[0].total_mass
    # potential energy and static joint loads scale exactly with g
    q = rng.uniform(-0.4, 0.4, 36)
    g0 = pin.computeGeneralizedGravity(models[0].model, models[0].data, q).copy()
    for h in models[1:]:
        gg = pin.computeGeneralizedGravity(h.model, h.data, q)
        assert np.allclose(gg, g0 * h.g / models[0].g, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("g", [9.81, 3.71, 1.62])
def test_free_fall_acceleration(g, rng):
    h = build_human(g=g)
    q = rng.uniform(-0.3, 0.3, 36)
    q[3:6] = 0.0                                       # world-aligned root so that a[2] is vertical
    a = pin.aba(h.model, h.data, q, np.zeros(36), np.zeros(36))
    assert abs(a[2] + g) < TOL["free_fall_accel_abs"]
    a[2] = 0.0
    assert np.abs(a).max() < TOL["free_fall_accel_abs"]
