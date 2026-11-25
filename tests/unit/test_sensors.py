import numpy as np
import pinocchio as pin
import pytest

from planetsense.human.model import build_human
from planetsense.sensor.imu import G0_STANDARD, imu_ideal, ug_to_si, white_noise
from tolerances import TOL


@pytest.mark.parametrize("g", [9.81, 3.71, 1.62])
def test_sensor_specific_force_static_orientation(g):
    """Brief 17.5 / 10.3: a sensor at rest reads the local gravity, resolved in
    its own axes: f = R^T (0, 0, +g)."""
    h = build_human(g=g)
    z = np.zeros(36)
    for pitch, roll in ((0.0, 0.0), (0.3, 0.0), (0.0, -0.2), (0.4, 0.25)):
        q = h.neutral()
        q[h.dof("root_ry")], q[h.dof("root_rx")] = pitch, roll
        out = imu_ideal(h, q, z, z)
        for name in ("pelvis_L", "pelvis_R"):
            f = out[name]["f"]
            R = pin.rpy.rpyToMatrix(roll, pitch, 0.0)            # Rz(0) Ry(pitch) Rx(roll)
            assert np.abs(f - R.T @ np.array([0, 0, g])).max() < TOL["sensor_static_abs"]
            assert abs(np.linalg.norm(f) - g) < TOL["sensor_static_abs"]
            assert np.abs(out[name]["w"]).max() == 0.0
    # pure pitch: f = (-g sin(theta), 0, g cos(theta))
    q = h.neutral()
    q[h.dof("root_ry")] = 0.3
    f = imu_ideal(h, q, z, z)["pelvis_L"]["f"]
    assert np.allclose(f, [-g * np.sin(0.3), 0, g * np.cos(0.3)], atol=TOL["sensor_static_abs"])


def test_sensor_rotational_acceleration():
    """Brief 17.5: lever-arm terms.  The whole body spins about the vertical
    root axis with rate W and angular acceleration A; every joint is still.
    A sensor at horizontal offset r from the axis must read the centripetal
    term -W^2 r, the tangential term A x r, and gravity."""
    g = 9.81
    h = build_human(g=g)
    q = h.neutral()
    W, A = 2.5, 4.0
    v = np.zeros(36)
    a = np.zeros(36)
    v[h.dof("root_rz")] = W
    a[h.dof("root_rz")] = A
    out = imu_ideal(h, q, v, a)
    pin.forwardKinematics(h.model, h.data, q)
    pin.updateFramePlacements(h.model, h.data)
    for name, fid in h.sensor_frame.items():
        r = h.data.oMf[fid].translation.copy()                   # root axis passes through the origin
        R = h.data.oMf[fid].rotation
        acc = np.array([-W * W * r[0] - A * r[1], -W * W * r[1] + A * r[0], 0.0])
        expect = R.T @ (acc + np.array([0, 0, g]))
        assert np.abs(out[name]["f"] - expect).max() <= TOL["sensor_rotating_rel"] * np.abs(expect).max()
        assert np.abs(out[name]["w"] - R.T @ np.array([0, 0, W])).max() < 1e-12


def test_noise_density_uses_standard_gravity_not_local_gravity():
    assert abs(ug_to_si(75.0) - 75e-6 * G0_STANDARD) < 1e-18
    rng = np.random.default_rng(0)
    fs, dens = 1000.0, ug_to_si(75.0)
    x = white_noise(400000, fs, dens, rng)
    # one-sided PSD of white noise = sigma^2 / (fs/2) = density^2
    assert abs(x.var() / (fs / 2) / dens ** 2 - 1) < 0.02
