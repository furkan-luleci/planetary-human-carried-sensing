import numpy as np
import pinocchio as pin
import pytest

from planetsense.human.model import build_human, to_mjcf
from planetsense.sensor.imu import imu_ideal
from tolerances import TOL

mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("g", [9.81, 1.62])
def test_multibody_dynamics_and_sensors_agree_with_mujoco(g, rng):
    """Independent engine cross-check (brief 15.2, "cross-solver comparison"):
    forward dynamics, mass matrix, accelerometer specific force and gyroscope
    output of the same 36-coordinate model from Pinocchio and from MuJoCo."""
    h = build_human(g=g)
    m, d = h.model, h.data
    mj = mujoco.MjModel.from_xml_string(to_mjcf(h))
    md = mujoco.MjData(mj)
    assert abs(mj.body_mass.sum() - h.total_mass) < 1e-9
    worst = 0.0
    for _ in range(30):
        q, v, tau = rng.uniform(-0.6, 0.6, 36), rng.uniform(-3, 3, 36), rng.uniform(-50, 50, 36)
        a = pin.aba(m, d, q, v, tau)
        md.qpos[:], md.qvel[:], md.qfrc_applied[:] = q, v, tau
        mujoco.mj_forward(mj, md)
        worst = max(worst, np.abs(a - md.qacc).max() / np.abs(a).max())
        Mp = pin.crba(m, d, q)
        Mp = np.triu(Mp) + np.triu(Mp, 1).T
        Mm = np.zeros((36, 36))
        mujoco.mj_fullM(mj, md, Mm)
        worst = max(worst, np.abs(Mp - Mm).max() / np.abs(Mp).max())
        imu = imu_ideal(h, q, v, a)
        for name in h.sensor_frame:
            sa, sg = md.sensor("acc_" + name).data, md.sensor("gyr_" + name).data
            worst = max(worst, np.abs(imu[name]["f"] - sa).max() / np.abs(sa).max(),
                        np.abs(imu[name]["w"] - sg).max() / np.abs(sg).max())
    assert worst < TOL["cross_solver_rel"]
