"""Virtual inertial measurement units rigidly mounted on body segments.

Accelerometer (specific force):

    f_meas = R_ws^T (a_ws - g_w) + b_acc + n_acc

where a_ws is the inertial acceleration of the mounting point in world axes
(it contains the translational, tangential and centripetal terms because it is
the classical acceleration of that material point), R_ws is the sensor-to-world
rotation and g_w = (0, 0, -g) is the local gravity vector.  A sensor at rest
with its z axis up therefore reads (0, 0, +g).

Gyroscope:  w_meas = angular velocity of the segment in sensor axes + b + n.

Noise densities are specified in micro-g per sqrt(Hz), where "g" is the
STANDARD gravity g0 = 9.80665 m/s^2 used by sensor datasheets.  It must not be
replaced by the local planetary gravity: sensor electronics do not become
quieter on the Moon.  (The earlier study's baseline converts with its global
G_ACC = 9.81, which is harmless on Earth but would silently rescale the noise
if that constant were changed to a planetary value.)
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

G0_STANDARD = 9.80665     # m/s^2, conventional standard gravity for unit conversion


def imu_ideal(human, q, v, a, g_out=None) -> dict:
    """Noise-free IMU outputs for all mounted sensors at one state.

    Returns ``{name: dict(f=specific force (3,), w=angular rate (3,), R=R_ws)}``.
    ``g_out`` overrides the gravity used in the OUTPUT equation only (the
    dynamics are untouched).  ``g_out=0`` gives the kinematic acceleration in
    sensor axes; the difference to the true output isolates the gravity-tilt
    term of the specific force (a diagnostic, not a physical sensor).
    """
    m, d = human.model, human.data
    pin.forwardKinematics(m, d, q, v, a)
    pin.updateFramePlacements(m, d)
    gw = np.array([0.0, 0.0, -(human.g if g_out is None else g_out)])
    out = {}
    for name, fid in human.sensor_frame.items():
        R = d.oMf[fid].rotation
        acc = pin.getFrameClassicalAcceleration(m, d, fid, pin.LOCAL).linear
        w = pin.getFrameVelocity(m, d, fid, pin.LOCAL).angular
        out[name] = dict(f=acc - R.T @ gw, w=w.copy(), R=R.copy())
    return out


def white_noise(n: int, fs: float, density: float, rng) -> np.ndarray:
    """Gaussian white noise with one-sided amplitude spectral density
    ``density`` (unit / sqrt(Hz)) sampled at ``fs``: sigma = density*sqrt(fs/2)."""
    if density <= 0:
        return np.zeros(n)
    return density * np.sqrt(0.5 * fs) * rng.standard_normal(n)


def ug_to_si(density_ug: float) -> float:
    """micro-g/sqrt(Hz) -> (m/s^2)/sqrt(Hz) using standard gravity."""
    return density_ug * 1e-6 * G0_STANDARD


def apply_sensor_errors(f, w, fs, acc_noise_ug=75.0, gyro_noise_dps=0.0,
                        acc_bias=None, gyro_bias=None, seed=0):
    """Add white noise and constant bias to ideal records.

    ``f`` and ``w`` are (n, 3) arrays in sensor axes.  Bias drift, scale-factor
    error, axis misalignment, quantisation and sampling jitter are not modelled
    here (planned).
    """
    rng = np.random.default_rng(seed)
    n = f.shape[0]
    fo = f + np.column_stack([white_noise(n, fs, ug_to_si(acc_noise_ug), rng) for _ in range(3)])
    wo = w + np.column_stack([white_noise(n, fs, np.deg2rad(gyro_noise_dps), rng) for _ in range(3)])
    if acc_bias is not None:
        fo = fo + np.asarray(acc_bias, float)
    if gyro_bias is not None:
        wo = wo + np.asarray(gyro_bias, float)
    return fo, wo
