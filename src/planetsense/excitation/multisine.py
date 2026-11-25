"""Band-limited random-phase multisine excitation.

The earlier study's baseline draws white noise at the integration rate and low-pass
filters it, so a force record is tied to one time step.  Here an excitation is
a finite sum of sinusoids

    F_c(t) = sum_k A_k cos(2 pi k t / T + phi_{c,k})

defined in CONTINUOUS time by (T, A_k, phi_{c,k}).  The same physical input can
therefore be sampled on any time grid, which is what a time-step convergence
study and the "identical structural input across gravities" comparison
(brief E1) both need.  With many lines the signal is close to Gaussian.

The amplitude profile is flat up to ``f_flat`` and then follows the magnitude
of a Butterworth low-pass of order ``order`` (default: 75 Hz, 4th order, the
earlier study's shaker filter) up to ``f_max``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Multisine:
    T: float                 # period (s)
    k: np.ndarray            # (K,) harmonic indices, f_k = k / T
    amp: np.ndarray          # (n_chan, K) amplitudes
    phase: np.ndarray        # (n_chan, K) phases (rad)
    seed: int

    @property
    def n_chan(self) -> int:
        return self.amp.shape[0]

    @property
    def freq(self) -> np.ndarray:
        return self.k / self.T

    def rms(self) -> np.ndarray:
        return np.sqrt(0.5 * np.sum(self.amp ** 2, axis=1))

    def sample(self, h: float, n: int, t0: float = 0.0) -> np.ndarray:
        """Samples F(t0 + j h), j = 0..n-1, shape (n_chan, n).

        Uses an inverse FFT when ``T / h`` is an integer (exact), otherwise a
        direct sum.
        """
        Np = self.T / h
        if abs(Np - round(Np)) < 1e-9 * Np and round(Np) >= 2 * self.k.max() + 2:
            Np = int(round(Np))
            X = np.zeros((self.n_chan, Np // 2 + 1), complex)
            ph = self.phase + 2 * np.pi * self.k * t0 / self.T
            X[:, self.k] = 0.5 * self.amp * np.exp(1j * ph) * Np
            x = np.fft.irfft(X, n=Np, axis=1)
            reps = int(np.ceil(n / Np))
            return np.tile(x, reps)[:, :n]
        t = t0 + h * np.arange(n)
        out = np.zeros((self.n_chan, n))
        for c in range(self.n_chan):
            for j0 in range(0, n, 8192):
                tt = t[j0:j0 + 8192]
                out[c, j0:j0 + 8192] = (self.amp[c][None, :] * np.cos(
                    2 * np.pi * np.outer(tt, self.k) / self.T + self.phase[c][None, :])).sum(axis=1)
        return out


def make_multisine(n_chan: int, T: float, rms, seed: int, f_min: float = 0.2,
                   f_flat: float = 75.0, f_max: float = 150.0, order: int = 4) -> Multisine:
    """Random-phase multisine with per-channel RMS ``rms`` (scalar or (n_chan,))."""
    k = np.arange(max(1, int(np.ceil(f_min * T))), int(np.floor(f_max * T)) + 1)
    f = k / T
    prof = 1.0 / np.sqrt(1.0 + (f / f_flat) ** (2 * order))
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2 * np.pi, size=(n_chan, k.size))
    rms = np.broadcast_to(np.asarray(rms, float), (n_chan,))
    amp = prof[None, :] * (rms / np.sqrt(0.5 * np.sum(prof ** 2)))[:, None]
    return Multisine(T=float(T), k=k, amp=amp, phase=phase, seed=int(seed))


def shaker_layout(L: float, W: float, edge_frac: float = 0.90):
    """Three shaker points on the walking surface, as in the earlier study, Fig. 4(a):
    L/4 and 3L/4 at -/+ 90 % of the half-width, L/2 on the centreline."""
    hw = 0.5 * W * edge_frac
    return np.array([[0.25 * L, -hw], [0.50 * L, 0.0], [0.75 * L, +hw]])


class ShakerForcing:
    """The three shakers of the the earlier study ambient test (Section 2.5).

    ``main``: 9 channels (x, y, z per shaker point), each of RMS ``rms``
    (the earlier study's per-component standard deviation, 800 N at baseline).
    ``floor``: 3 channels, one vertical broadband floor per point (5 N), drawn
    independently and added to the vertical channels.  Either part may be
    absent (``rms`` = 0 gives the earlier study's static-load case, in which the
    floor is the only dynamic shaker force).  Both parts follow the
    earlier study's shaker spectrum (flat to 75 Hz, 4th-order Butterworth
    roll-off) but are random-phase multisines, so the force is defined in
    continuous time and can be sampled at the integrator's half steps.
    """

    def __init__(self, main: Multisine | None, floor: Multisine | None, n_points: int = 3):
        self.main, self.floor, self.n_points = main, floor, int(n_points)

    @property
    def n_chan(self) -> int:
        return 3 * self.n_points

    def sample(self, h: float, n: int, t0: float = 0.0) -> np.ndarray:
        out = np.zeros((self.n_chan, n))
        if self.main is not None:
            out += self.main.sample(h, n, t0)
        if self.floor is not None:
            out[2::3] += self.floor.sample(h, n, t0)
        return out


def make_baseline_shakers(T: float, rms: float, floor_rms: float, seed: int, n_points: int = 3) -> ShakerForcing | None:
    """Shaker forcing of one record: period ``T`` (longer than lead-in plus
    record, so nothing repeats), main shakers of RMS ``rms`` per component and
    a vertical floor of RMS ``floor_rms`` per point.  The floor has its own
    random phases (seed + 50 000)."""
    T = float(np.ceil(T))          # whole seconds: Multisine.sample then uses its exact inverse-FFT path
    main = make_multisine(3 * n_points, T=T, rms=rms, seed=seed) if rms > 0 else None
    floor = make_multisine(n_points, T=T, rms=floor_rms, seed=seed + 50000) if floor_rms > 0 else None
    if main is None and floor is None:
        return None
    return ShakerForcing(main, floor, n_points)
