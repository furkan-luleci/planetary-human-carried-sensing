"""Compliant unilateral point contact with anchored Coulomb friction.

Normal law (Hunt-Crossley form, per contact point)

    delta   = penetration of the point below the (deformed) walking surface
    F_n     = max(0, k_n * delta^n * (1 + c_n * d(delta)/dt))      for delta > 0
    F_n     = 0                                                      for delta <= 0

The force vanishes continuously at touch-down, and the clip at zero removes
the attractive part that the damping term would otherwise produce during fast
rebound (no attraction through the floor).  Clipping can
only remove energy: while F_n is clipped the stored elastic energy
k_n delta^(n+1)/(n+1) is released without doing work.

Tangential law (stick-slip with a material anchor on the surface)

    s       = (p_xy - u_surface_xy) - anchor       elastic deformation
    f_t     = -k_t * s - c_t * d(s)/dt,   limited to  |f_t| <= mu * F_n

``anchor`` is a discrete state: it is fixed while integrating one time step,
released when the point leaves the surface and dragged along when the elastic
part of the force exceeds the friction limit.  A sticking point therefore has
no creep, unlike velocity-regularised Coulomb friction.

All parameter VALUES below are provisional assumptions for the computational
pilot (see docs/assumptions_and_sources.md); none is a measured property of a
particular foot, shoe or boot.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ContactParams:
    k_n: float = 3.6e5      # N / m^n per point (n = n_exp)
    n_exp: float = 1.5      # 1.5: Hertz-like stiffening pad; 1.0: linear pad
    c_n: float = 1.0        # s/m, Hunt-Crossley dissipation
    k_t: float = 5.0e4      # N/m per point, tangential (stick) stiffness
    c_t: float = 100.0      # N s/m per point, tangential damping
    mu: float = 0.8         # Coulomb friction coefficient

    def asdict(self):
        return asdict(self)


def contact_forces(cp: ContactParams, gap, gap_rate, s, s_rate):
    """Point forces acting ON THE FOOT in world axes.

    gap (npts,): signed vertical distance point - surface (negative in contact);
    gap_rate: its time derivative; s, s_rate (npts, 2): tangential deformation
    relative to the anchor and its rate.  Returns ``f`` (npts, 3) and the
    normal force (npts,).
    """
    delta = np.maximum(-gap, 0.0)
    dn = delta ** cp.n_exp
    fn = np.maximum(cp.k_n * dn * (1.0 - cp.c_n * gap_rate), 0.0)
    ft = -cp.k_t * s - cp.c_t * s_rate
    lim = cp.mu * fn
    mag = np.sqrt(ft[:, 0] ** 2 + ft[:, 1] ** 2)
    scale = np.where(mag > lim, lim / np.maximum(mag, 1e-300), 1.0)
    f = np.empty((gap.size, 3))
    f[:, :2] = ft * scale[:, None]
    f[:, 2] = fn
    return f, fn


def update_anchor(cp: ContactParams, anchor, rel_xy, gap, fn):
    """End-of-step anchor update (release when open, drag when slipping).

    ``rel_xy`` is the tangential position of the contact point relative to the
    surface material (p_xy - u_surface_xy).  Returns the new anchor array and
    a boolean array marking points that slipped.
    """
    new = anchor.copy()
    open_ = gap >= 0.0
    new[open_] = rel_xy[open_]
    s = rel_xy - new
    mag = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
    lim = cp.mu * fn / cp.k_t
    slip = (~open_) & (mag > lim)
    if slip.any():
        new[slip] = rel_xy[slip] - s[slip] * (lim[slip] / mag[slip])[:, None]
    return new, slip


def elastic_energy(cp: ContactParams, gap, s, fn):
    """Recoverable energy stored in the contact springs (J)."""
    delta = np.maximum(-gap, 0.0)
    en = cp.k_n * delta ** (cp.n_exp + 1.0) / (cp.n_exp + 1.0)
    mag = np.minimum(np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2), cp.mu * fn / cp.k_t)
    et = np.where(delta > 0.0, 0.5 * cp.k_t * mag ** 2, 0.0)
    return float(en.sum() + et.sum())


def static_penetration(cp: ContactParams, load_per_point: float) -> float:
    return float((load_per_point / cp.k_n) ** (1.0 / cp.n_exp))


def tangent_stiffness(cp: ContactParams, load_per_point: float) -> float:
    """d F_n / d delta at a static load (N/m per point)."""
    d = static_penetration(cp, load_per_point)
    return float(cp.n_exp * load_per_point / d) if d > 0 else 0.0
