"""Compiled inner loop of the coupled right-hand side.

This is a line-by-line transcription of the NumPy reference path in
``CoupledSystem.rhs`` (foot-point kinematics, deformed-surface gap, contact
law, anchor update, equal-and-opposite deck loading, modal deck equation and
power terms).  It exists only for speed; ``tests/unit/test_kernel_equivalence``
checks that both paths return the same derivatives to round-off.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def points_world(R, p, cloc, P):
    nf, npf = cloc.shape[0], cloc.shape[1]
    for i in range(nf):
        for k in range(npf):
            ip = i * npf + k
            for a in range(3):
                P[ip, a] = (R[i, a, 0] * cloc[i, k, 0] + R[i, a, 1] * cloc[i, k, 1]
                            + R[i, a, 2] * cloc[i, k, 2] + p[i, a])


@njit(cache=True)
def contact_deck_kernel(R, p, vl, w, cloc, S2, eta, etad, anchor, par, k_modal,
                        c_modal, Qe, do_anchor, enabled,
                        P, relp, relv, f, fn, flin, fang, detadd):
    """par = (k_n, n_exp, c_n, k_t, c_t, mu).  Returns the three power terms
    (contact, deck damping, external) and the number of slipping points."""
    k_n, n_exp, c_n, k_t, c_t, mu = par[0], par[1], par[2], par[3], par[4], par[5]
    nf, npf = cloc.shape[0], cloc.shape[1]
    nm = eta.shape[0]
    p_contact = 0.0
    n_slip = 0
    for i in range(nf):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        mx = 0.0
        my = 0.0
        mz = 0.0
        if enabled:
            for k in range(npf):
                ip = i * npf + k
                lx, ly, lz = cloc[i, k, 0], cloc[i, k, 1], cloc[i, k, 2]
                # point velocity in the foot frame: v + w x loc
                cx = w[i, 1] * lz - w[i, 2] * ly + vl[i, 0]
                cy = w[i, 2] * lx - w[i, 0] * lz + vl[i, 1]
                cz = w[i, 0] * ly - w[i, 1] * lx + vl[i, 2]
                for a in range(3):
                    P[ip, a] = R[i, a, 0] * lx + R[i, a, 1] * ly + R[i, a, 2] * lz + p[i, a]
                    Va = R[i, a, 0] * cx + R[i, a, 1] * cy + R[i, a, 2] * cz
                    us = 0.0
                    uds = 0.0
                    for mm in range(nm):
                        us += S2[3 * ip + a, mm] * eta[mm]
                        uds += S2[3 * ip + a, mm] * etad[mm]
                    relp[ip, a] = P[ip, a] - us
                    relv[ip, a] = Va - uds
                gap = relp[ip, 2]
                gr = relv[ip, 2]
                fnn = 0.0
                if gap < 0.0:
                    fnn = k_n * (-gap) ** n_exp * (1.0 - c_n * gr)
                    if fnn < 0.0:
                        fnn = 0.0
                if do_anchor:
                    if gap >= 0.0:
                        anchor[ip, 0] = relp[ip, 0]
                        anchor[ip, 1] = relp[ip, 1]
                    else:
                        sx = relp[ip, 0] - anchor[ip, 0]
                        sy = relp[ip, 1] - anchor[ip, 1]
                        mag = np.sqrt(sx * sx + sy * sy)
                        lim = mu * fnn / k_t
                        if mag > lim:
                            anchor[ip, 0] = relp[ip, 0] - sx * lim / mag
                            anchor[ip, 1] = relp[ip, 1] - sy * lim / mag
                            n_slip += 1
                sx = relp[ip, 0] - anchor[ip, 0]
                sy = relp[ip, 1] - anchor[ip, 1]
                ftx = -k_t * sx - c_t * relv[ip, 0]
                fty = -k_t * sy - c_t * relv[ip, 1]
                mag = np.sqrt(ftx * ftx + fty * fty)
                lim = mu * fnn
                if mag > lim:
                    sc = lim / mag
                    ftx *= sc
                    fty *= sc
                f[ip, 0] = ftx
                f[ip, 1] = fty
                f[ip, 2] = fnn
                fn[ip] = fnn
                p_contact += ftx * relv[ip, 0] + fty * relv[ip, 1] + fnn * relv[ip, 2]
                # force in the foot frame (R^T f) and its moment about the joint
                l0 = R[i, 0, 0] * ftx + R[i, 1, 0] * fty + R[i, 2, 0] * fnn
                l1 = R[i, 0, 1] * ftx + R[i, 1, 1] * fty + R[i, 2, 1] * fnn
                l2 = R[i, 0, 2] * ftx + R[i, 1, 2] * fty + R[i, 2, 2] * fnn
                fx += l0
                fy += l1
                fz += l2
                mx += ly * l2 - lz * l1
                my += lz * l0 - lx * l2
                mz += lx * l1 - ly * l0
        flin[i, 0] = fx
        flin[i, 1] = fy
        flin[i, 2] = fz
        fang[i, 0] = mx
        fang[i, 1] = my
        fang[i, 2] = mz
    p_damp = 0.0
    p_ext = 0.0
    npts = nf * npf
    for mm in range(nm):
        Q = 0.0
        if enabled:
            for ip in range(npts):
                Q -= (S2[3 * ip, mm] * f[ip, 0] + S2[3 * ip + 1, mm] * f[ip, 1]
                      + S2[3 * ip + 2, mm] * f[ip, 2])
        detadd[mm] = Q + Qe[mm] - c_modal[mm] * etad[mm] - k_modal[mm] * eta[mm]
        p_damp += c_modal[mm] * etad[mm] * etad[mm]
        p_ext += Qe[mm] * etad[mm]
    return p_contact, p_damp, p_ext, n_slip
