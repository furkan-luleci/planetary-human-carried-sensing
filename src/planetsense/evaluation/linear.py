"""Linear analysis of the coupled standing system about its equilibrium.

For small vibration with all contact points closed and sticking, the coupled
system is smooth and can be linearised:  x' = A x + B F,  y = C x + D F.
The linear model gives exact small-amplitude frequency responses and modal
properties, which is what "transmission" (brief outcome O1) means for a
standing person.  It does NOT describe contact loss, slip or large sway; those
need the time-domain model.
"""
from __future__ import annotations

import numpy as np

from ..structure.spine_beam import DeckParams, ModalDeck


def frf(A, B, C, D, freqs):
    """H(f) = C (i w I - A)^-1 B + D, shape (nf, ny, nu)."""
    n = A.shape[0]
    I = np.eye(n)
    H = np.empty((len(freqs), C.shape[0], B.shape[1]), complex)
    for k, f in enumerate(freqs):
        H[k] = C @ np.linalg.solve(2j * np.pi * f * I - A, B) + D
    return H


def mode_table(A, names, top=3):
    """Oscillatory eigenmodes: frequency |lambda|/2pi, damping ratio and the
    coordinates that dominate each eigenvector (among the first len(names))."""
    lam, V = np.linalg.eig(A)
    out = []
    for i in np.argsort(np.abs(lam)):
        if lam[i].imag <= 1e-6:
            continue
        v = np.abs(V[:len(names), i])
        idx = np.argsort(-v)[:top]
        out.append(dict(f=float(abs(lam[i]) / (2 * np.pi)),
                        zeta=float(-lam[i].real / abs(lam[i])),
                        dominant=[(names[j], float(v[j] / v.max())) for j in idx]))
    return out, lam


def heavy_platform(mass: float = 1.0e9, f0: float = 0.05, L: float = 14.0) -> ModalDeck:
    """A rigid platform that can translate in x, y, z, so heavy that the person
    cannot move it.  Driving it with a force gives pure base excitation of the
    human, i.e. the body's own transmissibility without any structural
    dynamics.  Represented as a 'deck' with three uniform translation modes."""
    nodal = np.zeros((2, 6, 3))
    for a in range(3):
        nodal[:, a, a] = 1.0 / np.sqrt(mass)
    w = np.full(3, 2 * np.pi * f0)
    return ModalDeck(params=DeckParams(L=L), omega=w, zeta=np.zeros(3),
                     labels=["PX", "PY", "PZ"], nodal=nodal,
                     x_nodes=np.array([0.0, L]), ztop=0.0, interp="linear",
                     meta=dict(kind="heavy_platform", mass=mass, f0=f0))


def base_transmissibility(scenario, freqs, g_out=None):
    """Transmissibility from platform acceleration (x, y, z) to every sensor
    channel for a scenario built on ``heavy_platform``.

    Returns T with shape (nf, n_sensor, 6, 3): sensor channels
    (f_x, f_y, f_z, w_x, w_y, w_z) per unit base acceleration in x, y, z.
    Specific-force rows are dimensionless; gyro rows are (rad/s)/(m/s^2).
    """
    s, y = scenario.system, scenario.y_eq
    A, BF, _ = s.linearize(y)
    C, D, _, names = s.sensor_jacobians(y, g_out=g_out)
    nq, nm = s.nq, s.nm
    rows = slice(2 * nq + nm, 2 * nq + 2 * nm)
    S = s.deck.surface_modes([y[0]], [y[1]])[0]           # (3, nm), uniform
    Ca, Da = S @ A[rows], S @ BF[rows]
    # one force component per platform direction: the first shaker point
    BF1, Da1, D1 = BF[:, :3], Da[:, :3], D[:, :3]
    H = frf(A, BF1, np.vstack([C, Ca]), np.vstack([D1, Da1]), freqs)
    ns = len(names)
    T = np.empty((len(freqs), ns, 6, 3), complex)
    for a in range(3):
        base = H[:, 6 * ns + a, a]
        T[:, :, :, a] = (H[:, :6 * ns, a] / base[:, None]).reshape(len(freqs), ns, 6)
    return T, names, A


def com_participation(human, q_eq, A):
    """Fraction of each oscillatory mode's kinetic energy carried by rigid
    translation of the whole-body centre of mass along x, y, z:

        r_axis = m |J_com,axis phi|^2 / (phi^H M phi)

    Used to name modes without relying on coordinate units: the "bounce" mode
    of the body on its foot pads is the one with the largest r_z, the sagittal
    sway mode the slow one with large r_x.  Returns a list of dicts sorted by
    frequency.
    """
    import pinocchio as pin
    m, d = human.model, human.data
    M = pin.crba(m, d, q_eq)
    M = np.triu(M) + np.triu(M, 1).T
    J = pin.jacobianCenterOfMass(m, d, q_eq)
    lam, V = np.linalg.eig(A)
    mt = human.total_mass
    out = []
    for i in np.argsort(np.abs(lam)):
        if lam[i].imag <= 1e-6:
            continue
        phi = V[:m.nv, i]
        T = float(np.real(np.conj(phi) @ M @ phi))
        if T <= 0:
            continue
        r = mt * np.abs(J @ phi) ** 2 / T
        out.append(dict(f=float(abs(lam[i]) / (2 * np.pi)), zeta=float(-lam[i].real / abs(lam[i])),
                        rx=float(r[0]), ry=float(r[1]), rz=float(r[2])))
    return out


def apparent_mass(scenario, freqs):
    """Vertical driving-point apparent mass of the standing body (kg, complex):
    total vertical foot force per unit vertical platform acceleration.  The
    scenario must stand on ``heavy_platform``.  Tends to the body mass at low
    frequency."""
    s, y = scenario.system, scenario.y_eq
    A, BF, _ = s.linearize(y)
    CF = s.force_jacobian(y)
    nq, nm = s.nq, s.nm
    rows = slice(2 * nq + nm, 2 * nq + 2 * nm)
    h = scenario.human
    S = s.deck.surface_modes([y[h.dof("root_x")]], [y[h.dof("root_y")]])[0]
    Ca, Da = S @ A[rows], S @ BF[rows]
    H = frf(A, BF[:, 2:3], np.vstack([CF[2:3], Ca[2:3]]),
            np.vstack([np.zeros((1, 1)), Da[2:3, 2:3]]), freqs)
    return H[:, 0, 0] / H[:, 1, 0]


def fit_sdof(freqs, m_norm):
    """Fit the base-excited single-degree-of-freedom apparent mass
    (1 + 2 i z r) / (1 - r^2 + 2 i z r),  r = f / f_n,  to a normalised complex
    apparent mass.  Returns (f_n, zeta)."""
    from scipy.optimize import least_squares
    freqs = np.asarray(freqs, float)

    def res(p):
        r = freqs / p[0]
        m = (1 + 2j * p[1] * r) / (1 - r * r + 2j * p[1] * r)
        d = m - m_norm
        return np.concatenate([d.real, d.imag])

    k = int(np.argmax(np.abs(m_norm)))
    sol = least_squares(res, [freqs[k], 0.3], bounds=([0.5, 0.01], [40.0, 3.0]))
    return float(sol.x[0]), float(sol.x[1])
