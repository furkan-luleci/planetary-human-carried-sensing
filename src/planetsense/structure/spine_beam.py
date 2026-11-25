"""S1 benchmark deck: simply supported 3D Euler-Bernoulli spine beam.

This is an independent NumPy re-implementation of the OpenSeesPy spine-beam
deck of the earlier study (Section 2.3).  It exists so that the structure
can be embedded in one monolithic human-structure integrator, and so that the
structural matrices can be checked against the baseline's OpenSees matrices
(``tests/regression/test_deck_vs_baseline.py``).

Conventions (SI units throughout)
---------------------------------
* World / deck axes: x along the span, y across the deck (walker's left
  positive), z vertical up.  Right-handed.
* Nodal DOF order: (u_x, u_y, u_z, psi_x, psi_y, psi_z); rotations are
  right-hand rotations about the global axes.
* Euler-Bernoulli kinematics: psi_y = -d u_z/dx,  psi_z = +d u_y/dx.
* Supports (as in the earlier study): node 0 fixes u_x, u_y, u_z, psi_x;
  the last node fixes u_y, u_z, psi_x ("fork" supports, axially released).
* Mass: rho*A lumped at the nodes for the three translations, rho*I_p lumped
  for torsion, no mass on the bending rotations (earlier study, Section 2.3).
* The structure is linear about its own gravity-loaded equilibrium.  Gravity
  does not appear anywhere in this module: with the axial release there is no
  first-order gravity prestress in the beam, so the unoccupied
  stiffness-to-mass eigenproblem is independent of g.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")


@dataclass(frozen=True)
class DeckParams:
    """Nominal S1 deck of the earlier study (Table 1 baseline column)."""
    L: float = 14.0            # span (m)
    W: float = 3.0             # plan width of the walking surface (m)
    t: float = 0.20            # nominal slab thickness (m)
    E: float = 30.0e9          # elastic modulus (Pa)
    rho: float = 2400.0        # density (kg/m^3)
    G_over_E: float = 1.0 / 2.6   # shear modulus ratio (nu = 0.3)
    # Vertical second moment of area prescribed independently of W*t^3/12, as
    # an equivalent-section property (earlier study, Section 2.3).  ``None`` uses
    # the solid rectangle.  The value follows geometry changes as in the
    # baseline code: I = I_vert * (W / W_ref) * (t / t_ref)^3.
    I_vert: float | None = 0.0024274616740991993
    I_ref_t: float = 0.20
    I_ref_W: float = 3.00
    zeta: float = 0.02         # Rayleigh ratio assigned to the first two modes
    n_elem: int = 40


def section_properties(p: DeckParams) -> dict:
    """Equivalent-rectangle section properties (earlier study, Fig. 4c)."""
    A0, I0 = p.W * p.t, p.W * p.t ** 3 / 12.0
    if p.I_vert is None:
        I = I0
    else:
        I = p.I_vert * (p.W / p.I_ref_W) * (p.t / p.I_ref_t) ** 3
    beta = np.sqrt(I / I0)           # A is preserved, depth scaled by beta
    b_eq, h_eq = p.W / beta, p.t * beta
    a, b = max(b_eq, h_eq), min(b_eq, h_eq)
    J = a * b ** 3 * (1.0 / 3.0 - 0.21 * (b / a) * (1.0 - b ** 4 / (12.0 * a ** 4)))
    return dict(A=b_eq * h_eq, I=b_eq * h_eq ** 3 / 12.0, J=J,
                Ip=b_eq * h_eq * (b_eq ** 2 + h_eq ** 2) / 12.0,
                Ilat=h_eq * b_eq ** 3 / 12.0, b_eq=b_eq, h_eq=h_eq,
                ztop=0.5 * h_eq, G=p.E * p.G_over_E)


def closed_form_frequency(p: DeckParams, family: str, n: int) -> float:
    """Continuous-beam natural frequency (Hz) of mode ``n`` of a family.

    B: vertical bending, L: lateral bending, T: torsion (fork supports),
    A: axial (fixed-free).
    """
    s = section_properties(p)
    L = p.L
    if family == "B":
        return n * n * np.pi / (2 * L * L) * np.sqrt(p.E * s["I"] / (p.rho * s["A"]))
    if family == "L":
        return n * n * np.pi / (2 * L * L) * np.sqrt(p.E * s["Ilat"] / (p.rho * s["A"]))
    if family == "T":
        return n / (2 * L) * np.sqrt(s["G"] * s["J"] / (p.rho * s["Ip"]))
    if family == "A":
        return (2 * n - 1) / (4 * L) * np.sqrt(p.E / p.rho)
    raise ValueError(family)


def _hermite(xi):
    """Cubic Hermite functions and their xi-derivatives (w1, w1', w2, w2')."""
    xi = np.asarray(xi, float)
    H = np.stack([1 - 3 * xi ** 2 + 2 * xi ** 3, xi - 2 * xi ** 2 + xi ** 3,
                  3 * xi ** 2 - 2 * xi ** 3, -xi ** 2 + xi ** 3])
    dH = np.stack([-6 * xi + 6 * xi ** 2, 1 - 4 * xi + 3 * xi ** 2,
                   6 * xi - 6 * xi ** 2, -2 * xi + 3 * xi ** 2])
    return H, dH


@dataclass
class ModalDeck:
    """Modal model handed to the coupled integrator.

    eta'' + 2 zeta_i omega_i eta' + omega_i^2 eta = Phi^T f
    with mass-normalised modes, so modal mass is 1 for every mode.
    """
    params: DeckParams
    omega: np.ndarray            # (nm,) rad/s
    zeta: np.ndarray             # (nm,) modal damping ratios (Rayleigh)
    labels: list
    nodal: np.ndarray            # (n_nodes, 6, nm) mode shapes incl. fixed DOFs
    x_nodes: np.ndarray
    ztop: float
    interp: str = "hermite"
    meta: dict = field(default_factory=dict)

    @property
    def nm(self) -> int:
        return self.omega.size

    @property
    def freq_hz(self) -> np.ndarray:
        return self.omega / (2 * np.pi)

    def on_deck(self, x):
        x = np.asarray(x, float)
        return (x >= 0.0) & (x <= self.params.L)

    def surface_modes(self, x, y) -> np.ndarray:
        """Modal surface-displacement matrices at walking-surface points.

        Returns ``S`` with shape (npts, 3, nm): displacement of the surface
        point (x, y) in world axes per unit modal coordinate, using the
        plane-section kinematics of the earlier study's Eq. (2):
            u_x = u_x + ztop*psi_y - y*psi_z
            u_y = u_y - ztop*psi_x
            u_z = u_z + y*psi_x
        The same matrix maps a point force to modal forces (S^T f), which makes
        loading and displacement recovery work-conjugate.  Points beyond the
        span return zeros (rigid ground).
        """
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        xn = self.x_nodes
        le = xn[1] - xn[0]
        ne = xn.size - 1
        inside = self.on_deck(x)
        xc = np.clip(x, 0.0, self.params.L)
        e = np.minimum((xc / le).astype(int), ne - 1)
        xi = (xc - xn[e]) / le
        a, b = self.nodal[e], self.nodal[e + 1]          # (npts, 6, nm)
        lin = lambda k: (1 - xi)[:, None] * a[:, k] + xi[:, None] * b[:, k]
        ux, rx = lin(0), lin(3)
        if self.interp == "hermite":
            H, dH = _hermite(xi)
            H, dH = H[:, :, None], dH[:, :, None]
            uz = H[0] * a[:, 2] - H[1] * le * a[:, 4] + H[2] * b[:, 2] - H[3] * le * b[:, 4]
            uy = H[0] * a[:, 1] + H[1] * le * a[:, 5] + H[2] * b[:, 1] + H[3] * le * b[:, 5]
            duz = (dH[0] * a[:, 2] - dH[1] * le * a[:, 4]
                   + dH[2] * b[:, 2] - dH[3] * le * b[:, 4]) / le
            duy = (dH[0] * a[:, 1] + dH[1] * le * a[:, 5]
                   + dH[2] * b[:, 1] + dH[3] * le * b[:, 5]) / le
            ry, rz = -duz, duy
        elif self.interp == "linear":       # legacy (the earlier study) interpolation
            uy, uz, ry, rz = lin(1), lin(2), lin(4), lin(5)
        else:
            raise ValueError(self.interp)
        S = np.empty((x.size, 3, self.nm))
        S[:, 0] = ux + self.ztop * ry - y[:, None] * rz
        S[:, 1] = uy - self.ztop * rx
        S[:, 2] = uz + y[:, None] * rx
        S[~inside] = 0.0
        return S


class SpineBeamDeck:
    """Finite-element model of the S1 deck."""

    def __init__(self, params: DeckParams = DeckParams()):
        self.params = p = params
        self.sec = s = section_properties(p)
        ne = p.n_elem
        self.x_nodes = np.linspace(0.0, p.L, ne + 1)
        self.le = le = p.L / ne
        nn = ne + 1
        K = np.zeros((6 * nn, 6 * nn))
        E, G = p.E, s["G"]
        EA, GJ, EIy, EIz = E * s["A"], G * s["J"], E * s["I"], E * s["Ilat"]
        k_ax = EA / le * np.array([[1.0, -1.0], [-1.0, 1.0]])
        k_to = GJ / le * np.array([[1.0, -1.0], [-1.0, 1.0]])
        l = le
        # bending in the x-z plane, DOFs (uz1, ry1, uz2, ry2), psi_y = -duz/dx
        k_v = EIy / l ** 3 * np.array([[12, -6 * l, -12, -6 * l],
                                       [-6 * l, 4 * l * l, 6 * l, 2 * l * l],
                                       [-12, 6 * l, 12, 6 * l],
                                       [-6 * l, 2 * l * l, 6 * l, 4 * l * l]], float)
        # bending in the x-y plane, DOFs (uy1, rz1, uy2, rz2), psi_z = +duy/dx
        k_l = EIz / l ** 3 * np.array([[12, 6 * l, -12, 6 * l],
                                       [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                                       [-12, -6 * l, 12, -6 * l],
                                       [6 * l, 2 * l * l, -6 * l, 4 * l * l]], float)
        m = np.zeros(6 * nn)
        for e in range(ne):
            i, j = 6 * e, 6 * (e + 1)
            for ke, dofs in ((k_ax, (i + 0, j + 0)), (k_to, (i + 3, j + 3)),
                             (k_v, (i + 2, i + 4, j + 2, j + 4)),
                             (k_l, (i + 1, i + 5, j + 1, j + 5))):
                K[np.ix_(dofs, dofs)] += ke
            for base in (i, j):
                m[base:base + 3] += 0.5 * p.rho * s["A"] * le
                m[base + 3] += 0.5 * p.rho * s["Ip"] * le
        fixed = [0, 1, 2, 3, 6 * ne + 1, 6 * ne + 2, 6 * ne + 3]
        self.fixed = np.array(fixed)
        self.free = np.setdiff1d(np.arange(6 * nn), self.fixed)
        self.K = K[np.ix_(self.free, self.free)]
        self.Mdiag = m[self.free]
        self.n_nodes = nn
        self._eig = None
        w = self._eigen()[0]
        w1, w2 = w[0], w[1]
        self.alphaM = p.zeta * 2 * w1 * w2 / (w1 + w2)
        self.betaK = p.zeta * 2 / (w1 + w2)

    # ------------------------------------------------------------------ data
    @property
    def M(self) -> np.ndarray:
        return np.diag(self.Mdiag)

    @property
    def C(self) -> np.ndarray:
        return self.alphaM * self.M + self.betaK * self.K

    @property
    def total_mass(self) -> float:
        return self.params.rho * self.sec["A"] * self.params.L

    def dof_index(self, node: int, dof: int) -> int:
        """Row in the free-DOF system, or -1 if the DOF is restrained."""
        g = 6 * node + dof
        k = np.searchsorted(self.free, g)
        return int(k) if k < self.free.size and self.free[k] == g else -1

    # ---------------------------------------------------------------- modes
    def _eigen(self):
        """All modes; massless bending rotations are condensed out exactly."""
        if self._eig is None:
            mm = self.Mdiag > 0
            Kmm, Kms = self.K[np.ix_(mm, mm)], self.K[np.ix_(mm, ~mm)]
            Kss = self.K[np.ix_(~mm, ~mm)]
            T = -np.linalg.solve(Kss, Kms.T)               # slave = T @ master
            Kc = Kmm + Kms @ T
            Kc = 0.5 * (Kc + Kc.T)
            lam, V = eigh(Kc, np.diag(self.Mdiag[mm]))
            Phi = np.zeros((self.free.size, lam.size))
            Phi[mm] = V
            Phi[~mm] = T @ V
            self._eig = (np.sqrt(np.maximum(lam, 0.0)), Phi)
        return self._eig

    def _labels(self, Phi_full):
        """Classify modes as B / L / T / A the way the baseline does."""
        yedge = 0.5 * self.params.W
        counts = {"B": 0, "T": 0, "L": 0, "A": 0}
        out = []
        for k in range(Phi_full.shape[2]):
            sh = Phi_full[:, :, k]
            cand = {"B": np.abs(sh[:, 2]).max(), "L": np.abs(sh[:, 1]).max(),
                    "T": np.abs(sh[:, 3]).max() * yedge, "A": np.abs(sh[:, 0]).max()}
            kind = max(cand, key=cand.get)
            counts[kind] += 1
            out.append(f"{kind}{counts[kind]}")
        return out

    def modal(self, f_max: float | None = None, n_modes: int | None = None,
              interp: str = "hermite") -> ModalDeck:
        """Truncated modal model (all modes below ``f_max`` or first ``n_modes``)."""
        w, Phi = self._eigen()
        if n_modes is None:
            n_modes = w.size if f_max is None else int(np.sum(w <= 2 * np.pi * f_max))
        w, Phi = w[:n_modes], Phi[:, :n_modes]
        full = np.zeros((6 * self.n_nodes, n_modes))
        full[self.free] = Phi
        nodal = full.reshape(self.n_nodes, 6, n_modes)
        zeta = 0.5 * (self.alphaM / w + self.betaK * w)
        return ModalDeck(params=self.params, omega=w, zeta=zeta,
                         labels=self._labels(nodal), nodal=nodal,
                         x_nodes=self.x_nodes, ztop=self.sec["ztop"], interp=interp,
                         meta=dict(alphaM=self.alphaM, betaK=self.betaK,
                                   n_total_modes=self._eigen()[0].size,
                                   f_cut=f_max))

    def frequencies(self, f_max: float = 50.0):
        md = self.modal(f_max=f_max)
        return md.freq_hz, md.labels
