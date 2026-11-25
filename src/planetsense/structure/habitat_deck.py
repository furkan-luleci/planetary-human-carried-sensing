"""S2, the second structure (Phase K, 2026-11-21): a floor of the NASA Common
Habitat's mid deck, built the way NASA built the floor of its Habitat Demonstration Unit,
sized here to the documented loads.  No number below is chosen to suit the study; each
is documented, derived from documented numbers, or marked ASSUMED.

Documented
  [CH]  Howard R.L. (2021) Internal Architecture of the Common Habitat, NASA NTRS 20210021782.
        Habitat for the Moon and Mars: 8.4 m diameter, three decks; mid-deck floor 15.24 cm
        thick (floor structure plus utilities); mid-deck floor 16 in below the bottom of the
        hatches, which are centred on the vessel axis.
  [HDU] Smith R.W., Langford W.M. (2012) Design, Analysis and Fabrication of Secondary
        Structural Components for the Habitat Demonstration Unit - Deep Space Habitat, NASA
        NTRS 20120008160.  Floor: aircraft-grade honeycomb sandwich panels (Nomex core
        9 lb/ft^3, 0.66 in; facesheets 3 plies x 0.010 in woven E-glass/epoxy), two panels
        stacked (1.44 in); floor beams at 20 in centres; designed for 50 lb/ft^2 live load
        with a factor of safety of 1.5.
  [Y89] Yin P.K. (1989) A preliminary design of interior structure and foundation of an
        inflatable lunar habitat, NASA NTRS 19900015683.  Interior floor framing of aluminium
        2219 (E = 73.1 GPa, yield 351.6 MPa).
Derived
  span: the mid-deck chord, 8.2 m (hatch opening of 1.0-1.2 m ASSUMED: 8.16-8.20 m);
  beam depth: deck thickness minus the two panels, 115.8 mm;
  supports: no beam of that depth spans 8.2 m under the HDU load and an L/360 deflection
  limit, so the deck needs a central support: two continuous spans of 4.1 m; the lightest
  aluminium I-beam that meets strength (FS 1.5) and deflection, with flange outstands no
  more slender than 12 thicknesses, is used.
  effective width: the floor shares load between beams; the width of floor that moves with
  one beam line follows the joist-mode rule of AISC Design Guide 11 (B = 2.0 (Ds/Dj)^0.25 L).
ASSUMED: L/360 live-load deflection limit; E-glass facesheet modulus 20 GPa and density
  1900 kg/m^3; utilities and ceiling 10 kg/m^2; aluminium density 2840 kg/m^3; damping 1 %;
  twist of the strip restrained (it is one beam line of a continuous floor; plate modes are outside a
  one-dimensional model).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh

from . import spine_beam as SB

LBFT2 = 47.880            # Pa per lb/ft^2
IN = 0.0254


@dataclass(frozen=True)
class HabitatDeck:
    # documented
    D_vessel: float = 8.4                 # [CH]
    t_deck: float = 0.1524                # [CH] mid deck
    s_beam: float = 20 * IN               # [HDU]
    q_live: float = 50 * LBFT2            # [HDU]
    FS: float = 1.5                       # [HDU]
    t_face: float = 3 * 0.010 * IN        # [HDU]
    h_core: float = 0.66 * IN             # [HDU]
    rho_core: float = 9 * 16.0185         # [HDU] 9 lb/ft^3
    n_panels: int = 2                     # [HDU] stacked, 1.44 in
    E_al: float = 73.1e9                  # [Y89]
    fy_al: float = 351.6e6                # [Y89]
    # derived geometry (hatch opening 1.0 m ASSUMED; 1.2 m gives 8.16 m)
    hatch: float = 1.0
    # assumed
    defl_limit: float = 360.0
    E_face: float = 20e9
    G_face: float = 3.5e9
    rho_face: float = 1900.0
    m_util: float = 10.0
    rho_al: float = 2840.0
    t_web: float = 3e-3
    outstand: float = 12.0
    zeta: float = 0.01
    n_elem: int = 40
    # variants (pre-run checks): width of floor moving with one beam line, as a multiple of the
    # Design Guide 11 value; gravity the floor is sized for, as a fraction of Earth's (1 = the
    # Earth-tested HDU criterion; 1.62/9.81 = sized for lunar weight only, as in Yin 1989)
    B_factor: float = 1.0
    g_ratio: float = 1.0
    n_spans: int | None = None      # None: one span if any beam can carry it, else two (the design rule);
                                     # variants keep the baseline's layout (2) and resize the members

    @property
    def span_total(self):
        z = self.hatch / 2 + 16 * IN
        return float(2 * np.sqrt((self.D_vessel / 2) ** 2 - z ** 2))

    @property
    def d_beam(self):
        return self.t_deck - self.n_panels * (self.h_core + 2 * self.t_face)

    @property
    def panel_mass(self):                 # kg/m^2
        return self.n_panels * (self.rho_core * self.h_core + 2 * self.t_face * self.rho_face)

    def _section(self, bf, tf):
        d, tw = self.d_beam, self.t_web
        hw = d - 2 * tf
        A = 2 * bf * tf + hw * tw
        I = 2 * (bf * tf ** 3 / 12 + bf * tf * (d / 2 - tf / 2) ** 2) + tw * hw ** 3 / 12
        return A, I

    def size_beam(self, n_spans):
        """Lightest I-beam of the available depth meeting strength and deflection, or None."""
        L = self.span_total / n_spans
        kdef = 5 / 384 if n_spans == 1 else 1 / 185            # max deflection coefficient, UDL
        best = None
        for bf in np.arange(0.04, 0.2001, 0.005):
            for tf in np.arange(0.002, 0.02001, 0.0005):
                if (bf - self.t_web) / 2 / tf > self.outstand:
                    continue
                A, I = self._section(bf, tf)
                w_dead = ((self.panel_mass + self.m_util) * self.s_beam + self.rho_al * A) * 9.81 * self.g_ratio
                M = (self.FS * self.q_live * self.g_ratio * self.s_beam + w_dead) * L ** 2 / 8   # simple span / support of two equal spans
                if M * (self.d_beam / 2) / I > self.fy_al:
                    continue
                if kdef * self.q_live * self.g_ratio * self.s_beam * L ** 4 / (self.E_al * I) > L / self.defl_limit:
                    continue
                if best is None or A < best["A"]:
                    best = dict(bf=float(bf), tf=float(tf), A=float(A), I=float(I), L=float(L), n_spans=n_spans)
        return best

    def design(self) -> dict:
        one = self.size_beam(1)
        if self.n_spans is not None:
            b = self.size_beam(self.n_spans)
        else:
            b = self.size_beam(2) if one is None else one
        h = self.h_core + self.t_face                          # facesheet centroid distance, one panel
        Ds = self.n_panels * self.E_face * self.t_face * h ** 2 / 2      # per unit width, panels not bonded
        Dj = self.E_al * b["I"] / self.s_beam
        B = float(self.B_factor * 2.0 * (Ds / Dj) ** 0.25 * b["L"])     # AISC DG11 joist mode, C = 2.0
        m_area = self.panel_mass + self.m_util + self.rho_al * b["A"] / self.s_beam
        GJ = B * self.n_panels * 2 * self.G_face * self.t_face * h ** 2          # sandwich twisting, approximate
        return dict(span_total=self.span_total, single_span_possible=one is not None, beam=b, d_beam=self.d_beam,
                    panel_mass=self.panel_mass, Ds=Ds, Dj=Dj, B_eff=B, m_area=m_area, m_per_m=m_area * B,
                    EI=self.E_al * b["I"] * B / self.s_beam, EI_lat=self.E_al * b["I"] * B / self.s_beam * (B / self.d_beam) ** 2,
                    GJ=GJ, I_mass=m_area * B * B ** 2 / 12, total_mass_strip=m_area * B * self.span_total)


@dataclass
class _Deck:
    """Spine-beam deck with supports at both walls and, for two spans, at mid-chord."""
    params: SB.DeckParams
    sec: dict
    x_nodes: np.ndarray
    K: np.ndarray
    Mdiag: np.ndarray
    free: np.ndarray
    n_nodes: int
    alphaM: float = 0.0
    betaK: float = 0.0
    meta: dict = field(default_factory=dict)

    def modal(self, f_max=150.0):
        mm = self.Mdiag > 0
        Kmm, Kms, Kss = self.K[np.ix_(mm, mm)], self.K[np.ix_(mm, ~mm)], self.K[np.ix_(~mm, ~mm)]
        T = -np.linalg.solve(Kss, Kms.T)
        Kc = Kmm + Kms @ T
        lam, V = eigh(0.5 * (Kc + Kc.T), np.diag(self.Mdiag[mm]))
        Phi = np.zeros((self.free.size, lam.size)); Phi[mm] = V; Phi[~mm] = T @ V
        w = np.sqrt(np.maximum(lam, 0.0))
        if not self.alphaM:
            w1, w2 = w[0], w[1]
            z = self.params.zeta
            self.alphaM, self.betaK = z * 2 * w1 * w2 / (w1 + w2), z * 2 / (w1 + w2)
        n = int(np.sum(w <= 2 * np.pi * f_max))
        w, Phi = w[:n], Phi[:, :n]
        full = np.zeros((6 * self.n_nodes, n)); full[self.free] = Phi
        nodal = full.reshape(self.n_nodes, 6, n)
        zeta = 0.5 * (self.alphaM / w + self.betaK * w)
        labels = SB.SpineBeamDeck._labels(self, nodal)
        return SB.ModalDeck(params=self.params, omega=w, zeta=zeta, labels=labels, nodal=nodal, x_nodes=self.x_nodes,
                            ztop=self.sec["ztop"], interp="hermite", meta=dict(self.meta, alphaM=self.alphaM, betaK=self.betaK))


def build(h: HabitatDeck = HabitatDeck()):
    d = h.design()
    b = d["beam"]
    L, ne = d["span_total"], h.n_elem
    W = d["B_eff"]
    A_eff = b["A"] * W / h.s_beam                                    # aluminium area of the strip (axial)
    rho_eff = d["m_per_m"] / A_eff
    params = SB.DeckParams(L=L, W=W, t=h.d_beam, E=h.E_al, rho=rho_eff, G_over_E=1.0, I_vert=None, zeta=h.zeta, n_elem=ne)
    sec = dict(A=A_eff, I=d["EI"] / h.E_al, Ilat=d["EI_lat"] / h.E_al, J=d["GJ"] / h.E_al, Ip=d["I_mass"] / rho_eff,
               ztop=0.5 * h.d_beam + h.n_panels * (h.h_core + 2 * h.t_face), G=h.E_al)
    x = np.linspace(0.0, L, ne + 1); le = L / ne; nn = ne + 1
    E, G = h.E_al, sec["G"]
    EA, GJ, EIy, EIz = E * sec["A"], G * sec["J"], E * sec["I"], E * sec["Ilat"]
    k_ax = EA / le * np.array([[1.0, -1.0], [-1.0, 1.0]]); k_to = GJ / le * np.array([[1.0, -1.0], [-1.0, 1.0]])
    l = le
    k_v = EIy / l ** 3 * np.array([[12, -6 * l, -12, -6 * l], [-6 * l, 4 * l * l, 6 * l, 2 * l * l],
                                   [-12, 6 * l, 12, 6 * l], [-6 * l, 2 * l * l, 6 * l, 4 * l * l]], float)
    k_l = EIz / l ** 3 * np.array([[12, 6 * l, -12, 6 * l], [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                                   [-12, -6 * l, 12, -6 * l], [6 * l, 2 * l * l, -6 * l, 4 * l * l]], float)
    K = np.zeros((6 * nn, 6 * nn)); m = np.zeros(6 * nn)
    for e in range(ne):
        i, j = 6 * e, 6 * (e + 1)
        for ke, dofs in ((k_ax, (i, j)), (k_to, (i + 3, j + 3)), (k_v, (i + 2, i + 4, j + 2, j + 4)),
                         (k_l, (i + 1, i + 5, j + 1, j + 5))):
            K[np.ix_(dofs, dofs)] += ke
        for base in (i, j):
            m[base:base + 3] += 0.5 * rho_eff * sec["A"] * le
            m[base + 3] += 0.5 * rho_eff * sec["Ip"] * le
    fixed = [0, 1, 2, 3, 6 * ne + 1, 6 * ne + 2, 6 * ne + 3]          # walls, as S1's supports
    if b["n_spans"] == 2:
        c = 6 * (ne // 2)
        fixed += [c + 1, c + 2, c + 3]                                  # central support line
    # the strip is one beam line of a continuous floor: the floor on either side keeps it from
    # twisting as a free beam would, so twist is restrained; the model represents the floor's
    # bending along the beam line (the joist mode of AISC DG11), not its plate modes
    fixed = sorted(set(fixed) | {6 * k + 3 for k in range(nn)})
    free = np.setdiff1d(np.arange(6 * nn), fixed)
    return _Deck(params=params, sec=sec, x_nodes=x, K=K[np.ix_(free, free)], Mdiag=m[free], free=free, n_nodes=nn,
                 meta=dict(kind="S2 habitat deck strip", design=d))
