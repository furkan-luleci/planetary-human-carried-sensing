"""The structures of the study and their trial settings (Phase K, 2026-11-21).

  S1   the earlier study's 14 m footbridge (spine_beam.SpineBeamDeck), unchanged
  S2   a floor of NASA's Common Habitat mid deck built like the Habitat Demonstration Unit's
       floor, sized to the HDU criterion (habitat_deck.HabitatDeck)
  S2L  the same floor layout with members sized for lunar weight only (reduced-weight bracket)

Settings per structure: where the person stands, the walking line, and the shaker level.
S1 keeps every setting it had.  For S2 and S2L the shaker level follows the rule declared in
docs/phase_reports/PHASE_K_declared_rules.md: the EMPTY structure has the same RMS vertical
acceleration at its standing point as the empty S1 has at its standing point under 800 N per
component (computed in the frequency domain from the shaker spectrum).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..excitation.multisine import shaker_layout
from .habitat_deck import HabitatDeck, build
from .spine_beam import SpineBeamDeck

S1_SHAKER_RMS, S1_FLOOR_RMS = 800.0, 5.0
S1_STAND, S1_LINE = (4.2, 0.575), 0.575


@dataclass(frozen=True)
class Structure:
    name: str
    stand_xy: tuple
    line_y: float
    shaker_rms: float
    floor_rms: float
    description: str


def _hd(name):
    return {"S2": HabitatDeck(), "S2L": HabitatDeck(g_ratio=1.62 / 9.81, n_spans=2)}[name]


@lru_cache(maxsize=None)
def modal_deck(name: str = "S1", f_max: float = 150.0):
    """Modal model of a structure; S1 exactly as before."""
    if name == "S1":
        return SpineBeamDeck().modal(f_max=f_max)
    md = build(_hd(name)).modal(f_max=f_max)
    md.meta["line_y"] = 0.0
    md.meta["structure"] = name
    return md


def _acc_frf(md, x_in, y_in, d_in, x_out, y_out, f):
    b = md.surface_modes([x_in], [y_in])[0, d_in, :]
    c = md.surface_modes([x_out], [y_out])[0, 2, :]
    w = 2 * np.pi * np.asarray(f)[:, None]
    H = (c * b)[None, :] / (md.omega[None, :] ** 2 - w ** 2 + 2j * md.zeta[None, :] * md.omega[None, :] * w)
    return -(w[:, 0] ** 2) * H.sum(axis=1)


def empty_rms(md, x_out, y_out, rms, floor_rms, T=200.0):
    """RMS vertical acceleration of the empty structure at (x_out, y_out) under the three shakers
    (flat to 75 Hz, 4th-order Butterworth roll-off to 150 Hz, lines from 0.2 Hz)."""
    k = np.arange(int(np.ceil(0.2 * T)), int(np.floor(150.0 * T)) + 1)
    f = k / T
    prof = 1.0 / np.sqrt(1.0 + (f / 75.0) ** 8)
    unit = prof / np.sqrt(0.5 * np.sum(prof ** 2))
    tot = 0.0
    for xs, ys in shaker_layout(md.params.L, md.params.W):
        for d, r in ((0, rms), (1, rms), (2, np.hypot(rms, floor_rms))):
            tot += 0.5 * np.sum((r * unit) ** 2 * np.abs(_acc_frf(md, xs, ys, d, x_out, y_out, f)) ** 2)
    return float(np.sqrt(tot))


@lru_cache(maxsize=None)
def get(name: str = "S1") -> Structure:
    if name == "S1":
        return Structure("S1", S1_STAND, S1_LINE, S1_SHAKER_RMS, S1_FLOOR_RMS,
                         "the earlier study's 14 m footbridge")
    md = modal_deck(name)
    # standing point: 40 % of the first span from the wall, on a beam line; chosen, as for S1, so that
    # no mode below 30 Hz has a node there (checked in tests and by experiments/phase_K_scope_S2)
    stand = (0.2 * md.params.L, 0.0)
    target = empty_rms(modal_deck("S1"), *S1_STAND, S1_SHAKER_RMS, S1_FLOOR_RMS)
    unit = empty_rms(md, *stand, 1.0, S1_FLOOR_RMS / S1_SHAKER_RMS)
    sh = target / unit
    desc = {"S2": "Common Habitat mid-deck floor, HDU construction, sized to the HDU criterion",
            "S2L": "the same floor, members sized for lunar weight only"}[name]
    return Structure(name, stand, 0.0, float(sh), float(sh * S1_FLOOR_RMS / S1_SHAKER_RMS), desc)
