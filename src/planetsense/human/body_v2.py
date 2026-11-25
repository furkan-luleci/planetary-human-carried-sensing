"""Calibrated leg-compliance parameter sets (body model v2).

Values are produced by experiments/phase_B_body/calibrate_leg.py and stored in
configs/body_v2_leg_calibration.json.  See that script for the targets.
"""
from __future__ import annotations

import json
import os

from .control import LegSpring

_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs",
                                     "body_v2_leg_calibration.json"))
NOMINAL = "nominal_n1.5_constant"


def leg_sets() -> dict:
    with open(_PATH) as fh:
        return json.load(fh)["sets"]


def load_leg(name: str = NOMINAL) -> LegSpring:
    s = leg_sets()[name]
    return LegSpring(k=s["k"], n=s["n"], c=s["c"], damping=s["damping"],
                     k_ext=s["k_ext"], c_ext=s["c_ext"])
