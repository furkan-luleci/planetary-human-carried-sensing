"""Collect the verification evidence into one table (for the
viewer): the automated test suite with its pre-declared tolerances, and audits
of every simulated trial.

    python experiments/phase_F_outcomes/verification_summary.py
"""
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from tolerances import TOL                                    # noqa: E402
from planetsense.simulation.records import load_record             # noqa: E402

PLAIN = [  # (test-name pattern, what it checks in plain words, limit)
    ("test_deck_vs_baseline|test_table3|test_deck_frequencies", "Model of the reference bridge (the earlier study's bridge model) reproduces that study's model (the frequencies of its Table 3, its matrices)", "1e-4 Hz"),
    ("test_gravity_static_support", "Standing body is held up by the feet with exactly its weight, on every planet", f"{TOL['static_weight_rel']:g} of weight"),
    ("test_static_support_with_leg_compliance", "Same, with the calibrated springy legs", f"{TOL['static_weight_rel']:g} of weight"),
    ("test_quiet_standing", "Quiet standing stays put and sensors read local gravity", f"{TOL['standing_mean_force_rel']:g}"),
    ("test_weight_scales", "Weight changes with gravity, mass does not", "1e-9"),
    ("test_free_flight", "Energy and momentum conserved in free flight (no contact)", f"{TOL['energy_free_flight_rel']:g}"),
    ("test_energy_balance_passive|test_energy_passive_drop", "Work-energy balance of person + structure + contact (drop test)", f"{TOL['energy_balance_rel']:g}"),
    ("test_contact_unilateral", "Feet can push but never pull on the floor", "0 N"),
    ("test_action_reaction", "Person and structure receive equal and opposite forces", f"{TOL['action_reaction_rel']:g}"),
    ("test_multibody_dynamics_and_sensors_agree|test_model_variants_agree", "Body dynamics and virtual sensors agree with an independent physics engine (MuJoCo)", f"{TOL['cross_solver_rel']:g}"),
    ("test_compiled_path|kernel", "Fast compiled code gives the same answer as the reference code", f"{TOL['kernel_rel']:g}"),
    ("test_stored_calibration", "Calibrated body reproduces its published resonance and damping targets", "1 % / 2 %"),
    ("test_leg_unloads", "Legs compress and soften under lower weight (the mechanism studied)", "exact"),
    ("test_walker_mechanics", "Walking on the structure: force balance and work-energy balance hold", f"{TOL['walk_energy_balance_rel']:g}"),
    ("test_walker3d_mechanics", "3-D walking on the structure: force balance in all three directions and work-energy balance hold", f"{TOL['walk3d_energy_balance_rel']:g}"),
    ("test_rigid", "A structure with no flexibility behaves exactly like solid ground", f"{TOL['rigid_limit_rel']:g}"),
    ("test_convergence|test_dt", "Results converge as the time step shrinks", f"{TOL['dt_error_rel']:g}"),
    ("test_sensor", "Virtual accelerometer and gyroscope formulas", f"{TOL['sensor_static_abs']:g} m/s²"),
    ("test_vendored", "Earlier study's index and spectral processing reused unchanged and give identical output", "identical"),
    ("test_shaker|test_deck_lead_in", "Shakers, 5 N broadband force and lead-in of the structure set up as in the earlier study", "1e-9"),
    ("test_repeatab", "Same inputs give bit-identical results", "identical"),
    ("test_device_clock|test_counting_survives|test_two_events|test_four_similar", "A device's own gait clock (synthetic gait): right number of events per stride and stride period, no count lost at quick steps, random timing error under 1 ms", "2 %; 1 ms"),
    ("test_each_side_alone", "Each side processed with its own clock gives the earlier study's two-sided result", "1e-12"),
    ("test_s1_unchanged|test_s2_settings|test_window_uses", "Habitat floor: the reference bridge unchanged by the generalised code; the floor's settings follow the declared rules", "identical / rules"),
    ("test_finds_every_line|test_line_below|test_harmonic_rejection|test_matching_is_one_to_one|test_identifier_sees_only",
     "Blind identifier: finds lines above the prominence threshold, rejects gait harmonics, matches one-to-one within the tolerance, never sees the true frequencies", "exact"),
]

r = subprocess.run([sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-rA", "--tb=no", "-p", "no:cacheprovider"],
                   cwd=ROOT, capture_output=True, text=True)
lines = [l for l in r.stdout.splitlines() if l.startswith(("PASSED", "FAILED", "SKIPPED", "ERROR"))]
summary = [l for l in r.stdout.splitlines() if re.search(r"\d+ passed", l)]
open(os.path.join(ROOT, "outputs", "tables", "pytest_run_latest.txt"), "w").write(r.stdout)
checks = []
matched = set()
for pat, what, lim in PLAIN:
    hit = [l for l in lines if re.search(pat, l)]
    matched.update(hit)
    if not hit:
        continue
    npass = sum(l.startswith("PASSED") for l in hit)
    nfail = sum(l.startswith(("FAILED", "ERROR")) for l in hit)
    nskip = sum(l.startswith("SKIPPED") for l in hit)
    checks.append(dict(what=what, result=f"{npass} of {npass + nfail} cases passed" + (f", {nskip} skipped" if nskip else ""),
                       limit=lim, pass_=nfail == 0))
rest = [l for l in lines if l not in matched]           # every test appears in exactly one row of the table
if rest:
    npass = sum(l.startswith("PASSED") for l in rest)
    nfail = sum(l.startswith(("FAILED", "ERROR")) for l in rest)
    nskip = sum(l.startswith("SKIPPED") for l in rest)
    checks.append(dict(what="Other unit tests (body model set-up, gait controllers, records, filters and helpers)",
                       result=f"{npass} of {npass + nfail} cases passed" + (f", {nskip} skipped" if nskip else ""),
                       limit="per test", pass_=nfail == 0))
# audits of every simulated trial
recs = sorted(glob.glob(os.path.join(ROOT, "outputs", "raw", "records", "*.npz")) +      # S1
              glob.glob(os.path.join(ROOT, "outputs", "raw", "records_S2", "*.npz")) +   # S2 (Phase K)
              glob.glob(os.path.join(ROOT, "outputs", "raw", "records_S2L", "*.npz")))   # S2L (Phase K)
drift, falls, slips = [], 0, 0
for p in recs:
    z = load_record(p)
    drift.append(float(z["energy_drift_rel"]))
    falls += int(bool(z["fell"]))
if recs:
    checks.append(dict(what=f"Energy book-keeping closes in every simulated trial ({len(recs)} trials)",
                       result=f"largest error {max(drift):.1e} of the energy exchanged", limit="1e-3", pass_=max(drift) < 1e-3))
    checks.append(dict(what="No simulated person fell", result=f"{falls} falls in {len(recs)} trials", limit="0", pass_=falls == 0))
for c in checks:
    c["pass"] = c.pop("pass_")
out = dict(checks=checks, pytest_summary=summary[-1] if summary else "", n_tests=len(lines),
           note="Limits were written into tests/tolerances.py before the tests were run. "
                "Full log: outputs/tables/pytest_run_latest.txt.")
json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "verification_v2.json"), "w"), indent=1)
print(out["pytest_summary"])
for c in checks:
    print(("PASS " if c["pass"] else "FAIL ") + c["what"] + " | " + c["result"])
