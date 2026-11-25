"""Phase K, for the occupied-floor curves (no new verdict; every quantity below was
already evaluated as a number in Phase K, here it is kept as a curve):
(1) linear model: vertical driving-point accelerance of the floor under the feet at the standing
    point, empty and with a person standing (nominal calibrated body; control body as a check),
    on Earth, Mars and the Moon, for S2 and S1;
(2) standing trials (shakers on, 20 records of 60 s): ensemble spectrum of the floor's vertical
    acceleration under the feet near S2's first mode, the quantity of the declared criterion 4.
Writes outputs/tables/occupied_curves.json.

    PYTHONPATH=src python experiments/phase_K_S2/occupied_curves.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_K_scope_S2"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase_F_outcomes"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from scope_s2 import G, empty_acc_frf  # noqa: E402
from scope_s2_variants import occupied_variant  # noqa: E402
from planetsense.structure import registry as SR  # noqa: E402

out = dict(note="occupied-floor curves; linear model: vertical acceleration under the feet per unit vertical "
                "force under the feet ((m/s^2)/N); trials: ensemble PSD of the floor's vertical acceleration under "
                "the feet ((m/s^2)^2/Hz), standing, shakers on, 20 records, 16 s Welch segments", linear={}, trials={})
for name, f in (("S2", np.linspace(6.0, 20.0, 2801)), ("S1", np.linspace(0.8, 20.0, 3841))):
    md, st = SR.modal_deck(name), SR.get(name)
    x, y = st.stand_xy
    d = dict(f=f.tolist(), empty=np.abs(empty_acc_frf(md, x, y, 2, x, y, f)).tolist(), modes={
        l: float(q) for l, q in zip(md.labels, md.freq_hz) if q < 30.0})
    for p, g in G.items():
        d[p] = np.abs(occupied_variant(md, g, x, y, f, "nominal")).tolist()
        d[p + "_control"] = np.abs(occupied_variant(md, g, x, y, f, "fully_linear")).tolist()
        print(name, p, "done", flush=True)
    out["linear"][name] = d

os.environ["PLANETSENSE_STRUCTURE"] = "S2"
import outcomes as OC  # noqa: E402
from planetsense.identification.spectral import ensemble_csd  # noqa: E402

fb1 = float(SR.modal_deck("S2").freq_hz[0])
for p in G:
    recs = OC.load_group("standing", "E1", p, None)
    Ds = [np.asarray(r["a_under"], float)[OC.window(r)][:, 2:3] for r in recs]
    f, Gm, _ = ensemble_csd(Ds, OC.FS_IMU, int(16.0 * OC.FS_IMU))
    k = (f >= 6.0) & (f <= 20.0)
    out["trials"][p] = dict(f=f[k].tolist(), psd=Gm[k, 0, 0].real.tolist(), n_records=len(recs),
                            peak_hz=OC.occupied_estimate(recs, fb1))
    print("trials", p, len(recs), out["trials"][p]["peak_hz"], flush=True)
json.dump(out, open(os.path.join(ROOT, "outputs", "tables", "occupied_curves.json"), "w"))
