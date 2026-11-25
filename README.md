# Habitat floor sensing with occupants' phones and watches

**Furkan Lüleci and Jovdat Babazade** · Louisiana State University

Contributors: [@furkan-luleci](https://github.com/furkan-luleci) (Furkan Lüleci) · [@covdetbaba](https://github.com/covdetbaba) (Jovdat Babazade)

An occupant of a lunar or Martian habitat wears a watch on each wrist and carries a phone in each front trouser pocket. This code simulates the occupant and the floor together under Earth, Mars, and Moon gravity and records what the devices measure.

| | |
|---|---|
| Viewer | https://furkan-luleci.github.io/planetary-human-carried-sensing/ |
| License | MIT. See [LICENSE](LICENSE). |

## 1. What is modelled 

**Structures.** A floor of NASA's Common Habitat, built like the floor of NASA's Habitat Demonstration Unit and sized to its documented loads (the habitat floor, the main case), a version sized for lunar weight only, and the 14 m concrete bridge model of the earlier simulation study (the reference bridge). Each is a modal model, the same on every planet.

**Occupant.** A 75 kg, 1.75 m articulated body of 15 segments, with leg stiffness that grows with the load the legs carry, tuned to published Earth-gravity measurements of standing resonance and damping. Seven calibrated bodies and a control body show how the results depend on these assumptions. The occupant stands, walks at 1.0 m/s with a side-view or a 3-D walker tuned to published gait targets, or walks with what-if gaits that transfer an Earth gait by dynamic similarity.

**Coupling.** Body and structure are solved together. The feet touch the moving surface through unilateral contact with friction. Gravity enters only the body's equations of motion, so every difference between the planets comes from the occupant.

**Devices.** A watch on each wrist and a phone in each front trouser pocket, all triaxial, sampled at 200 Hz with sensor noise.

**Excitation.** Three shakers on the structure, or the occupant's footsteps with a weak broadband force.

**Analysis.** Transmission from the structure to each device, visibility of each natural frequency above the wearer's own motion, blind recovery of the natural frequencies, gait removal using each device's own step timing, and a rigid-ground twin of every walk to measure chance.

## 2. Viewer

Open https://furkan-luleci.github.io/planetary-human-carried-sensing/, or download [`viewer/Planetary_Sensing_Viewer.html`](viewer/Planetary_Sensing_Viewer.html) and open it in a browser. It is one file, works offline, and shows stored results.

| Sheet | What it shows |
|----|----|
| S-1 Watch a trial | A trial in 3-D or as a flat drawing, with the floor motion, the device signals in time and frequency, and the key numbers of the trial |
| S-2 What gravity changes | Standing-body resonance, walking characteristics, transmission to the devices, and the occupant's effect on the floor |
| S-3 Can the structure be detected? | Visibility, blind recovery against the rigid-ground chance level, gait removal and device attachment |
| S-4 Model and checks | The setting, the model and every verification check with its tolerance |

Every chart can be saved as an image (PNG) or as data (CSV).

## 3. Run it

Linux, macOS or WSL, Python 3.10 or newer, and Node.js (the viewer build uses it to check the page's code).

```bash
git clone https://github.com/furkan-luleci/planetary-human-carried-sensing.git
cd planetary-human-carried-sensing
pip install -e ".[dev]"
python tools/fetch_companion_code.py
```

The last command writes two modules taken from the earlier study's repository, [human-carried-indirect-sensing](https://github.com/furkan-luleci/human-carried-indirect-sensing). See Section 5.

Run the verification tests and rebuild the viewer from the stored results:

```bash
python -m pytest
python viewer/build_viewer.py
```

Run the whole study again (long: 1380 simulated trials, many hours on a multi-core machine). The stages, in order:

```bash
python experiments/phase_B_body/calibrate_leg.py                    # calibrate the standing body
bash experiments/phase_C_walking/run_upright_gaits.sh               # tune the side-view walker
bash experiments/phase_G_walk3d/run_upright3d.sh                    # tune the 3-D walker on Earth
bash experiments/phase_G_walk3d/run_stage2.sh                       # 3-D walker on Mars and the Moon
python experiments/phase_E_records/run_records.py --structure S1    # trials on the reference bridge
python experiments/phase_E_records/run_records.py --structure S2    # trials on the habitat floor
python experiments/phase_E_records/run_records.py --structure S2L   # trials on the lunar-weight floor
python experiments/phase_O_consistency/whatif_floor_trials.py       # what-if gaits on the habitat floor
python experiments/phase_F_outcomes/outcomes.py                     # transmission, visibility, blind recovery
python experiments/phase_H_mounting/mounting_check.py               # device attachment
python experiments/phase_J_step_timing/step_timing_test.py          # gait removal, reference bridge
python experiments/phase_K_S2/step_timing_s2.py S2                  # gait removal, habitat floor
python experiments/phase_K_S2/hsi_bodies.py                         # occupied first mode, every body
python experiments/phase_K_S2/check_phaseK.py                       # criteria C1 to C4
python experiments/phase_F_outcomes/verification_summary.py         # verification table
python viewer/build_viewer.py                                       # the viewer
```

The other scripts in `experiments/` give the study's sensitivity checks (clock accuracy, timing jitter, sampling rate, occupied-mode sensitivity, longer walks, device signals).

## 4. Files

| Path | Role |
|---|---|
| `src/planetsense/` | The model: structures, articulated body and leg compliance, foot contact, controllers, excitation, devices, coupled integration, spectral analysis, identification and evaluation |
| `configs/` | Calibration of the leg element |
| `experiments/` | Scripts that run the trials and the analyses, one folder per stage |
| `tests/` | Verification tests, with their tolerances in `tests/tolerances.py` |
| `outputs/tables/` | Stored results per case (JSON, with logs) |
| `outputs/processed/`, `outputs/gait/` | Processed curves and the tuned gait parameters |
| `outputs/raw/` | Repetition 0 of every trial, the records the viewer animates. The full simulated records (about 2.1 GB) are available from the authors on request |
| `viewer/` | The viewer, its source (`viewer/src/`) and its build script |
| `tools/fetch_companion_code.py` | Writes the two modules taken from the earlier study's code |

In the code and the result files, `S2` is the habitat floor, `S2L` the floor sized for lunar weight only and `S1` the reference bridge.

## 5. Code from the earlier study

`src/planetsense/evaluation/legacy_pipeline.py` (spectral processing and gait-harmonic removal) and `src/planetsense/evaluation/legacy_oi.py` (the Observability Index) are character-for-character copies of definitions in the earlier study's repository, [human-carried-indirect-sensing](https://github.com/furkan-luleci/human-carried-indirect-sensing). That code has its own licence, so the copies are not stored here; `tools/fetch_companion_code.py` writes them from that repository. Tests that need them are skipped until they are written.

## 6. Third-party components

The viewer embeds three.js r149 (MIT licence, `viewer/vendor/THREE_LICENSE.txt`) and the B612 and B612 Mono typefaces (SIL Open Font License 1.1, `viewer/fonts/`).

© 2026 Furkan Lüleci and Jovdat Babazade. MIT licence.
