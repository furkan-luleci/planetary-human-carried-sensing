#!/bin/bash
# Upright-trunk re-tuning (2026-11-19).  Every gait is re-tuned with one target added:
# mean forward lean of the trunk <= 0.07 rad (tolerance 0.025 rad), an appearance bound.
# Each case starts from its previous (leaning) result, kept in outputs/gait/_leaning_v1/;
# the previous gaits fall when their lean is simply reduced, so the tuning brings the trunk
# upright gradually.  Resumable: finished cases are skipped, interrupted ones resume.
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}
O=outputs/gait; L=$O/_leaning_v1
export OMP_NUM_THREADS=1
run() { out=$1; shift; [ -f "$out" ] && { echo "skip $out"; return; }; $PY experiments/phase_C_walking/optimize_gait.py "$@" --out "$out"; }
TR='"trunk_lean":[0.07,0.025,"max"]'
# Earth 1.0 m/s first: the 3-D walker starts from its gait cycle
run $O/earth_1.00.json --g 9.81 --speed 1.0 --x0 $L/earth_1.00.json --random 24 --sigma 0.08 --maxiter 60
# 3-D walker (Earth, 1.0 m/s); its planar seed is regenerated from the new Earth gait
if [ ! -f $O/earth3d_1.00.json ]; then
  [ -f $O/_start3d.json ] || $PY experiments/phase_G_walk3d/pick_start3d.py
  $PY experiments/phase_G_walk3d/optimize_gait3d.py --g 9.81 --speed 1.0 --x0 $O/_start3d.json \
      --random 24 --sigma 0.04 --popsize 12 --maxiter 40 --out $O/earth3d_1.00.json
fi
echo DONE > $O/_stage1_done
run $O/earth_1.25.json --g 9.81 --speed 1.25 --x0 $L/earth_1.25.json --random 24 --sigma 0.08 --maxiter 60
$PY experiments/phase_C_walking/similarity_cases.py
T_MARS='{"speed":[1.0,0.04,"eq"],"step_frequency":[1.72,0.10,"eq"],"duty_factor":[0.60,0.02,"eq"],"grf_foot_peak_over_W":[1.6,0.2,"max"],'$TR'}'
T_MOON='{"speed":[1.0,0.04,"eq"],"step_frequency":[1.72,0.10,"eq"],"duty_factor":[0.58,0.025,"eq"],"grf_foot_peak_over_W":[1.8,0.25,"max"],'$TR'}'
run $O/mars_1.00.json --g 3.71 --speed 1.0 --no-similarity --g-gain 9.81 --x0 $L/mars_1.00.json --random 24 --sigma 0.08 --maxiter 60 --targets "$T_MARS"
run $O/moon_1.00.json --g 1.62 --speed 1.0 --no-similarity --g-gain 9.81 --x0 $L/moon_1.00.json --random 24 --sigma 0.08 --maxiter 60 --targets "$T_MOON"
echo ALL_DONE > $O/_all_done
