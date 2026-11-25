#!/bin/bash
# Upright 3-D walker (2026-11-19).  History:
#  try 1: re-tuning from the previous (leaning) 3-D gait lowered the trunk lean only from 13.5 to
#         12.0 degrees in 20 generations (outputs/gait/_upright3d_try1.log); stopped.
#  try 2: started from the upright planar gait's sagittal parameters (+ previous frontal-plane
#         parameters); balance could not be recovered in 9 generations (best cost 142, most
#         candidates fell; outputs/gait/_upright3d_try2.log); stopped.
#  Diagnosis: with less forward lean the try-1 gait does not tip over; it slows down, stops and
#  walks backwards.  A new controller parameter, lumbar_ext (extension target of the lower back;
#  0 reproduces the previous controller exactly), lets the trunk stand more upright while the
#  pelvis keeps a forward tilt.  From the try-1 best, lumbar_ext 0.05 with lean 0.18 walks
#  (trunk lean 10.6 degrees, cost 39.3): outputs/gait/_start3d_lumbar.json.
#  This run starts there; the trunk-lean bound keeps its value (0.07 rad) but its tolerance is
#  halved to 0.0125 rad for the 3-D walker (outputs/gait/_targets3d_upright.json), to favour the
#  posture change inside the narrow stable region of this controller.
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}
export OMP_NUM_THREADS=1
O=outputs/gait
[ -f $O/earth3d_1.00.json ] || $PY experiments/phase_G_walk3d/optimize_gait3d.py --g 9.81 --speed 1.0 \
    --x0 $O/_start3d_lumbar.json --targets "$(cat $O/_targets3d_upright.json)" --random 12 --sigma 0.03 \
    --popsize 12 --maxiter 45 --workers ${WORKERS:-2} --out $O/earth3d_1.00.json
echo DONE > $O/_stage1_done
