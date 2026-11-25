#!/bin/bash
# Stage 2 (2026-11-20, for consistency across planets): 3-D walkers on
# Mars and the Moon at 1.0 m/s.  Targets: the forward targets of the side-view Mars and Moon gaits
# (speed, step rate, duty factor, peak single-foot force) and the sideways targets and trunk bound of
# the Earth 3-D walker (no unsuited reduced-gravity step-width data exist: ASSUMED).  Start: the side-view
# gait's forward parameters with the Earth 3-D walker's sideways parameters (outputs/gait/_start3d_<planet>.json);
# each evaluation starts on that planet's side-view gait cycle.  Joint gains scaled for Earth gravity, as
# for the side-view Mars and Moon gaits.  Resumable.
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-python}
export OMP_NUM_THREADS=1
O=outputs/gait
run() {  # planet g out workers random sigma maxiter
  [ -f $O/$3 ] && { echo "skip $3"; return; }
  $PY experiments/phase_G_walk3d/optimize_gait3d.py --g $2 --speed 1.0 --no-similarity --g-gain 9.81 \
      --seed-gait $O/$1_1.00.json --seed-npz $O/_seed_planar_$1_1.00.npz --x0 ${START:-$O/_start3d_$1.json} \
      --targets "$(cat $O/_targets3d_$1.json)" --random $5 --sigma $6 --popsize 12 --maxiter $7 --workers $4 --out $O/$3
}
# Moon by continuation in gravity (the direct start fell sideways after the start-up aid in every variant
# tried: Earth sideways gains as they are, the sideways velocity gain scaled by sqrt(g_E/g), sideways joint
# stiffness scaled by g/g_E; see docs/phase_reports/PHASE_G_walker3d.md): first an intermediate gravity
# (2.6 m/s^2, targets between Mars and Moon, Mars side-view seed state) starting from the Mars 3-D gait,
# then the Moon starting from that.
moonstep() {  # g target-file start out seed-planet workers random sigma maxiter
  [ -f $O/$4 ] && { echo "skip $4"; return; }
  $PY experiments/phase_G_walk3d/optimize_gait3d.py --g $1 --speed 1.0 --no-similarity --g-gain 9.81 \
      --seed-gait $O/$5_1.00.json --seed-npz $O/_seed_planar_$5_1.00.npz --x0 $3 \
      --targets "$(cat $O/$2)" --random $7 --sigma $8 --popsize 12 --maxiter $9 --workers $6 --out $O/$4
}
case "$1" in
  mars) run mars 3.71 mars3d_1.00.json ${WORKERS:-2} 12 0.04 45 ;;
  moon) run moon 1.62 moon3d_1.00.json ${WORKERS:-2} 24 0.05 60 ;;
  moon_from_mars)   # tried first: the Mars 3-D gait at generation 15 walked 9 s at lunar gravity, the finished one fell
    moonstep 1.62 _targets3d_moon.json $O/mars3d_1.00.json moon3d_1.00.json moon ${WORKERS:-4} 12 0.03 45 ;;
  moon_cont21)      # used: the finished Mars 3-D gait fell at lunar gravity but walked 9 s at 2.1 m/s^2
    moonstep 2.10 _targets3d_moonmid.json $O/mars3d_1.00.json _mid3d_2.10.json mars ${WORKERS_A:-2} 12 0.03 15
    moonstep 1.62 _targets3d_moon.json $O/_mid3d_2.10.json moon3d_1.00.json moon ${WORKERS_B:-4} 12 0.03 45 ;;
  moon_cont)        # fallback through an intermediate gravity
    moonstep 2.60 _targets3d_moonmid.json $O/mars3d_1.00.json _mid3d_2.60.json mars ${WORKERS:-4} 12 0.03 20
    moonstep 1.62 _targets3d_moon.json $O/_mid3d_2.60.json moon3d_1.00.json moon ${WORKERS:-4} 12 0.03 45 ;;
esac
