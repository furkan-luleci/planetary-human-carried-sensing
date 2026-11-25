import numpy as np

from planetsense.excitation.multisine import make_multisine
from planetsense.simulation.standing import make_standing
from tolerances import TOL


def test_time_step_and_mesh_convergence(modal_deck):
    """Brief 17.5 / 8.3: time-step convergence of the wearable-sensor output
    for a standing person on the excited deck.  (Mesh and modal-truncation
    convergence of the structure are in tests/unit/test_deck.py.)

    The same continuous-time multisine drives every run.  Errors are measured
    against dt = 0.05 ms."""
    ms = make_multisine(9, T=8.0, rms=80.0, seed=5)
    res = {}
    for dt in (4e-4, 2e-4, 1e-4, 5e-5):
        sc = make_standing(g=9.81, deck=modal_deck)
        res[dt] = sc.system.simulate(sc.y_eq.copy(), T=0.8, dt=dt, out_dt=2e-3, forcing=ms)
    ref = res[5e-5]
    err = {}
    for dt in (4e-4, 2e-4, 1e-4):
        e = []
        for name in ("pelvis_L", "wrist_L"):
            k = ref["sensors"].index(name)
            x, xr = res[dt]["f"][:, k], ref["f"][:, k]
            dyn = xr - xr[0]
            e.append(np.sqrt(np.mean((x - xr) ** 2)) / np.sqrt(np.mean(dyn ** 2)))
        err[dt] = max(e)
    assert err[2e-4] < TOL["dt_error_rel"], err
    assert err[4e-4] > err[2e-4] > err[1e-4]
    order = np.log2(err[4e-4] / err[2e-4])
    assert order > TOL["dt_min_order"], (order, err)
