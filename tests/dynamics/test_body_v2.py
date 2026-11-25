"""Body model v2: passive axial leg compliance (38 coordinates) and the planar
walker (19 coordinates).  Same checks as for the first model, plus the
calibration targets."""
import numpy as np
import pinocchio as pin
import pytest

from planetsense.contact.compliant import ContactParams
from planetsense.coupling.system import CoupledSystem
from planetsense.evaluation.linear import apparent_mass, fit_sdof, heavy_platform
from planetsense.human.body_v2 import NOMINAL, leg_sets, load_leg
from planetsense.human.control import PostureController, standing_posture
from planetsense.human.model import build_human, to_mjcf
from planetsense.sensor.imu import imu_ideal
from planetsense.simulation.standing import make_standing
from planetsense.simulation.walking import WalkParams, make_walker
from tolerances import TOL

mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("g", [9.81, 3.71, 1.62])
@pytest.mark.parametrize("with_deck", [False, True])
def test_static_support_with_leg_compliance(g, with_deck, modal_deck):
    leg = load_leg()
    sc = make_standing(g=g, deck=modal_deck if with_deck else None, leg=leg)
    s, y, info = sc.system, sc.y_eq, sc.info
    W = sc.human.total_mass * g
    assert s.nq == 38
    assert abs(info["sum_fn"] - W) / W < TOL["static_weight_rel"]
    assert (info["fn"] > 0).all()
    assert np.abs(info["residual_root"]).max() < 1e-8
    dy = s.rhs(0.0, y, first=True)
    assert np.abs(dy[:s.nx]).max() < TOL["equilibrium_accel"]
    # the two passive springs carry what is above the feet
    F = leg.force(info["leg_compression"], np.zeros(2)).sum()
    m_feet = sum(sc.human.segments[f"foot_{k}"].mass for k in "LR")
    assert abs(F - (W - m_feet * g)) / W < TOL["leg_axial_load_rel"]
    A, _, _ = s.linearize(y)
    assert np.linalg.eigvals(A).real.max() < 0.0, "standing equilibrium is unstable"


def test_leg_unloads_and_softens_in_low_gravity():
    """The mechanism the study is about: a stiffening leg (n > 1) is softer
    under a smaller weight; a linear leg (n = 1) is not."""
    comp, kt = {}, {}
    for g in (9.81, 3.71, 1.62):
        leg = load_leg()
        sc = make_standing(g=g, deck=None, leg=leg)
        comp[g] = sc.info["leg_compression"].mean()
        kt[g] = leg.tangent(leg.force(sc.info["leg_compression"], np.zeros(2)).mean())
    assert comp[9.81] > comp[3.71] > comp[1.62] > 0
    assert kt[9.81] > kt[3.71] > kt[1.62]
    # n = 1.5: tangent stiffness ~ load^(1/3)
    assert abs(kt[1.62] / kt[9.81] - (1.62 / 9.81) ** (1 / 3)) < 0.01
    lin = load_leg("nominal_n1_constant")
    assert abs(lin.tangent(100.0) - lin.tangent(400.0)) < 1e-9 * lin.tangent(100.0)


@pytest.mark.parametrize("name", [NOMINAL, "soft_damped_n1.5_constant", "stiff_light_n1.5_constant",
                                  "nominal_n1_constant", "nominal_n2_constant"])
def test_stored_calibration_reproduces_its_targets(name):
    sets = leg_sets()
    fr = np.linspace(0.5, 20.0, 100)
    sc = make_standing(g=9.81, deck=heavy_platform(), leg=load_leg(name))
    M = apparent_mass(sc, fr) / sc.human.total_mass
    f, z = fit_sdof(fr, M)
    assert abs(f / sets[name]["target_f"] - 1) < TOL["apparent_mass_freq_rel"]
    assert abs(z / sets[name]["target_zeta"] - 1) < TOL["apparent_mass_zeta_rel"]
    assert abs(abs(M[0]) - 1) < TOL["apparent_mass_static_rel"]


@pytest.mark.parametrize("kw,nv", [(dict(leg_axial=True), 38), (dict(sagittal=True, leg_axial=True), 19),
                                   (dict(sagittal=True), 17)])
def test_model_variants_agree_with_mujoco(kw, nv, rng):
    h = build_human(g=3.71, **kw)
    m, d = h.model, h.data
    assert h.nv == nv
    mj = mujoco.MjModel.from_xml_string(to_mjcf(h))
    md = mujoco.MjData(mj)
    assert mj.nv == nv
    assert abs(mj.body_mass.sum() - h.total_mass) < 1e-9
    assert abs(h.total_mass - 75.0) < 1e-9
    worst = 0.0
    for _ in range(20):
        q, v, tau = rng.uniform(-0.5, 0.5, nv), rng.uniform(-3, 3, nv), rng.uniform(-50, 50, nv)
        for k in "LR":                       # keep the leg coordinate small (metres)
            if kw.get("leg_axial"):
                q[h.dof(f"ankle_{k}_z")] *= 0.05
        a = pin.aba(m, d, q, v, tau)
        md.qpos[:], md.qvel[:], md.qfrc_applied[:] = q, v, tau
        mujoco.mj_forward(mj, md)
        worst = max(worst, np.abs(a - md.qacc).max() / np.abs(a).max())
        imu = imu_ideal(h, q, v, a)
        for name in h.sensor_frame:
            sa, sg = md.sensor("acc_" + name).data, md.sensor("gyr_" + name).data
            worst = max(worst, np.abs(imu[name]["f"] - sa).max() / np.abs(sa).max(),
                        np.abs(imu[name]["w"] - sg).max() / max(np.abs(sg).max(), 1e-12))
    assert worst < TOL["cross_solver_rel"]


def test_compiled_path_matches_reference_with_leg_compliance(modal_deck, rng):
    sc = make_standing(g=3.71, deck=modal_deck, leg=load_leg())
    s, y = sc.system, sc.y_eq
    F = rng.normal(0, 100, 9)
    nq = s.nq
    scale = np.concatenate([np.full(nq, 2e-3), np.full(nq, 5e-2), np.full(s.nm, 1e-4),
                            np.full(s.nm, 1e-2), np.full(s.nc, 1e-3)])
    worst = 0.0
    for trial in range(40):
        yp = y.copy()
        yp[:s.nx] += rng.normal(0, 1, s.nx) * scale
        if trial % 3 == 0:
            yp[2] += 0.006
        first = bool(trial % 2)
        a0 = s.anchor.copy()
        s.use_kernel = False
        d_ref = s.rhs(0.0, yp, F, first=first)
        a_ref, s.anchor = s.anchor.copy(), a0.copy()
        s.use_kernel = True
        d_ker = s.rhs(0.0, yp, F, first=first)
        a_ker, s.anchor = s.anchor.copy(), a0.copy()
        worst = max(worst, np.abs(d_ref - d_ker).max() / np.abs(d_ref).max(), np.abs(a_ref - a_ker).max())
    assert worst < TOL["kernel_rel"]


@pytest.mark.parametrize("g", [9.81, 1.62])
def test_energy_passive_drop_with_leg_springs(g, modal_deck):
    """Passive body with leg springs dropped on the flexible deck: the
    work-energy identity holds and the total energy (including what the leg
    springs, joint springs and contact pads store) never increases."""
    h = build_human(g=g, leg_axial=True)
    leg = load_leg()
    q = standing_posture(h)
    nr = h.n_root
    kp, kd = np.full(h.nv - nr, 300.0), np.full(h.nv - nr, 8.0)
    li = (h.dof("ankle_L_z"), h.dof("ankle_R_z"))
    for i in li:                               # the leg coordinate has its own spring
        kp[i - nr] = kd[i - nr] = 0.0
    ctrl = PostureController(q_ref=q.copy(), kp=kp, kd=kd, tau_ff=np.zeros(h.nv - nr), meta={},
                             leg_idx=li, leg=leg)
    s = CoupledSystem(h, ContactParams(), ctrl, deck=modal_deck)
    q0 = q.copy()
    q0[0], q0[1] = 7.0, 0.575
    q0[2] += 0.02
    y = s.pack(q0, np.zeros(h.nv))
    s.reset_anchors(y)
    out = s.simulate(y, T=0.6, dt=1e-4, out_dt=1e-3, record_q=True, fall_height=None)
    E = out["energy"]
    c = {k: i for i, k in enumerate(out["energy_cols"])}
    assert out["fn"].max() > 0 and out["fn"].min() >= 0.0
    through = (np.abs(np.diff(E[:, c["W_act"]])).sum() + np.abs(np.diff(E[:, c["W_contact"]])).sum()
               + E[-1, c["D_deck"]])
    drift = np.abs(E[:, c["balance"]] - E[0, c["balance"]]).max()
    assert drift / through < TOL["energy_balance_rel"], (drift, through)
    Esp = 0.5 * np.sum(kp * (out["q"][:, nr:] - q[nr:]) ** 2, axis=1)
    Eleg = np.array([leg.energy(qq[list(li)]) for qq in out["q"]])
    total = E[:, c["T_h"]] + E[:, c["U_h"]] + E[:, c["E_deck"]] + E[:, c["E_contact"]] + Esp + Eleg
    scale = total[0] - total.min()
    assert np.diff(total).max() < TOL["energy_creation_rel"] * scale, np.diff(total).max()
    assert total[-1] < total[0]


@pytest.mark.parametrize("g", [9.81, 1.62])
def test_walker_mechanics_are_consistent(g, modal_deck):
    """The planar walker on the flexible deck: contact never pulls, the
    centre of mass obeys m a = sum f + m g at every sampled state, and the
    work-energy identity holds while it walks (no energy is created by the
    coupling, whatever the controller does)."""
    p = WalkParams().similarity_scaled(g)
    s, y = make_walker(g, p, load_leg(), deck=modal_deck, x0=4.0, y_walk=0.575, v0=0.5)
    h = s.h
    m_tot = h.total_mass
    dt, t, worst = 2.5e-4, 0.0, 0.0
    E0 = s.energy(y)
    through, last, lateral = 0.0, np.zeros(4), 0.0
    for i in range(int(1.5 / dt)):
        if i % 100 == 0:
            k1 = s.rhs(t, y, None, first=True)
            q, v, a = y[:s.nq], y[s.nq:2 * s.nq], k1[s.nq:2 * s.nq]
            assert s._fn.min() >= 0.0
            pin.centerOfMass(h.model, h.data, q, v, a)
            lhs = m_tot * h.data.acom[0]
            rhs = s._f.sum(axis=0) + m_tot * np.array([0, 0, -g])
            # in-plane components only: the planar walker is held in its plane by a
            # constraint, which reacts the (small) lateral friction force from deck torsion
            worst = max(worst, np.abs((lhs - rhs)[[0, 2]]).max() / (m_tot * g))
            lateral = max(lateral, abs((lhs - rhs)[1]) / (m_tot * g))
            W = y[-4:]
            through += np.abs(W - last).sum()
            last = W.copy()
        y, _ = s.step(t, y, dt)
        t += dt
        if y[s.iz] < 0.5:
            break
    assert worst < TOL["walk_centroidal_rel"]
    assert lateral < 0.01                 # constraint force stays below 1 % of the weight
    drift = abs(s.energy(y)["balance"] - E0["balance"])
    assert drift / through < TOL["walk_energy_balance_rel"], (drift, through)
