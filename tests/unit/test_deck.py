import inspect

import numpy as np

from planetsense.structure import spine_beam as sb
from tolerances import TOL

TABLE3 = {"B1": 1.8023, "B2": 7.2091, "T1": 12.3022, "B3": 16.2204, "L1": 22.2735,
          "T2": 24.5853, "B4": 28.8361, "T3": 36.8306, "B5": 45.0559, "T4": 49.0191}


def test_deck_modes_match_earlier_study_table3(fe_deck):
    f, lab = fe_deck.frequencies(50.0)
    assert list(lab) == list(TABLE3)
    for fi, l in zip(f, lab):
        assert abs(fi - TABLE3[l]) < TOL["deck_freq_table3_hz"], (l, fi)


def test_deck_matrices_symmetric_definite(fe_deck):
    K = fe_deck.K
    assert np.abs(K - K.T).max() <= 1e-9 * np.abs(K).max()
    assert np.linalg.eigvalsh(0.5 * (K + K.T)).min() > 0
    assert (fe_deck.Mdiag >= 0).all() and (fe_deck.Mdiag > 0).sum() == 157
    md = fe_deck.modal(f_max=50.0)
    assert (md.zeta > 0).all()
    assert abs(md.zeta[0] - 0.02) < 1e-12 and abs(md.zeta[1] - 0.02) < 1e-12


def test_deck_static_flexibility_closed_form(fe_deck):
    md = fe_deck.modal()                       # all modes == full FE model
    p, s = fe_deck.params, fe_deck.sec
    S = md.surface_modes([7.0, 7.0], [0.0, 0.575])
    flex = np.einsum("pam,m,pbm->pab", S, 1 / md.omega ** 2, S)
    bend = p.L ** 3 / (48 * p.E * s["I"])
    tors = 0.575 ** 2 * (7.0 * 7.0 / (p.L * s["G"] * s["J"]))
    assert abs(flex[0, 2, 2] / bend - 1) < TOL["deck_static_rel"]
    assert abs(flex[1, 2, 2] / (bend + tors) - 1) < TOL["deck_static_rel"]


def test_surface_matrix_is_work_conjugate(fe_deck, rng):
    """Displacement read-out and load mapping use the same matrix, so the
    flexibility matrix between any set of surface points is symmetric."""
    md = fe_deck.modal(f_max=150.0)
    x, y = rng.uniform(0.5, 13.5, 6), rng.uniform(-1.4, 1.4, 6)
    S = md.surface_modes(x, y).reshape(-1, md.nm)
    F = S @ np.diag(1 / md.omega ** 2) @ S.T
    assert np.abs(F - F.T).max() <= 1e-12 * np.abs(F).max()
    assert np.linalg.eigvalsh(F).min() > -1e-12 * np.abs(F).max()


def test_deck_unoccupied_modes_gravity_invariant(fe_deck):
    """Brief 17.5.  Gravity is not an input of the structural model at all, and
    the coupled system without a person returns the bare-deck eigenvalues for
    every gravity."""
    for obj in (sb.SpineBeamDeck.__init__, sb.section_properties, sb.DeckParams):
        assert "g" not in inspect.signature(obj).parameters
    from planetsense.contact.compliant import ContactParams
    from planetsense.coupling.system import CoupledSystem
    from planetsense.human.model import build_human
    md = fe_deck.modal(f_max=50.0)
    ref = None
    for g in (9.81, 3.71, 1.62):
        h = build_human(g=g)
        sys = CoupledSystem(h, ContactParams(), lambda t, q, v: np.zeros(36), deck=md,
                            contact_enabled=False)
        y = sys.pack(h.neutral(), np.zeros(36))
        A, _, _ = sys.linearize(y)
        i0 = 2 * sys.nq
        Ad = A[i0:i0 + 2 * md.nm, i0:i0 + 2 * md.nm]
        ev = np.linalg.eigvals(Ad)
        f = np.sort(np.abs(ev[ev.imag > 0])) / (2 * np.pi)
        fd = md.freq_hz
        assert np.allclose(f, fd, rtol=1e-6, atol=0)   # damped poles |lambda| = omega_n
        if ref is None:
            ref = Ad.copy()
        assert np.array_equal(Ad, ref)                 # bitwise identical for every g
        # no coupling between the falling body and the deck when there is no contact
        assert np.abs(A[i0:i0 + 2 * md.nm, :i0]).max() == 0.0


def test_mesh_convergence_of_deck(fe_deck):
    fine = sb.SpineBeamDeck(sb.DeckParams(n_elem=80))
    f40, l40 = fe_deck.frequencies(50.0)
    f80, l80 = fine.frequencies(50.0)
    assert list(l40) == list(l80)
    for a, b, l in zip(f40, f80, l40):
        tol = TOL["mesh_torsion_rel"] if l[0] == "T" else TOL["mesh_bending_rel"]
        assert abs(a / b - 1) < tol, (l, a, b)
    # contact-location static response
    for deck in (fe_deck, fine):
        md = deck.modal()
        S = md.surface_modes([7.0], [0.575])[0]
        deck._flex = float(S[2] @ (S[2] / md.omega ** 2))
    assert abs(fe_deck._flex / fine._flex - 1) < 1e-3


def test_modal_truncation_of_contact_flexibility(fe_deck):
    full = fe_deck.modal()
    cut = fe_deck.modal(f_max=150.0)
    fl = []
    for md in (full, cut):
        S = md.surface_modes([7.0], [0.575])[0]
        fl.append(float(S[2] @ (S[2] / md.omega ** 2)))
    assert abs(fl[1] / fl[0] - 1) < TOL["modal_truncation_static_rel"]
