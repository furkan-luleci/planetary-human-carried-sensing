import numpy as np

from planetsense.excitation.multisine import make_multisine
from planetsense.simulation.standing import make_standing
from planetsense.structure.spine_beam import DeckParams, SpineBeamDeck
from tolerances import TOL


def test_rigid_floor_control_vs_flexible_floor(fe_deck, modal_deck):
    """Brief 17.5 / 8.4.
    (a) A deck with no modes is the rigid floor (same code path, same numbers).
    (b) The flexible-floor equilibrium tends to the rigid one as the deck
        stiffens, with a difference proportional to 1/E.
    (c) Under the same shaker forces the matched rigid-floor twin shows no
        structural vibration at the sensors, the flexible deck does."""
    rigid = make_standing(g=9.81, deck=None)
    zero = make_standing(g=9.81, deck=fe_deck.modal(n_modes=0))
    assert np.abs(rigid.y_eq[:72] - zero.y_eq[:72]).max() < TOL["rigid_limit_rel"]
    d0 = rigid.system.rhs(0.0, rigid.y_eq)[:72]
    d1 = zero.system.rhs(0.0, zero.y_eq)[:72]
    assert np.abs(d0 - d1).max() < 1e-9

    dz = []
    for factor in (1.0, 10.0, 100.0):
        deck = SpineBeamDeck(DeckParams(E=30e9 * factor)).modal(n_modes=24)
        sc = make_standing(g=9.81, deck=deck)
        dz.append(rigid.y_eq[2] - sc.y_eq[2])              # extra sag = deck deflection
    assert dz[0] > dz[1] > dz[2] > 0
    assert abs(dz[0] / dz[1] - 10) < 0.05 and abs(dz[1] / dz[2] - 10) < 0.05

    ms = make_multisine(9, T=8.0, rms=80.0, seed=3)
    flex = make_standing(g=9.81, deck=modal_deck)
    of = flex.system.simulate(flex.y_eq.copy(), T=1.0, dt=2.5e-4, out_dt=1e-3, forcing=ms)
    orr = rigid.system.simulate(rigid.y_eq.copy(), T=1.0, dt=2.5e-4, out_dt=1e-3, forcing=ms)
    k = of["sensors"].index("pelvis_L")
    assert of["f"][:, k, 2].std() > 1e-3                    # structure-borne vibration reaches the pelvis
    assert orr["f"][:, k, 2].std() < 1e-9                   # nothing to transmit on the rigid twin
    assert np.abs(of["a_foot"]).max() > 1e-2
