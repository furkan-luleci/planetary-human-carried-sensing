"""Phase K: the structure registry leaves S1 untouched and places S2's settings by the declared rules."""
import numpy as np

from planetsense.simulation.records import DECK_REF_XY, analysis_window, deck_ref_xy
from planetsense.structure import registry as SR
from planetsense.structure.spine_beam import SpineBeamDeck


def test_s1_unchanged():
    md = SR.modal_deck("S1")
    ref = SpineBeamDeck().modal(f_max=150.0)
    np.testing.assert_array_equal(md.omega, ref.omega)
    np.testing.assert_array_equal(md.nodal, ref.nodal)
    assert deck_ref_xy(md) is DECK_REF_XY
    st = SR.get("S1")
    assert (st.stand_xy, st.line_y, st.shaker_rms, st.floor_rms) == ((4.2, 0.575), 0.575, 800.0, 5.0)


def test_s2_settings_follow_the_rules():
    s1 = SR.modal_deck("S1")
    target = SR.empty_rms(s1, 4.2, 0.575, 800.0, 5.0)
    for name in ("S2", "S2L"):
        md, st = SR.modal_deck(name), SR.get(name)
        assert abs(SR.empty_rms(md, *st.stand_xy, st.shaker_rms, st.floor_rms) / target - 1) < 1e-9
        xs = np.linspace(0, md.params.L, 801)
        Sz = md.surface_modes(xs, np.zeros_like(xs))[:, 2, :]
        at = np.abs(md.surface_modes([st.stand_xy[0]], [st.stand_xy[1]])[0, 2, :])
        for k, f in enumerate(md.freq_hz):
            if f < 30.0:
                assert at[k] > 0.3 * np.abs(Sz[:, k]).max()          # no node at the standing point


def test_window_uses_the_record_length():
    t = np.arange(0, 20, 0.005)
    rec = dict(t=t, activity="walking", com=np.c_[-2.0 + t, np.zeros_like(t), np.ones_like(t)])
    m14 = analysis_window(dict(rec))                       # S1 records carry no length: 14 m
    m8 = analysis_window(dict(rec, deck_L=8.2))
    assert abs(t[m14][-1] - 16.0) < 0.01 and abs(t[m8][-1] - 10.2) < 0.01
