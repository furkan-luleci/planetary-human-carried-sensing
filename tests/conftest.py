import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from planetsense.structure.spine_beam import SpineBeamDeck  # noqa: E402

GRAVITIES = {"Earth": 9.81, "Mars": 3.71, "Moon": 1.62}


@pytest.fixture(scope="session")
def fe_deck():
    return SpineBeamDeck()


@pytest.fixture(scope="session")
def modal_deck(fe_deck):
    return fe_deck.modal(f_max=150.0)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20261118)
