"""Shared test setup. pytest finds this file automatically."""
import json
from pathlib import Path

import numpy as np
import pytest

from surrogate.bundle import load_model_bundle
from surrogate.preprocess import preprocess

FIXTURES = Path(__file__).parent / "fixtures"
BRACKET = "448"


@pytest.fixture(scope="session")
def bundle():
    return load_model_bundle("v2")


@pytest.fixture(scope="session")
def raw_files():
    return FIXTURES / f"{BRACKET}.vtk", FIXTURES / f"{BRACKET}.fem"


@pytest.fixture(scope="session")
def processed(raw_files):
    return preprocess(*raw_files)            # (graph, features), built once for all tests


@pytest.fixture(scope="session")
def training_features():
    return np.load(FIXTURES / f"{BRACKET}_features.npz")


@pytest.fixture(scope="session")
def expected():
    return json.loads((FIXTURES / f"{BRACKET}_expected.json").read_text())