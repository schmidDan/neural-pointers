"""The default simulations match the stored reference trajectories."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.conftest import DATA, load_config
from tests.reference import simulate

# Tolerances cover numerical differences between platforms and library versions.
TOLERANCES = {
    "pointers": dict(rtol=1e-6, atol=1e-6),
    "competitive_queueing": dict(rtol=1e-5, atol=1e-5),
    "monitoring": dict(rtol=1e-5, atol=1e-5),
    # The thresholds of the closed loop amplify small numerical differences (up to 3e-3 between Python 3.10 and 3.12);
    # its discrete outcome is checked exactly in test_coupled_execution_matches_reference.
    "coupled": dict(rtol=0.0, atol=1e-2),
}


@pytest.fixture(scope="module")
def simulations():
    cache = {}

    def get(experiment):
        if experiment not in cache:
            cache[experiment] = simulate(experiment, load_config(experiment))
        return cache[experiment]

    return get


def test_config_matches_reference(experiment):
    cfg = OmegaConf.to_container(load_config(experiment), resolve=True)
    reference = json.loads((DATA / f"{experiment}_config.json").read_text())
    assert cfg == reference


def test_time_grid_matches_reference(experiment, simulations):
    reference = np.load(DATA / f"{experiment}_reference.npz")
    data = simulations(experiment)
    assert data["n_samples"] == reference["n_samples"]
    np.testing.assert_array_equal(data["t"], reference["t"])


@pytest.mark.parametrize("key", ["h", "stage_end_states"])
def test_trajectories_match_reference(experiment, simulations, key):
    reference = np.load(DATA / f"{experiment}_reference.npz")
    np.testing.assert_allclose(
        simulations(experiment)[key], reference[key], **TOLERANCES[experiment]
    )


def test_convergence_times_match_reference(simulations):
    reference = np.load(DATA / "monitoring_reference.npz")
    np.testing.assert_allclose(
        simulations("monitoring")["convergence_times"],
        reference["convergence_times"],
        rtol=0.0,
        atol=0.011,  # one sample
    )


def test_coupled_execution_matches_reference(simulations):
    reference = np.load(DATA / "coupled_reference.npz")
    data = simulations("coupled")
    np.testing.assert_array_equal(data["selected"], reference["selected"])
    np.testing.assert_array_equal(data["retrieved"], reference["retrieved"])
    np.testing.assert_allclose(
        data["convergence_signals"],
        reference["convergence_signals"],
        rtol=0.0,
        atol=0.011,  # one sample
    )
    np.testing.assert_allclose(
        data["execution_durations"],
        reference["execution_durations"],
        rtol=0.0,
        atol=0.1,
    )
