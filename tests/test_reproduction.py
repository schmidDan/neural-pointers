"""The default configurations reproduce the simulations of the BonnBrain 2024 poster."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.conftest import DATA, load_config
from tests.reference import simulate

# Tolerances cover numerical differences between platforms and library versions. The difference computation model is
# sensitive to the internal steps of the ODE solver (see docs/known_issues.md) and therefore only checked coarsely.
TOLERANCES = {
    "pointers": dict(rtol=1e-6, atol=1e-6),
    "competitive_queueing": dict(rtol=1e-5, atol=1e-5),
    "difference_computation": dict(rtol=0.0, atol=0.15),
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
    reference = np.load(DATA / "difference_computation_reference.npz")
    np.testing.assert_allclose(
        simulations("difference_computation")["convergence_times"],
        reference["convergence_times"],
        rtol=0.0,
        atol=0.5,
    )
