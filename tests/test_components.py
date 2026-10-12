import importlib

import numpy as np
import pytest

from neural_pointers import activations
from neural_pointers.difference_computation.model import DifferenceComputationModel
from neural_pointers.pointers.inputs import PointerInputs
from neural_pointers.pointers.model import DOMAIN_MODE, SynapticPointerModel
from neural_pointers.simulation import simulate_stages
from tests.conftest import load_config

# Short protocols (Hydra override syntax) to exercise storing and plotting quickly.
SHORT_PROTOCOLS = {
    "pointers": "[{name:binding0,mode:BINDING,duration:1},{name:reset,mode:BINDING,duration:1},"
    "{name:WTA0,mode:WTA,duration:1}]",
    "competitive_queueing": "[{name:numbed,duration:1},{name:aroused,duration:1},"
    "{name:numbed,duration:1}]",
    "difference_computation": "[{name:pattern_zeros,duration:1},{name:pattern0,duration:1}]",
}


@pytest.mark.parametrize(
    "fn",
    [
        activations.relu,
        activations.square,
        lambda h: activations.threshold_linear(h, 0.5),
        lambda h: activations.step(h, 0.5),
    ],
)
def test_activations_do_not_modify_input(fn):
    h = np.array([-1.0, 0.0, 0.5, 2.0])
    h_copy = h.copy()
    fn(h)
    np.testing.assert_array_equal(h, h_copy)


def test_simulate_stages_shares_boundary_time_points():
    stages = [type("Stage", (), {"name": "a", "duration": 1})] * 2
    results = simulate_stages(lambda h, t: -h, lambda name: (), stages, 0.1, np.ones(1))
    assert results[1].t[0] == results[0].t[-1]
    np.testing.assert_array_equal(results[1].h[0], results[0].h[-1])


def test_pointer_model_rejects_invalid_mode():
    model = SynapticPointerModel(3, load_config("pointers").model)
    with pytest.raises(TypeError):
        model.set_mode("WTA")
    model.set_mode(DOMAIN_MODE.WTA)


def test_pointer_inputs_reject_unknown_stage():
    inputs = PointerInputs(3, np.random.default_rng(0))
    assert len(inputs.get_input("binding2")) == 3
    with pytest.raises(ValueError):
        inputs.get_input("binding")


def test_pointer_inputs_single_item():
    inputs = PointerInputs(1, np.random.default_rng(0))
    np.testing.assert_array_equal(inputs.pointers[0], [1.0])


def test_difference_computation_requires_single_neuron():
    with pytest.raises(ValueError):
        DifferenceComputationModel(2, load_config("difference_computation").model)


@pytest.mark.parametrize("experiment", sorted(SHORT_PROTOCOLS))
def test_experiment_stores_results(experiment, tmp_path):
    cfg = load_config(experiment, ["simulation.stages=" + SHORT_PROTOCOLS[experiment]])
    module = importlib.import_module(f"neural_pointers.{experiment}.experiment")
    module.run_experiment(cfg, out_dir=tmp_path)
    data = np.load(tmp_path / "results.npz")
    assert data["h"].shape[0] == data["t"].shape[0] == data["stage_lengths"].sum()
    assert len(data["stage_names"]) == len(cfg.simulation.stages)
    assert any(tmp_path.glob("*.png"))


def test_repeated_stages_keep_their_figures(tmp_path):
    cfg = load_config(
        "competitive_queueing",
        ["simulation.stages=" + SHORT_PROTOCOLS["competitive_queueing"]],
    )
    module = importlib.import_module("neural_pointers.competitive_queueing.experiment")
    module.run_experiment(cfg, out_dir=tmp_path)
    assert (tmp_path / "stage_00_numbed_final_h_neur_x.png").exists()
    assert (tmp_path / "stage_02_numbed_final_h_neur_x.png").exists()
