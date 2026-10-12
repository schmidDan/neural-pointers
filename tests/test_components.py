import importlib

import numpy as np
import pytest

from neural_pointers import activations
from neural_pointers.monitoring.model import MonitoringModel
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
    "monitoring": "[{name:pattern_zeros,duration:1},{name:pattern0,duration:1}]",
}


@pytest.mark.parametrize(
    "fn",
    [
        activations.relu,
        activations.square,
        lambda h: activations.threshold_linear(h, 0.5),
        lambda h: activations.logistic(h, 0.5, 0.1),
    ],
)
def test_activations_do_not_modify_input(fn):
    h = np.array([-1.0, 0.0, 0.5, 2.0])
    h_copy = h.copy()
    fn(h)
    np.testing.assert_array_equal(h, h_copy)


def test_simulate_stages_contiguous_time_grid():
    stages = [type("Stage", (), {"name": "a", "duration": d}) for d in (1, 0.5, 2)]
    results = simulate_stages(lambda h, t: -h, lambda name: (), stages, 0.1, np.ones(1))
    t = np.concatenate([r.t for r in results])
    h = np.concatenate([r.h for r in results])[:, 0]
    np.testing.assert_allclose(t, np.arange(35) * 0.1, atol=1e-12)
    assert [r.t[0] for r in results] == pytest.approx([0.0, 1.0, 1.5])
    # continuity across stage boundaries: the solution of dh/dt = -h is exp(-t)
    np.testing.assert_allclose(h, np.exp(-t), rtol=1e-5)


def test_simulate_stages_rejects_incompatible_duration():
    stages = [type("Stage", (), {"name": "a", "duration": 0.25})]
    with pytest.raises(ValueError):
        simulate_stages(lambda h, t: -h, lambda name: (), stages, 0.1, np.ones(1))


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


def test_monitoring_is_time_invariant_and_stateless():
    model = MonitoringModel(1, load_config("monitoring").model)
    h = np.random.default_rng(0).uniform(0, 1, model.h_0.shape)
    inp = np.full(7, 0.5)
    d_late = model(h, 500.0, inp)
    d_early = model(h, 0.0, inp)
    np.testing.assert_array_equal(d_late, d_early)
    np.testing.assert_array_equal(model(h, 500.0, inp), d_late)


def test_monitoring_supports_several_neurons():
    cfg = load_config(
        "monitoring",
        [
            "n_neurons=2",
            "simulation.stages=[{name:pattern_zeros,duration:30},{name:pattern0,duration:150}]",
        ],
    )
    model, results, convergence = importlib.import_module(
        "neural_pointers.monitoring.experiment"
    ).run_experiment(cfg)
    assert results[0].h.shape[1] == len(model.h_0)
    assert np.all(np.isfinite(convergence[:, 1]))


def test_repeated_stages_keep_their_figures(tmp_path):
    cfg = load_config(
        "competitive_queueing",
        ["simulation.stages=" + SHORT_PROTOCOLS["competitive_queueing"]],
    )
    module = importlib.import_module("neural_pointers.competitive_queueing.experiment")
    module.run_experiment(cfg, out_dir=tmp_path)
    assert (tmp_path / "stage_00_numbed_final_h_neur_x.png").exists()
    assert (tmp_path / "stage_02_numbed_final_h_neur_x.png").exists()


def test_monitoring_weights_are_row_normalized():
    model = MonitoringModel(1, load_config("monitoring").model)
    np.testing.assert_allclose(model._weights_exc.sum(axis=1), 1.0)
    np.testing.assert_allclose(model._weights_inh.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "variant",
    [
        "model.variant.excitation=additive",
        "model.variant.inhibition=subtractive",
        "model.variant.pointer_weights=offset",
        "model.variant.pointer_dynamics=false",
    ],
)
def test_pointer_variants_run(variant):
    cfg = load_config(
        "pointers", [variant, "simulation.stages=" + SHORT_PROTOCOLS["pointers"]]
    )
    model, results = importlib.import_module(
        "neural_pointers.pointers.experiment"
    ).run_experiment(cfg)
    n = cfg.n_neurons
    expected = n * n + n + (n if model.has_pointer_dynamics else 0)
    assert len(model.h_0) == expected
    assert all(np.all(np.isfinite(r.h)) for r in results)


def test_pointer_model_rejects_unknown_variant():
    cfg = load_config("pointers", ["model.variant.excitation=multiplicative"])
    with pytest.raises(ValueError):
        SynapticPointerModel(3, cfg.model)


@pytest.mark.parametrize("preset", ["default", "poster_simulation"])
def test_pointer_presets_read_out_all_items(preset):
    from neural_pointers.summary import summarize_pointers

    cfg = load_config("pointers", ["model=" + preset])
    model, results = importlib.import_module(
        "neural_pointers.pointers.experiment"
    ).run_experiment(cfg)
    summary = summarize_pointers(model, results)
    assert summary["readout_correct"]
    if preset == "default":
        # shunting excitation and inhibition bound the potentials to [-eta / gamma, beta]
        assert -10.0 <= summary["potential_min"] and summary["potential_max"] <= 1.0


def test_coupled_simulation_executes_all_operations():
    cfg = load_config("coupled")
    _, _, summary = importlib.import_module(
        "neural_pointers.coupled.experiment"
    ).run_experiment(cfg)
    order = list(range(cfg.n_operations))
    assert summary["selected"] == order
    assert summary["retrieved"] == order
    # every operation converges (after the queue is exhausted, the idle loop keeps signaling)
    assert np.all(np.isfinite(summary["execution_durations"]))
    assert len(summary["convergence_signals"]) >= cfg.n_operations


# a short three-operation program, used to check that durations belong to the operations
_THREE_OPS = [
    "n_operations=3",
    "simulation.stages=[{name:reset,mode:BINDING,duration:2},"
    "{name:binding0,mode:BINDING,duration:5},{name:reset,mode:BINDING,duration:2},"
    "{name:binding1,mode:BINDING,duration:5},{name:reset,mode:BINDING,duration:2},"
    "{name:binding2,mode:BINDING,duration:5},{name:reset,mode:BINDING,duration:2},"
    "{name:go,mode:WTA,duration:10},{name:execute,mode:WTA,duration:450}]",
]


def test_execution_durations_follow_the_operation():
    run = importlib.import_module("neural_pointers.coupled.experiment").run_experiment
    durations = {}
    for assignment in ([0, 1, 2], [2, 0, 1]):
        cfg = load_config(
            "coupled",
            _THREE_OPS + ["coupling.processing_patterns=" + str(assignment)],
        )
        _, _, summary = run(cfg)
        assert summary["retrieved"] == [0, 1, 2]
        for position, pattern in enumerate(assignment):
            durations.setdefault(pattern, []).append(
                summary["execution_durations"][position]
            )
    # the duration belongs to the processing pattern, wherever it occurs in the schedule
    for pattern, values in durations.items():
        np.testing.assert_allclose(values[0], values[1], rtol=0.15)


def test_poster_simulation_stalls_in_the_closed_loop():
    run = importlib.import_module("neural_pointers.coupled.experiment").run_experiment
    stages = (
        "[{name:reset,mode:BINDING,duration:2},"
        "{name:binding0,mode:BINDING,duration:5},{name:reset,mode:BINDING,duration:2},"
        "{name:binding1,mode:BINDING,duration:5},{name:reset,mode:BINDING,duration:2},"
        "{name:binding2,mode:BINDING,duration:5},{name:reset,mode:BINDING,duration:2},"
        "{name:go,mode:WTA,duration:10},{name:execute,mode:WTA,duration:150}]"
    )
    cfg = load_config(
        "coupled",
        [
            "pointers/model@models.pointers=poster_simulation",
            "n_operations=3",
            "simulation.stages=" + stages,
        ],
    )
    _, _, summary = run(cfg)
    # the unbounded retrieval saturates the monitoring input above the prediction
    # ceiling, so convergence is never signaled (see docs/poster_differences.md)
    assert summary["retrieved"] == [0]
    assert np.all(np.isnan(summary["execution_durations"]))
