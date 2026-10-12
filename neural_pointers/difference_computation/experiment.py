"""Difference computation (progress monitoring) experiment."""

import logging
from pathlib import Path

import numpy as np

from neural_pointers.difference_computation.inputs import DifferenceComputationInputs
from neural_pointers.difference_computation.model import (
    DifferenceComputationModel,
    convergence_time,
)
from neural_pointers.difference_computation.visualization import (
    visualize_stage,
    visualize_trial,
)
from neural_pointers.simulation import (
    concatenate_stages,
    save_results,
    simulate_stages,
    stage_label,
)

log = logging.getLogger(__name__)


def run_experiment(cfg, out_dir=None):
    """Simulates the difference computation model for the stages defined in `cfg.simulation`.

    Args:
        cfg: Experiment config with `seed`, `n_neurons`, `plot`, `model` and `simulation`.
        out_dir: Directory to store results and figures in. Nothing is stored if `None`.

    Returns:
        tuple: The model, the list of `StageResult`s and an array with one row `(onset, convergence time, duration)`
        per stage.
    """
    rng = np.random.default_rng(cfg.seed)
    model = DifferenceComputationModel(cfg.n_neurons, cfg.model)
    inputs = DifferenceComputationInputs(
        cfg.n_neurons * cfg.model.n_inputs_per_neuron, rng
    )

    results = simulate_stages(
        model, inputs.get_input, cfg.simulation.stages, cfg.simulation.dt, model.h_0
    )
    convergence_times = np.array([convergence_time(model, r.h, r.t) for r in results])
    for r, (_, t_conv, t_dur) in zip(results, convergence_times):
        log.info("%s: convergence at t=%.2f (after %.2f)", r.name, t_conv, t_dur)

    if out_dir is not None:
        out_dir = Path(out_dir)
        save_results(
            out_dir / "results.npz", results, convergence_times=convergence_times
        )
        if cfg.plot:
            for i, (r, (_, t_conv, _)) in enumerate(zip(results, convergence_times)):
                visualize_stage(
                    model,
                    r.h,
                    r.t,
                    inputs.get_input(r.name),
                    stage_label(i, r.name),
                    t_conv,
                    out_dir,
                )
            t, h = concatenate_stages(results)
            visualize_trial(model, h, t, out_dir)
    return model, results, convergence_times
