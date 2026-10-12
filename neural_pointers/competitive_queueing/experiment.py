"""Competitive queueing experiment."""

from pathlib import Path

import numpy as np

from neural_pointers.competitive_queueing.inputs import CompetitiveQueueingInputs
from neural_pointers.competitive_queueing.model import CompetitiveQueueingModel
from neural_pointers.competitive_queueing.visualization import (
    visualize_stage,
    visualize_trial,
)
from neural_pointers.simulation import (
    concatenate_stages,
    save_results,
    simulate_stages,
    stage_label,
)


def run_experiment(cfg, out_dir=None):
    """Simulates the competitive queueing model for the stages defined in `cfg.simulation`.

    Args:
        cfg: Experiment config with `seed`, `n_neurons`, `plot`, `model` and `simulation`.
        out_dir: Directory to store results and figures in. Nothing is stored if `None`.

    Returns:
        tuple: The model and the list of `StageResult`s.
    """
    rng = np.random.default_rng(cfg.seed)
    model = CompetitiveQueueingModel(cfg.n_neurons, cfg.model)
    inputs = CompetitiveQueueingInputs(cfg.n_neurons, rng)

    results = simulate_stages(
        model, inputs.get_input, cfg.simulation.stages, cfg.simulation.dt, model.h_0
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        save_results(out_dir / "results.npz", results)
        if cfg.plot:
            for i, r in enumerate(results):
                visualize_stage(
                    model,
                    r.h,
                    r.t,
                    inputs.get_input(r.name),
                    stage_label(i, r.name),
                    out_dir,
                )
            t, h = concatenate_stages(results)
            visualize_trial(model, h, t, out_dir)
    return model, results
