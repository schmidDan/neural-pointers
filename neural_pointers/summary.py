"""Key quantities of the default experiments, used to compare model versions (see `tools/compare.py`)."""

import re
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from neural_pointers.simulation import concatenate_stages

EXPERIMENTS = ["pointers", "competitive_queueing", "difference_computation"]


def _round(x, digits=4):
    """Rounds to significant digits, recursively for lists."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_round(v, digits) for v in x]
    x = float(x)
    if not np.isfinite(x) or x == 0.0:
        return x
    return float(f"{x:.{digits}g}")


def summarize_pointers(model, results):
    """Read-out winners per `WTA<k>` stage, synaptic weights after binding and value ranges."""
    winners, winner_rates, margins = [], [], []
    for r in results:
        if re.fullmatch(r"WTA\d+", r.name):
            rates = model.get_neuron_rates(r.h[-1])
            order = np.argsort(rates)
            winners.append(int(order[-1]))
            winner_rates.append(rates[order[-1]])
            margins.append(rates[order[-1]] - rates[order[-2]])
    wta_items = [int(r.name[3:]) for r in results if re.fullmatch(r"WTA\d+", r.name)]
    binding_end = [r for r in results if re.fullmatch(r"binding\d+", r.name)][-1].h[-1]
    weights = model.get_synaptic_weights(binding_end)
    _, h = concatenate_stages(results)
    states = np.array([model.get_neuron_state(x) for x in h])
    syn = np.array([model.get_synaptic_state(x) for x in h])
    return {
        "readout_winners": winners,
        "readout_correct": winners == wta_items,
        "readout_winner_rate": _round(winner_rates),
        "readout_margin": _round(margins),
        "weight_bound_mean": _round(np.mean(np.diag(weights))),
        "weight_unbound_mean": _round(
            np.mean(weights[~np.eye(len(weights), dtype=bool)])
        ),
        "syn_state_max": _round(syn.max()),
        "potential_min": _round(states.min()),
        "potential_max": _round(states.max()),
    }


def summarize_competitive_queueing(model, results):
    """Item selected by each arousal pulse and output rates."""
    selected, peaks = [], []
    for r in results:
        if r.name == "aroused":
            rates = np.array([model.get_neuron_rate_output_nodes(x) for x in r.h])
            selected.append(int(np.argmax(rates[-1])))
            peaks.append(rates.max())
    n = model.n_neurons
    return {
        "execution_order": selected,
        "execution_order_correct": selected
        == list(range(n - 1, n - 1 - len(selected), -1)),
        "output_peak_rate": _round(peaks),
    }


def summarize_difference_computation(model, results, convergence_times):
    """Convergence durations per stage and range of the difference signal."""
    _, h = concatenate_stages(results)
    d = np.array([model.get_neuron_rate_exc_nodes(x) for x in h])
    return {
        "convergence_duration": _round(convergence_times[:, 2]),
        "difference_min": _round(d.min()),
        "difference_max": _round(d.max()),
    }


def summarize(experiment, cfg):
    """Runs `experiment` with `cfg` (nothing is stored) and returns its summary."""
    if experiment == "pointers":
        from neural_pointers.pointers.experiment import run_experiment

        return summarize_pointers(*run_experiment(cfg))
    if experiment == "competitive_queueing":
        from neural_pointers.competitive_queueing.experiment import run_experiment

        return summarize_competitive_queueing(*run_experiment(cfg))
    if experiment == "difference_computation":
        from neural_pointers.difference_computation.experiment import run_experiment

        return summarize_difference_computation(*run_experiment(cfg))
    raise ValueError("Unknown experiment, got: {}".format(experiment))


def summarize_all(config_root, overrides=()):
    """Summaries of all experiments with their default configs from `config_root/<experiment>/config.yaml`."""
    summaries = {}
    for experiment in EXPERIMENTS:
        with initialize_config_dir(
            version_base="1.3", config_dir=str(Path(config_root, experiment).resolve())
        ):
            cfg = compose(config_name="config", overrides=list(overrides))
        summaries[experiment] = summarize(experiment, cfg)
    return summaries
