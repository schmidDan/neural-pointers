"""Closed-loop experiment: six operations are bound to pointers and then executed in sequence."""

import re
from pathlib import Path

import numpy as np

from neural_pointers.competitive_queueing.inputs import CompetitiveQueueingInputs
from neural_pointers.competitive_queueing.model import CompetitiveQueueingModel
from neural_pointers.coupled.model import CoupledModel
from neural_pointers.coupled.visualization import visualize_trial
from neural_pointers.monitoring.inputs import MonitoringInputs
from neural_pointers.monitoring.model import MonitoringModel
from neural_pointers.pointers.inputs import PointerInputs
from neural_pointers.pointers.model import DOMAIN_MODE, SynapticPointerModel
from neural_pointers.simulation import concatenate_stages, save_results, simulate_stages

_RGX_SCRIPTED = re.compile(r"reset|transition|binding\d+")


def build(cfg):
    """Creates the coupled model and the function mapping stage names to its inputs.

    Stage names: the scripted stages of `PointerInputs` (`reset`, `binding<k>`) with the loop open, `go` (external
    arousal pulse that starts the execution, loop closed) and `execute` (loop closed).

    Args:
        cfg: Config with `seed`, `n_operations`, `models` (configs of the three components) and `coupling`.

    Returns:
        tuple: The `CoupledModel` and a function mapping a stage name to the model inputs.
    """
    n = cfg.n_operations
    cq = CompetitiveQueueingModel(n, cfg.models.competitive_queueing)
    pointers = SynapticPointerModel(n, cfg.models.pointers)
    monitoring = MonitoringModel(1, cfg.models.monitoring)

    cq_inputs = CompetitiveQueueingInputs(n, None)
    pt_inputs = PointerInputs(n, np.random.default_rng(cfg.seed))
    monitoring_inputs = MonitoringInputs(
        cfg.models.monitoring.n_inputs_per_neuron,
        np.random.default_rng(cfg.seed),
    )
    # one pattern per operation; a longer list is truncated to the first n_operations entries
    assignment = list(cfg.coupling.get("processing_patterns") or range(n))[:n]
    processing_patterns = np.stack(
        [monitoring_inputs.patterns[k] for k in assignment], axis=1
    )
    model = CoupledModel(cq, pointers, monitoring, processing_patterns, cfg.coupling)

    def get_input(stage):
        if _RGX_SCRIPTED.fullmatch(stage):
            I, pointer, inh = pt_inputs.get_input(stage)
            return (cq_inputs.I_input_on, I, pointer, inh, 0.0, False)
        tonic, off = pt_inputs.I_input_tonic, pt_inputs.I_pointers_off
        if stage == "go":
            arousal = cfg.coupling.go_arousal
            return (cq_inputs.I_input_on, tonic, off, off, arousal, True)
        if stage == "execute":
            return (cq_inputs.I_input_on, tonic, off, off, 0.0, True)
        raise ValueError("Unrecognized stage, got: {}".format(stage))

    return model, get_input


def _sequence(winners):
    """Collapses a sequence of winners (-1 for none) into the order of distinct winners."""
    order = []
    for w in winners:
        if w >= 0 and (not order or order[-1] != w):
            order.append(int(w))
    return order


def execution_summary(model, results, t_start):
    """Order and timing of the executed operations after `t_start`.

    Args:
        model: The simulated `CoupledModel`.
        results: List of `StageResult`s.
        t_start: Onset of the execution.

    Returns:
        dict: `selected` (CQ), `retrieved` (pointer domain), `convergence_signals` (rising edges of the convergence signal)
        and `execution_durations` (per retrieved operation, from its retrieval onset to its convergence signal;
        `nan` without a signal).
    """
    t, h = concatenate_stages(results)
    keep = t >= t_start
    t, h = t[keep], h[keep]
    parts = [model.split(x) for x in h]
    y = np.array([model.cq.get_neuron_rate_output_nodes(p[0]) for p in parts])
    r = np.array([model.pointers.get_neuron_rates(p[1]) for p in parts])
    signal = np.array([model.convergence_signal(p[2]) for p in parts]) >= 0.5
    signals = t[1:][signal[1:] & ~signal[:-1]]

    retrieved_winner = np.where(r.max(axis=1) > 0.5, r.argmax(axis=1), -1)
    retrieved = _sequence(retrieved_winner)
    durations = []
    for op in retrieved:
        onset = t[np.argmax(retrieved_winner == op)]
        after = signals[signals > onset]
        durations.append(after[0] - onset if after.size else np.nan)
    return {
        "selected": _sequence(np.where(y.max(axis=1) > 1.0, y.argmax(axis=1), -1)),
        "retrieved": retrieved,
        "convergence_signals": signals,
        "execution_durations": np.array(durations),
    }


def run_experiment(cfg, out_dir=None):
    """Simulates the closed loop for the stages defined in `cfg.simulation`.

    Every stage additionally specifies the `mode` (`BINDING` or `WTA`) of the pointer model.

    Args:
        cfg: Experiment config, see `build`, plus `plot` and `simulation`.
        out_dir: Directory to store results and figures in. Nothing is stored if `None`.

    Returns:
        tuple: The model, the list of `StageResult`s and the execution summary (see `execution_summary`).
    """
    model, get_input = build(cfg)
    results = simulate_stages(
        model,
        get_input,
        cfg.simulation.stages,
        cfg.simulation.dt,
        model.h_0,
        on_stage_start=lambda stage: model.pointers.set_mode(DOMAIN_MODE[stage.mode]),
    )
    t_go = [r.t[0] for r in results if r.name == "go"][0]
    summary = execution_summary(model, results, t_go)

    if out_dir is not None:
        out_dir = Path(out_dir)
        save_results(
            out_dir / "results.npz",
            results,
            selected=np.array(summary["selected"]),
            retrieved=np.array(summary["retrieved"]),
            convergence_signals=summary["convergence_signals"],
            execution_durations=summary["execution_durations"],
        )
        if cfg.plot:
            visualize_trial(model, results, t_go, out_dir / "coupled_over_time.png")
    return model, results, summary
