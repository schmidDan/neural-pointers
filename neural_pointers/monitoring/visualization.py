"""Visualization of simulations of the monitoring model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _over_time(getter, h):
    """Stacks `getter(h[k])` (flattened) column-wise for all time steps `k`."""
    return np.concatenate([np.reshape(getter(x), (-1, 1)) for x in h], axis=1)


def _save_heatmap(data, path):
    plt.figure()
    hmp = sns.heatmap(data, rasterized=True)
    hmp.get_figure().savefig(path)
    return hmp


def visualize_stage(model, h_m, t, ode_args, stage, t_converged, out_dir="."):
    """Plots the evolution of all populations over a simulation stage.

    Args:
        model: The simulated `MonitoringModel`.
        h_m: Array of states for the time points `t`.
        t: Time points of the stage.
        ode_args: Inputs passed to `model` during the stage.
        stage: Stage label, used in the file names.
        t_converged: Time of the first convergence signal in this stage (`nan` if none), marked by a dashed line.
        out_dir: Directory to store the figures in.
    """
    out_dir = Path(out_dir)

    plt.figure()
    plt.plot(t, np.array([model.get_neuron_state_difference_nodes(x) for x in h_m]))
    plt.plot(t, np.array([model.get_neuron_state_prediction_nodes(x) for x in h_m]))
    plt.plot(t, np.array([np.sum(model.get_neuron_state_input_nodes(x)) for x in h_m]))
    plt.plot(t, np.array([model.get_neuron_state_evidence_nodes(x) for x in h_m]))
    plt.plot(t, np.array([model.get_adaptation_state(x) for x in h_m]))
    plt.plot([t_converged, t_converged], [0, 1], "k--")
    plt.legend(
        [
            "difference",
            "prediction",
            "summed input",
            "evidence",
            "adaptation",
            "convergence",
        ]
    )
    plt.savefig(out_dir / "stage_{}_states_over_time.png".format(stage))

    plt.figure()
    plt.plot(t, np.array([model.get_neuron_rate_difference_nodes(x) for x in h_m]))
    plt.plot(t, np.array([model.get_neuron_rate_prediction_nodes(x) for x in h_m]))
    plt.plot(t, np.array([np.sum(model.get_neuron_rate_input_nodes(x)) for x in h_m]))
    plt.plot(t, np.array([model.get_closeness(x) for x in h_m]))
    plt.plot(t, np.array([model.get_neuron_rate_evidence_nodes(x) for x in h_m]))
    plt.plot([t_converged, t_converged], [0, 1], "k--")
    plt.legend(
        [
            "difference",
            "prediction",
            "summed input",
            "closeness",
            "evidence",
            "convergence",
        ]
    )
    plt.savefig(out_dir / "stage_{}_rates_over_time.png".format(stage))

    _save_heatmap(
        _over_time(model.get_neuron_state_input_nodes, h_m),
        out_dir / "stage_{}_input_states_over_time.png".format(stage),
    )
    plt.close("all")


def visualize_trial(model, h, t, out_dir="."):
    """Plots the evolution of the model over the entire simulation.

    Args:
        model: The simulated `MonitoringModel`.
        h: Array of states of the entire simulation.
        t: Time points of the entire simulation.
        out_dir: Directory to store the figures in.
    """
    out_dir = Path(out_dir)

    plt.figure()
    plt.plot(t, np.array([model.get_neuron_state_difference_nodes(x) for x in h]))
    plt.plot(t, np.array([model.get_neuron_state_prediction_nodes(x) for x in h]))
    plt.plot(t, np.array([np.sum(model.get_neuron_state_input_nodes(x)) for x in h]))
    plt.plot(t, np.array([model.get_neuron_state_evidence_nodes(x) for x in h]))
    plt.legend(["difference", "prediction", "summed input", "evidence"])
    plt.savefig(out_dir / "states_over_time.png")

    plt.figure()
    plt.plot(t, np.array([model.get_neuron_rate_difference_nodes(x) for x in h]))
    plt.plot(t, np.array([model.get_neuron_rate_prediction_nodes(x) for x in h]))
    plt.plot(t, np.array([np.sum(model.get_neuron_rate_input_nodes(x)) for x in h]))
    plt.plot(t, np.array([model.get_neuron_rate_evidence_nodes(x) for x in h]))
    plt.legend(["difference", "prediction", "summed input", "evidence"])
    plt.savefig(out_dir / "rates_over_time.png")
    plt.savefig(out_dir / "rates_over_time.svg")

    plt.figure()
    plt.plot(np.array([model.get_neuron_state_difference_nodes(x) for x in h]))
    plt.savefig(out_dir / "difference_states_over_time_lines.png")

    plt.figure()
    plt.plot(np.array([model.get_neuron_rate_difference_nodes(x) for x in h]))
    plt.savefig(out_dir / "difference_rates_over_time.png")

    _save_heatmap(
        _over_time(model.get_neuron_state_difference_nodes, h),
        out_dir / "difference_states_over_time.png",
    )
    _save_heatmap(
        _over_time(model.get_neuron_state_prediction_nodes, h),
        out_dir / "prediction_states_over_time.png",
    )
    hmp = _save_heatmap(
        _over_time(model.get_neuron_state_input_nodes, h),
        out_dir / "inp_states_over_time.png",
    )
    hmp.get_figure().savefig(out_dir / "inp_states_over_time.svg")
    plt.close("all")
