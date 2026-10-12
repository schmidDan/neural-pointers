"""Visualization of simulations of the competitive queueing model."""

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


def _summary_plot(getter, h, n_neurons, path, getter_dashed=None):
    """Plots the summed rates (top row) and the rate of each item (one row per item) over time steps.

    Args:
        getter: Rate getter plotted dashed, or solid if `getter_dashed` is given.
        h: Array of states.
        n_neurons: Number of items.
        path: Output file path.
        getter_dashed: Optional second rate getter plotted dashed into the same rows.
    """
    getters = (
        [(getter, "-"), (getter_dashed, "--")] if getter_dashed else [(getter, "--")]
    )
    plt.figure(figsize=(12, 8), dpi=150)
    plt.subplot(n_neurons + 1, 1, 1)
    for g, style in getters:
        plt.plot(np.array([sum(g(x)) for x in h]), style)
    for idx in range(n_neurons):
        plt.subplot(n_neurons + 1, 1, idx + 2)
        for g, style in getters:
            plt.plot(np.array([g(x)[idx] for x in h]), style)
        plt.title("idx: {}".format(idx))
    plt.tight_layout()
    plt.savefig(path)


def visualize_stage(model, h_m, t, ode_args, stage, out_dir="."):
    """Plots the final values at the end of a simulation stage and their evolution over the stage.

    Args:
        model: The simulated `CompetitiveQueueingModel`.
        h_m: Array of states for the time points `t`.
        t: Time points of the stage.
        ode_args: Inputs passed to `model` during the stage.
        stage: Stage label, used in the file names.
        out_dir: Directory to store the figures in.
    """
    out_dir = Path(out_dir)

    plt.figure()
    plt.plot(model.get_neuron_state_input_nodes(h_m[-1]))
    plt.savefig(out_dir / "stage_{}_final_h_neur_x.png".format(stage))

    inp_norm = model.get_neuron_state_input_nodes(h_m[-1])
    inp_norm = (inp_norm - np.min(inp_norm)) / np.abs(np.max(inp_norm))

    plt.figure()
    plt.bar(list(range(model.n_neurons)), inp_norm)
    plt.savefig(out_dir / "result_stage_{}_final_h_neur_x_norm.png".format(stage))
    plt.savefig(out_dir / "result_stage_{}_final_h_neur_x_norm.svg".format(stage))

    plt.figure()
    plt.plot(model.get_neuron_state_output_nodes(h_m[-1]))
    plt.savefig(out_dir / "stage_{}_final_h_neur_y.png".format(stage))

    plt.figure()
    plt.plot(model.get_neuron_state_inhib_nodes(h_m[-1]))
    plt.savefig(out_dir / "stage_{}_final_h_neur_w.png".format(stage))

    _save_heatmap(
        _over_time(model.get_neuron_state_input_nodes, h_m),
        out_dir / "stage_{}_x_states_over_time.png".format(stage),
    )
    _save_heatmap(
        _over_time(model.get_neuron_state_output_nodes, h_m),
        out_dir / "stage_{}_y_states_over_time.png".format(stage),
    )
    _save_heatmap(
        _over_time(model.get_neuron_state_inhib_nodes, h_m),
        out_dir / "stage_{}_w_states_over_time.png".format(stage),
    )
    plt.close("all")


def visualize_trial(model, h, t, out_dir="."):
    """Plots the evolution of the model over the entire simulation.

    Args:
        model: The simulated `CompetitiveQueueingModel`.
        h: Array of states of the entire simulation.
        t: Time points of the entire simulation.
        out_dir: Directory to store the figures in.
    """
    out_dir = Path(out_dir)
    n = model.n_neurons

    _summary_plot(
        model.get_neuron_rate_input_nodes,
        h,
        n,
        out_dir / "summary_plot_over_time.png",
        getter_dashed=model.get_neuron_rate_output_nodes,
    )
    _summary_plot(
        model.get_neuron_rate_output_nodes,
        h,
        n,
        out_dir / "summary_plot_outputs_over_time.png",
    )
    _summary_plot(
        model.get_neuron_rate_input_nodes,
        h,
        n,
        out_dir / "summary_plot_inputs_over_time.png",
    )
    _summary_plot(
        model.get_neuron_rate_inhib_nodes,
        h,
        n,
        out_dir / "summary_plot_inhib_over_time.png",
    )

    # output rates in order of execution (the item with the highest index is executed first)
    plt.figure()
    for idx in range(n - 1, -1, -1):
        plt.plot(t, np.array([model.get_neuron_rate_output_nodes(x)[idx] for x in h]))
    plt.legend(["op{}".format(k + 1) for k in range(n)])
    plt.savefig(out_dir / "result_output_over_time.png")
    plt.savefig(out_dir / "result_output_over_time.svg")

    _save_heatmap(
        _over_time(model.get_neuron_state_input_nodes, h),
        out_dir / "states_x_over_time.png",
    )
    _save_heatmap(
        _over_time(model.get_neuron_state_output_nodes, h),
        out_dir / "states_y_over_time.png",
    )
    plt.savefig(out_dir / "states_y_over_time.svg")
    _save_heatmap(
        _over_time(model.get_neuron_state_inhib_nodes, h),
        out_dir / "states_w_over_time.png",
    )
    plt.close("all")
