"""Visualization of simulations of the synaptic pointer model."""

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


def visualize_stage(model, h_m, t, ode_args, stage, out_dir="."):
    """Plots the final values at the end of a simulation stage and their evolution over the stage.

    Args:
        model: The simulated `SynapticPointerModel`.
        h_m: Array of states for the time points `t`.
        t: Time points of the stage.
        ode_args: Inputs passed to `model` during the stage.
        stage: Stage label, used in the file names.
        out_dir: Directory to store the figures in.
    """
    out_dir = Path(out_dir)

    plt.figure()
    plt.plot(model.get_neuron_state(h_m[-1]))
    plt.savefig(out_dir / "stage_{}_final_h_neur.png".format(stage))

    plt.figure()
    plt.plot(model.get_neuron_state(h_m[-1]))
    plt.plot(model.get_neuron_rates(h_m[-1]))
    plt.plot(ode_args[0])
    plt.legend(["states", "rates", "input"])
    plt.savefig(out_dir / "stage_{}_final_rate.png".format(stage))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    sns.heatmap(model.get_synaptic_state(h_m[-1]), ax=ax1)
    ax1.set_title("syn state")
    sns.heatmap(model.get_synaptic_weights(h_m[-1]), ax=ax2)
    ax2.set_title("syn weights")
    ax3.plot(ode_args[1])
    ax3.set_title("syn input")
    f.tight_layout()
    plt.savefig(out_dir / "stage_{}_final_syn.png".format(stage))
    plt.savefig(out_dir / "stage_{}_final_syn.svg".format(stage))

    _save_heatmap(
        _over_time(model.get_synaptic_state, h_m),
        out_dir / "stage_{}_synapses_over_time.png".format(stage),
    )
    _save_heatmap(
        _over_time(model.get_synaptic_weights, h_m),
        out_dir / "stage_{}_synaptic_weights_over_time.png".format(stage),
    )
    _save_heatmap(
        _over_time(model.get_neuron_state, h_m),
        out_dir / "stage_{}_states_over_time.png".format(stage),
    )
    _save_heatmap(
        _over_time(model.get_neuron_rates, h_m),
        out_dir / "stage_{}_rates_over_time.png".format(stage),
    )
    plt.close("all")


def visualize_trial(model, h, t, out_dir="."):
    """Plots the evolution of the model over the entire simulation.

    Args:
        model: The simulated `SynapticPointerModel`.
        h: Array of states of the entire simulation.
        t: Time points of the entire simulation.
        out_dir: Directory to store the figures in.
    """
    out_dir = Path(out_dir)

    syn_states = _over_time(model.get_synaptic_state, h)
    _save_heatmap(syn_states, out_dir / "synapses_over_time.png")
    _save_heatmap(
        _over_time(model.get_synaptic_weights, h),
        out_dir / "synaptic_weights_over_time.png",
    )
    _save_heatmap(
        _over_time(model.get_neuron_state, h), out_dir / "states_over_time.png"
    )
    rates = _over_time(model.get_neuron_rates, h)
    _save_heatmap(rates, out_dir / "rates_over_time.png")
    plt.savefig(out_dir / "rates_over_time.svg")

    plt.figure()
    strongest_syn = np.argmax(np.max(syn_states, axis=1))
    domain, pointer = np.unravel_index(strongest_syn, (model.n_neurons,) * 2)
    plt.plot(t, syn_states[strongest_syn, :])
    plt.title(
        "strongest synapse: pointer {} to domain neuron {}".format(pointer, domain)
    )
    plt.xlabel("time")
    plt.ylabel("synaptic state")
    plt.savefig(out_dir / "syn_state_strongest_syn_over_time.png")

    plt.figure()
    plt.plot(t, np.transpose(rates))
    plt.savefig(out_dir / "result_rates_over_time.png")
    plt.savefig(out_dir / "result_rates_over_time.svg")

    plt.close("all")
