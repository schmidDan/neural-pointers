"""Methods to visualize experiments performed on synaptic pointer models.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_stage(model, h_m, t, ode_args, stage):
    """Visualization of a single `stage` of the simulation. Visualizes the final values at the end of the `stage` and
    how values evolved over time.

    Args:
        model: `SynapticPointerModel` which has been simulated druing the `stage`.
        h_m: Array of state values for the queried points in time `t`.
        t: Queried points in time, for which the `model` has been evaluated.
        ode_args: Arguments passed to the `model` during numerical intergration.
        stage: String containing the current stage that is simulated. Used to make the visualizations more
        informative.
    """
    plt.figure()
    plt.plot(model.get_neuron_state(h_m[-1]))
    plt.savefig("stage_{}_final_h_neur.png".format(stage))

    plt.figure()
    plt.plot(model.get_neuron_state(h_m[-1]))
    plt.plot(model.get_neuron_rates(h_m[-1]))
    plt.plot(ode_args[0])
    plt.legend(["states", "rates", "input"])
    plt.savefig("stage_{}_final_rate.png".format(stage))

    plt.figure()
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # xlabels = t[np.linspace(0, t.size-1, 10, dtype=int)]
    hmp = sns.heatmap(
        model.get_synaptic_state(h_m[-1]), ax=ax1
    )  # , xticklabels=xlabels)
    ax1.set_title("syn state")
    hmp = sns.heatmap(
        model.get_synaptic_weights(h_m[-1]), ax=ax2
    )  # , xticklabels=xlabels)
    ax2.set_title("syn weights")
    ax3.plot(ode_args[1])
    ax3.set_title("syn input")
    f.tight_layout()
    plt.legend(["syn_state", "syn_weights", "syn_input"])
    plt.savefig("stage_{}_final_syn.png".format(stage))
    plt.savefig("stage_{}_final_syn.svg".format(stage))

    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_synaptic_state(x), (-1, 1)), h_m)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("stage_{}_synapses_over_time.png".format(stage))
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_synaptic_weights(x), (-1, 1)), h_m)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("stage_{}_synaptic_weights_over_time.png".format(stage))
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_state(x), (-1, 1)), h_m)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("stage_{}_states_over_time.png".format(stage))
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_rates(x), (-1, 1)), h_m)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("stage_{}_rates_over_time.png".format(stage))
    plt.close("all")


def visualize_trial(model, state_list, t):
    """Visualization of the entire simulation over time.

    Args:
        model: `SynapticPointerModel` which has been simulated druing the `stage`.
        state_list: A list of state arrays, where each array was the returned state values by the numerical integration
        procedure for the queried points in time.
    """
    h_tot = np.concatenate(state_list, axis=0)
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_synaptic_state(x), (-1, 1)), h_tot)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("synapses_over_time.png")
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_synaptic_weights(x), (-1, 1)), h_tot)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("synaptic_weights_over_time.png")
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_state(x), (-1, 1)), h_tot)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("states_over_time.png")
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_rates(x), (-1, 1)), h_tot)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("rates_over_time.png")
    hmp.get_figure().savefig("rates_over_time.svg")

    plt.figure()
    syn_states = np.concatenate(
        [*map(lambda x: np.reshape(model.get_synaptic_state(x), (-1, 1)), h_tot)],
        axis=1,
    )
    strongest_syn = np.argmax(np.argmax(syn_states, axis=1), axis=0)
    plt.plot(syn_states[strongest_syn, :])
    plt.savefig("syn_state_strongest_syn_over_time.png")

    
    plt.figure()
    h = plt.plot(t,
        np.transpose(np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_rates(x), (-1, 1)), h_tot)],
            axis=1,
        ))
    )
    plt.savefig("result_rates_over_time.png")
    plt.savefig("result_rates_over_time.svg")
    
    plt.close("all")
