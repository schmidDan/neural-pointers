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

#    r_exc = np.array(
#            [*map(lambda x: model.get_neuron_rate_exc_nodes(x), h_m)],
#            )

#    idxs_converged_candidates = np.where(np.abs(r_exc) < 0.001)[0]
#    idx_not_start = np.argmax(idxs_converged_candidates > 100)
#    idx_converged = idxs_converged_candidates[idx_not_start]

#    print("------------------------")
#    print("RATES EXC")
#    print("idxs_converged_candidates: {}".format(idxs_converged_candidates))
#    print(np.where(idxs_converged_candidates[0] > 100)[0])
#    print("idx_not_start: {}".format(idx_not_start))
#    print("idx_converged: {}".format(idx_converged))
#    print("ts: {}".format(t))
#    print("convergence time: {}".format(t[idx_converged]))
#    print("________________________")

    r_evdc = np.array(
            [*map(lambda x: model.get_neuron_rate_evdc_nodes(x), h_m)],
            )
    
    idxs_converged_candidates = np.where(r_evdc)[0]
    print(r_evdc)
    print(any(r_evdc))
    print(idxs_converged_candidates)
    #idx_not_start = np.argmax(idxs_converged_candidates > 100)
    #idx_not_start = np.argmax(idxs_converged_candidates)
    #idx_converged = idxs_converged_candidates[idx_not_start]
    idx_converged = idxs_converged_candidates[0]
    t_start = t[0]
    t_converged = t[idx_converged]
    t_dur = t_converged - t_start
    
    print("/////////////////////////")
    print("RATES EVDC")
    print("idxs_converged_candidates: {}".format(idxs_converged_candidates))
    #print(np.where(idxs_converged_candidates[0] > 100)[0])
#    print("idx_not_start: {}".format(idx_not_start))
    print("idx_converged: {}".format(idx_converged))
    print("ts: {}".format(t))
    print("convergence time: {}".format(t_converged))
    print("|||||||||||||||||||||||||")
    
    plt.figure()
    h = plt.plot(t, np.array([*map(lambda x: model.get_neuron_state_exc_nodes(x), h_m)]))
    h = plt.plot(t, np.array([*map(lambda x: model.get_neuron_state_inh_nodes(x), h_m)]))
    h = plt.plot(t, np.array([*map(lambda x: np.sum(model.get_neuron_state_inp_nodes(x)), h_m)]))    
    h = plt.plot(t, np.array([*map(lambda x: model.get_neuron_state_evdc_nodes(x), h_m)]))  
    h = plt.plot([t_converged,t_converged],[0,1], "k--")
    plt.legend(["exc", "inh", "summed input", "evidence", "convergence"])
    plt.savefig("stage_{}_states_over_time.png".format(stage))


    plt.figure()
    h = plt.plot(t, np.array([*map(lambda x: model.get_neuron_rate_exc_nodes(x), h_m)]))
    h = plt.plot(t, np.array([*map(lambda x: model.get_neuron_rate_inh_nodes(x), h_m)]))
    h = plt.plot(t, np.array([*map(lambda x: np.sum(model.get_neuron_rate_inp_nodes(x)), h_m)]))  
    
    h = plt.plot(t, np.array([*map(lambda x: (np.abs(model.get_neuron_rate_exc_nodes(x)) < 0.001), h_m)]))
    
    h = plt.plot(t, np.array([*map(lambda x: model.get_neuron_rate_evdc_nodes(x), h_m)]))
    h = plt.plot([t_converged,t_converged],[0,1], "k--")
    plt.legend(["exc", "inh", "summed input", "evidence_input", "evidence", "convergence"])
    plt.savefig("stage_{}_rates_over_time.png".format(stage))

    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_state_inp_nodes(x), (-1, 1)), h_m)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("stage_{}_inp_states_over_time.png".format(stage))

    
    plt.close("all")

    return t_start, t_converged, t_dur

def visualize_trial(model, state_list, t):
    """Visualization of the entire simulation over time.

    Args:
        model: `SynapticPointerModel` which has been simulated druing the `stage`.
        state_list: A list of state arrays, where each array was the returned state values by the numerical integration
        procedure for the queried points in time.
    """
    h_tot = np.concatenate(state_list, axis=0)

    print(model.get_neuron_state_inp_nodes(h_tot[-1]))
    
    plt.figure()
    h = plt.plot(t,
        np.array(
            [*map(lambda x: model.get_neuron_state_exc_nodes(x), h_tot)],
        )
    )
    h = plt.plot(t,
        np.array(
            [*map(lambda x: model.get_neuron_state_inh_nodes(x), h_tot)],
        )
    )
    h = plt.plot(t,
        np.array(
            [*map(lambda x: np.sum(model.get_neuron_state_inp_nodes(x)), h_tot)],
        )
    )    
    h = plt.plot(t,
        np.array(
            [*map(lambda x: model.get_neuron_state_evdc_nodes(x), h_tot)],
        )
    )
    plt.legend(["exc", "inh", "summed input", "evidence"])
    plt.savefig("states_over_time.png")

    plt.figure()
    h = plt.plot(t,
        np.array(
            [*map(lambda x: model.get_neuron_rate_exc_nodes(x), h_tot)],
        )
    )
    h = plt.plot(t,
        np.array(
            [*map(lambda x: model.get_neuron_rate_inh_nodes(x), h_tot)],
        )
    )
    h = plt.plot(t,
        np.array(
            [*map(lambda x: np.sum(model.get_neuron_rate_inp_nodes(x)), h_tot)],
        )
    )    
    h = plt.plot(t,
        np.array(
            [*map(lambda x: model.get_neuron_rate_evdc_nodes(x), h_tot)],
        )
    )
    plt.legend(["difference", "prediction", "summed input", "evidence"])
    plt.savefig("rates_over_time.png")
    plt.savefig("rates_over_time.svg")

    plt.figure()
    h = plt.plot(
        np.array(
            [*map(lambda x: model.get_neuron_state_exc_nodes(x), h_tot)],
        )
    )
    plt.savefig("exc_states_over_time.png")
    
    plt.figure()
    h = plt.plot(
        np.array(
            [*map(lambda x: model.get_neuron_rate_exc_nodes(x), h_tot)],
        )
    )
    plt.savefig("exc_rates_over_time.png")
    
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_state_exc_nodes(x), (-1, 1)), h_tot)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("exc_states_over_time.png")
    
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_state_inh_nodes(x), (-1, 1)), h_tot)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("inh_states_over_time.png")
    
    plt.figure()
    hmp = sns.heatmap(
        np.concatenate(
            [*map(lambda x: np.reshape(model.get_neuron_state_inp_nodes(x), (-1, 1)), h_tot)],
            axis=1,
        )
    )
    hmp.get_figure().savefig("inp_states_over_time.png")
    hmp.get_figure().savefig("inp_states_over_time.svg")

    plt.close("all")
