# Known issues

Issues found in the code state of the BonnBrain 2024 poster (tag `poster-bonnbrain2024`). Open issues are listed
first, resolved ones at the end together with their effect on the results (see also [CHANGELOG.md](../CHANGELOG.md)).
Issues marked *(output)* only affect figures or stored files, all others change the simulated dynamics.

The poster states the dynamics of the domain (storage) neurons `v_s` and of the pointer synapses `w_ij`:

    tau dv_s/dt = -alpha v_s + (beta - v_s) Exc_basal (1 + Mod_apical) - (delta + gamma v_s) Inh_pool
    dw_ij/dt    = (w_max - w_ij) f(v_p,i, v_s,j) / tau_rise + (w_min - w_ij) (1 - f(v_p,i, v_s,j)) / tau_decay

These equations are cited below as *[poster]*. In the code, the domain potentials are called `x`, the pointer
activities `p` and the synaptic states `s`. Note that `delta` denotes the subtractive inhibition in *[poster]* but the
gain of the pointer modulation in the code.

## Differences between the poster model and the implementation

### S1 Pointer model: no shunting excitation, unbounded potentials
*[poster]* uses shunting excitation `(beta - v_s) Exc_basal (1 + Mod_apical)`. The implementation uses additive
excitation `beta * I_i * (1 + delta * m_i)`, and the default parameters have no leak (`alpha = 0`). As a result, the
potentials are not bounded. During read-out the winner's rate grows linearly (up to 274 in the default simulation) and
the potentials of the losing neurons diverge to large negative values (below -17,000). The read-out still selects the
correct item, since rates are rectified.

### S2 Pointer model: only subtractive inhibition, `gamma` unused
*[poster]* combines subtractive and shunting inhibition, `(delta + gamma v_s) Inh_pool`. The implementation only has
the subtractive part, `eta * inh_i`. The parameter `gamma` is configured (1.0 in both modes) but does not enter the
dynamics. Binding and read-out differ in `kappa`, `k_inh` (0 → 1) and the pointer gain `delta` (0.1 → 10).

### S3 Pointer model: composition of the pooled inhibition
The implementation computes `inh_i = (1 - k_inh - kappa) * r_i + k_inh * sum_j r_j + x_i * I_inh_ext_i`. For
`kappa = k_inh = 1` (read-out) the pooled inhibition is the summed activity of all other neurons, i.e. winner-take-all
competition without self-inhibition. For `kappa = k_inh = 0` (binding) it reduces to self-inhibition `r_i`, which acts
as the leak, since `alpha = 0`. The shunting term `x_i * I_inh_ext_i` resets the domain between stages and is not part
of *[poster]*.

### S4 Pointer model: no pointer neuron dynamics, all pointers coupled to all domain neurons
The poster depicts the pointers as a population of neurons `v_p`. In the implementation, pointer activities are given
as external inputs. The effective pointer weights are `1 + s_ij` with `s_ij ∈ [0, 1]`, so every pointer modulates
every domain neuron with a baseline weight of 1, whether bound or not. The Hebbian term `f` of *[poster]* is
implemented as the product of pointer activity and domain rate (see B6).

### S5 Serial order is encoded twice, in opposite directions
`PointerInputs` scales pointer `k` by `1 - 0.5 * k / (n - 1)` "to encode the serial order" (item 0 is strongest). This
gradient does not determine the read-out order, which is fixed by the external stage sequence `WTA0, WTA1, ...`.
`CompetitiveQueueingInputs` uses the opposite convention: the gradient increases with the index, so item `n - 1` is
executed first.

### S6 The three models are not coupled
The poster describes an architecture in which the competitive queueing network schedules operations, the monitoring
network advances the sequence via the arousal signal, and pointers retrieve the content. In the implementation, the
three models are simulated independently. They are related only through stage durations that were aligned by hand
(see the comments in `configs/*/simulation/default.yaml`). Arousal pulses (CQ) and read-out stages (pointers) are
scheduled externally instead of being triggered by the convergence signal (DC).

### S7 Difference computation: convergence signal relies on explicit time
The evidence neurons accumulate only if `|d| < 0.001` and at least 20 time units have passed since the last signal.
The two timers are updated based on the simulation time `t` inside the model (see B1), not by neural state variables.
The threshold 0.001, the 20 time-unit refractory period and the 10 time-unit hold period are hard-coded.

## Bugs and implementation issues

### B1 Difference computation: model state is changed inside the ODE right-hand side
`DifferenceComputationModel.__call__` updates `_t_start` and `_t_last` whenever the solver evaluates it, including
trial steps that are later rejected. The solution therefore depends on the solver's internal step sequence. Between
Python 3.10 and 3.12 environments (different SciPy versions) the trajectories differ by up to 0.05 (5 % of the signal
range), and the final convergence pulse in the poster figure differs from current runs. This is why the reproduction
test for this model uses a coarse tolerance.

### B2 Difference computation: weight normalization uses the wrong axis
`W / np.sum(W, axis=1)` divides column `j` by the sum of row `j` (broadcasting without `keepdims`), so neither rows
nor columns of the recurrent weight matrices sum to one.

### B3 Difference computation: convergence measurement picks up evidence from the previous stage
The convergence time of a stage is the first sample at which the evidence output is on. If the evidence neuron is
still above threshold at stage onset, convergence is reported immediately: in the default simulation `pattern1`
"converges" at its first sample (duration 0). The stage durations of the difference computation experiment, to which
the other experiments are aligned (S6), are roughly the measured convergence durations plus a margin (e.g. 124 → 135,
56 → 70, 81 → 95). For `pattern1` this measurement is unusable.

### B4 Difference computation: only `n_neurons = 1`
The evidence timers are scalar, so the model only supports a single difference neuron (an explicit error is raised
otherwise). The size of the input layer and the seed of the recurrent weights, previously hard-coded, are configurable
as `model.n_inputs_per_neuron` (7) and `model.weight_seed` (132).

### B5 All experiments: consecutive stages share one time point
`simulate_stages` starts each stage at the last time point of the previous one. Each stage therefore starts one `dt`
earlier than its nominal onset, the concatenated trajectories contain duplicated samples at stage boundaries, and
hand-aligned onsets across experiments drift by one `dt` per stage.

### B6 Pointer model: synaptic states exceed `syn_max`
The Hebbian coincidence term `c_ij = p_j * r_i` is not bounded to [0, 1]. For `c_ij > 1` the decay term changes sign
and the steady state lies above `syn_max` (maximum 1.005 in the default simulation).

## Resolved

### B8 Pointer visualization: wrong "strongest synapse" *(output)*
`syn_state_strongest_syn_over_time.png` plots the synapse whose maximum occurs last, not the synapse with the
largest value.

**Resolved:** The plot shows the synapse with the largest state over the simulation, labeled with its pointer and domain neuron. Figures only.
### B10 Pointer visualization: mislabeled legend *(output)*
In `stage_<name>_final_syn.png` the legend `syn_state, syn_weights, syn_input` is attached to the pointer input panel,
which contains a single line.

**Resolved:** Removed the legend; the panel title identifies the pointer input. Figures only.
### B9 Competitive queueing visualization: stacked heatmaps *(output)*
`states_x_over_time`, `states_y_over_time` and `states_w_over_time` are drawn into the same figure, so the latter two
contain the colorbars of the previous heatmaps. This includes `states_y_over_time.svg` used for the poster.

**Resolved:** Each heatmap is drawn into its own figure. The y heatmap (`states_y_over_time`) now has a single colorbar, unlike the poster version. Figures only.
### B11 Per-stage figures of repeated stages overwrite each other *(output)*
Stage figures are named after the stage, so for stages that occur several times (`reset`, `numbed`, `aroused`) only
the figures of the last occurrence remain. The *Queueing* panel of the poster therefore shows the input layer at the
end of the simulation rather than the initial queue.

**Resolved:** Stage figures are prefixed with the stage index (e.g. `stage_12_numbed_...`). The *Queueing* panel of the overview figure still shows the state at the end of the simulation, like the poster; which state it should show is left for the revision of the competitive queueing protocol (S5, S6). Figures only.

### B12 Heatmaps stored as vector graphics *(output)*
Heatmaps over time contain one cell per time step (about 57,000). Stored as SVG, every cell became a vector shape, so
each default run wrote 60-125 MB per SVG figure.

**Resolved:** Heatmap cells are rasterized while axes and labels stay vector graphics. All SVG figures are below
200 KB; PNG figures are unchanged. Figures only.
### B7 Pointer inputs: `n_neurons = 1` divides by zero
The pointer strength `1 - 0.5 * k / (n - 1)` is undefined for a single neuron.

**Resolved:** A single pointer has strength 1.0. No change for the default configuration.

## Fixed during the clean-up

These issues did not affect the simulated dynamics or the poster figures:

* The difference computation results were never stored (pickles were written to a non-existent directory). All
  experiments now store `results.npz` in the Hydra output directory.
* Running the difference computation experiment with `plot=false` crashed, and stages without a convergence signal
  raised an `IndexError`.
* The competitive queueing model printed its normalization term at every evaluation of the right-hand side.
* The competitive queueing figures assumed exactly six items.
* The difference computation line plot `exc_states_over_time.png` was overwritten by the heatmap of the same name. The
  line plot is now stored as `exc_states_over_time_lines.png`.
* Several stored configuration variants were copies of other models' configs and could not be run. Variants are
  available in the git history.
