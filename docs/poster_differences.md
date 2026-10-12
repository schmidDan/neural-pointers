# The poster simulations and this release

The models were revised after the poster was presented. This page summarizes how the released models differ from the
poster simulations and how to reproduce the latter.

## Reproducing the poster exactly

The tag `poster-bonnbrain2024` marks the code state whose default simulations are bit-identical to the poster
figures:

```bash
git checkout poster-bonnbrain2024
```

`python tools/compare.py poster-bonnbrain2024 WORKTREE` compares the key quantities and overview figures of two code
versions.

## The poster structure as a preset

The structure of the pointer model used for the poster simulations remains available in the released code as
`model=poster_simulation`: additive excitation without leak, subtractive inhibition, pointer weights `1 + s`, pointer
activities given as external input, and Hebbian plasticity in both modes. The figures in
`docs/figures/poster_simulation/` show its behavior under the released code.

## Model revisions since the poster

### Pointer model

* The default structure follows the model equations shown on the poster: shunting excitation and inhibition bound
  the potentials (to [-9, 1] in the default simulation, instead of growing linearly to 272 and diverging below
  -17,000), pointers are leaky-integrator neurons, and only bound pointer-domain pairs are coupled (`w = s` instead
  of `1 + s`), which makes binding more selective. Winner-take-all read-out with bounded potentials requires stronger
  pooled inhibition (`eta = 10` during read-out). Each structural choice is switchable via `model.variant`.
* The Hebbian coincidence term is bounded, `c = min(p r, 1)`, so synaptic states stay within `[s_min, s_max]`.
* Pointers are formed during binding only (`tau_rise = inf` during read-out). Otherwise a pointer that becomes
  active while the previous operation is still represented binds to it, which breaks the hand-over between
  operations in the closed loop.

### Monitoring network

* The convergence detection is a neural mechanism instead of timers: the evidence neurons accumulate a smooth
  closeness of the difference to zero, self-excitation makes the convergence signal all-or-none, and an adaptation
  variable terminates the signal and re-arms the detector when the input changes. The poster code changed timer
  variables inside the ODE right-hand side, which made the trajectories depend on the solver's internal steps
  (differences up to 5 % between SciPy versions; now below 1e-5).
* The recurrent weights of the input layer are normalized per neuron (the poster code normalized along the wrong
  axis), the prediction ceiling lies above the summed input (`beta_pred = 3`; the poster's saturating prediction
  stalled the difference), and the prediction is slower (`tau_pred = 4`) to damp the difference after input changes.
* The input layer processes its pattern with strong recurrent interactions (`lambda_input = 2`). Its settling time
  is therefore specific to the pattern, which gives each operation its own execution duration in the closed loop.
* Several difference neurons are supported, each monitoring its own group of input neurons.
* Convergence durations of the scripted default simulation: 35.4, 124.2, 0.0, 56.3, 80.7, 76.1, 63.6 on the poster
  (including a 20 time-unit lock-out per signal and a spurious measurement in the third stage) versus 69.1, 46.2,
  44.5, 44.0, 67.7, 44.8 in this release.

### Serial order, timing and coupling

* Operation 0 is executed first in all components (the poster used opposite conventions in the scheduler and the
  pointer inputs).
* Simulation stages cover `[T, T + duration)` on one global time grid; the poster code overlapped consecutive stages
  by one sample.
* The components run together in a closed loop (`run_coupled.py`), which replaces the poster's hand-aligned stage
  durations. The scripted standalone protocols are kept.

## Why the poster structure cannot drive the closed loop

With `model=poster_simulation` the closed loop stalls after the first operation, for two independent reasons
inherent to that structure (see `docs/figures/poster_simulation/coupled.png`):

1. The retrieved activity grows without bound and drives every input neuron of the monitoring network to its
   ceiling, so the summed input (about 7) exceeds the prediction ceiling (`beta_pred = 3`). The difference settles
   at `1 - beta_pred / U` (0.57, matching the simulation), never close to zero, and convergence is not signaled.
2. Even with the monitoring input bounded, Hebbian plasticity during read-out re-binds the active pointer to the
   still-represented previous operation at the hand-over, so only the first operation is ever retrieved.

The poster itself did not couple the components: the three simulations ran separately with hand-aligned stage
durations, which the tag reproduces exactly.

## Figure-level corrections

Compared to the poster outputs, the released figures fix mislabeled legends, overwritten figures of repeated stages,
stacked heatmap colorbars, a wrongly selected "strongest synapse" trace and oversized vector graphics. None of these
affect the simulated dynamics.
