# Changelog

## 1.0.0

First standalone release. The code was developed from the simulations of the BonnBrain 2024 poster; that exact state
is tagged `poster-bonnbrain2024`, and [docs/poster_differences.md](docs/poster_differences.md) summarizes how the
released models differ from it. `python tools/compare.py poster-bonnbrain2024 WORKTREE` compares the results of two
code versions.

### Added

* Closed-loop simulation of all three components (`run_coupled.py`): the monitoring network's convergence signal
  advances the competitive queueing network, whose output activates the pointers. All operations are executed in
  order, and the execution duration of an operation is the settling time of its own processing pattern.
* Structural variants of the pointer model (`model.variant`) and the preset `model=poster_simulation` with the
  structure used for the poster simulations.
* Reproduction tests against stored reference trajectories, unit tests, and a GitHub Actions workflow (Python 3.10
  and 3.12).
* Figures generated from the models (`plot_overview.py`): overview and closed loop per pointer preset, variant
  comparison, and schematics of the architecture and the pointer mechanism.
* Summary quantities (`neural_pointers.summary`) and a version comparison tool (`tools/compare.py`).

### Changed

Model revisions relative to the poster simulations; see
[docs/poster_differences.md](docs/poster_differences.md) for the reasons and measured effects.

* The default pointer model follows the equations shown on the poster (shunting excitation and inhibition, pointer
  neurons, purely synaptic pointer weights), bounds the Hebbian coincidence term and forms pointers during binding
  only.
* The monitoring network detects convergence with a neural mechanism (closeness, evidence with self-excitation,
  adaptation) instead of solver-dependent timers, normalizes its recurrent weights per neuron, avoids prediction
  saturation, damps the difference, and processes its input patterns with strong recurrence, which makes settling
  times operation-specific.
* Operation 0 is executed first in all components, and simulation stages lie on one contiguous time grid.

### Fixed

Output and figure handling, without effect on the dynamics: results of every experiment are stored as `results.npz`,
runs no longer fail without a Tk installation or with `plot=false`, repeated stages keep their figures, legends and
heatmaps are corrected, and vector figures no longer embed one shape per time step.
