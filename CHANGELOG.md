# Changelog

Issue IDs (S1, B1, ...) refer to [docs/known_issues.md](docs/known_issues.md). The code state that reproduces the
BonnBrain 2024 poster is tagged `poster-bonnbrain2024`. Changes of the results can be inspected with
`python tools/compare.py poster-bonnbrain2024 <version>`.

## [Unreleased]

### Changed

- B4 (part): the input layer size and the weight seed of the difference computation model are configurable (`model.n_inputs_per_neuron`, `model.weight_seed`); the defaults are unchanged.
- B11: stage figures are prefixed with the stage index (`stage_<index>_<name>_...`), so repeated stages no longer overwrite each other's figures.
- Restructured the code into one subpackage per model with shared activation functions and stage-wise simulation.
- Migrated to Hydra 1.3 and Python ≥ 3.10. Configurations live in `configs/<model>/`, runners are `run_<model>.py`.
- Every run stores its trajectories in `results.npz` in the Hydra output directory.

### Added

- Reproduction tests against the poster simulations and unit tests.
- Overview figure of all components (`plot_overview.py`), summary quantities (`neural_pointers.summary`) and a
  comparison tool for code versions (`tools/compare.py`).

### Fixed

These fixes do not change the simulated dynamics.

- The runners and tests use the non-interactive Agg backend of matplotlib and no longer fail on systems without a working Tk installation.
- B7: the pointer inputs support a single item.
- B12: heatmap cells are rasterized in SVG figures (60-125 MB → below 200 KB per figure).
- B9: the competitive queueing state heatmaps are drawn into separate figures instead of stacking colorbars.
- B10: removed a mislabeled legend from `stage_<name>_final_syn`.
- B8: `syn_state_strongest_syn_over_time.png` shows the synapse with the largest state.
- The difference computation results were never stored.
- The difference computation experiment crashed with `plot=false` or when no convergence was signaled.
- The competitive queueing model printed at every evaluation of its right-hand side.
- The competitive queueing figures assumed exactly six items.
- A difference computation figure was overwritten by another figure of the same name.
