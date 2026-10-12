"""Runs the difference computation experiment.

Usage:
    python run_difference_computation.py [hydra overrides, e.g. seed=1 plot=false]

Results and figures are stored in the Hydra output directory `outputs/difference_computation/<date>/<time>/`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only written to files

import hydra
from hydra.core.hydra_config import HydraConfig

from neural_pointers.difference_computation.experiment import run_experiment


@hydra.main(
    version_base="1.3",
    config_path="configs/difference_computation",
    config_name="config",
)
def main(cfg):
    """Entry point, `cfg` is composed by Hydra from `configs/difference_computation/`."""
    run_experiment(cfg, out_dir=Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
