"""Runs the synaptic pointer experiment.

Usage:
    python run_pointers.py [hydra overrides, e.g. seed=1 plot=false]

Results and figures are stored in the Hydra output directory `outputs/pointers/<date>/<time>/`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only written to files

import hydra
from hydra.core.hydra_config import HydraConfig

from neural_pointers.pointers.experiment import run_experiment


@hydra.main(version_base="1.3", config_path="configs/pointers", config_name="config")
def main(cfg):
    """Entry point, `cfg` is composed by Hydra from `configs/pointers/`."""
    run_experiment(cfg, out_dir=Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
