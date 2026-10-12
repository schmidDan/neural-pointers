"""Reference data for the reproduction tests.

The stored references were generated from the code state of the BonnBrain 2024 poster simulations. Regenerate them
only after intentional changes of the model dynamics:

    python -m tests.reference
"""

import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from neural_pointers.competitive_queueing.experiment import run_experiment as run_cq
from neural_pointers.difference_computation.experiment import run_experiment as run_dc
from neural_pointers.pointers.experiment import run_experiment as run_pt
from neural_pointers.simulation import concatenate_stages

RUNNERS = {
    "pointers": run_pt,
    "competitive_queueing": run_cq,
    "difference_computation": run_dc,
}

# Every SUBSAMPLE-th time step of the trajectories is stored.
SUBSAMPLE = 50


def simulate(experiment, cfg):
    """Runs `experiment` without storing anything and returns the quantities compared against the references."""
    out = RUNNERS[experiment](cfg)
    results = out[1]
    t, h = concatenate_stages(results)
    data = {
        "n_samples": len(t),
        "t": t[::SUBSAMPLE],
        "h": h[::SUBSAMPLE],
        "stage_end_states": np.array([r.h[-1] for r in results]),
    }
    if experiment == "difference_computation":
        data["convergence_times"] = out[2]
    return data


def main(out_dir):
    from tests.conftest import EXPERIMENTS, load_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for experiment in EXPERIMENTS:
        cfg = load_config(experiment)
        (out_dir / f"{experiment}_config.json").write_text(
            json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=1) + "\n"
        )
        np.savez_compressed(
            out_dir / f"{experiment}_reference.npz", **simulate(experiment, cfg)
        )
        print("stored reference for", experiment)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data")
