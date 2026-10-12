"""Reference data of the default simulations for the reproduction tests.

Regenerate the references only after intentional changes of the dynamics, and document the effect in CHANGELOG.md:

    python -m tests.reference
"""

import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from neural_pointers.competitive_queueing.experiment import run_experiment as run_cq
from neural_pointers.coupled.experiment import run_experiment as run_coupled
from neural_pointers.monitoring.experiment import run_experiment as run_mon
from neural_pointers.pointers.experiment import run_experiment as run_pt
from neural_pointers.simulation import concatenate_stages

RUNNERS = {
    "pointers": run_pt,
    "competitive_queueing": run_cq,
    "monitoring": run_mon,
    "coupled": run_coupled,
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
    if experiment == "monitoring":
        data["convergence_times"] = out[2]
    if experiment == "coupled":
        data["selected"] = np.array(out[2]["selected"])
        data["retrieved"] = np.array(out[2]["retrieved"])
        data["convergence_signals"] = out[2]["convergence_signals"]
        data["execution_durations"] = out[2]["execution_durations"]
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
