from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
EXPERIMENTS = ["pointers", "competitive_queueing", "monitoring", "coupled"]


def load_config(experiment, overrides=()):
    """Composes the default config of `experiment` like the runner scripts do."""
    if experiment == "coupled":
        config_dir, config_name = ROOT / "configs", "coupled"
    else:
        config_dir, config_name = ROOT / "configs" / experiment, "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


@pytest.fixture(params=EXPERIMENTS)
def experiment(request):
    return request.param
