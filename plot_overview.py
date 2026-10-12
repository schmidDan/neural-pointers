"""Plots the figures in `docs/figures/`. For each preset of the pointer model (`default`, `poster_simulation`):
an overview of the simulations of all three components, arranged like the results panel of the BonnBrain 2024 poster
(`<preset>/overview.png`), and the closed-loop simulation (`<preset>/coupled.png`). In addition, a comparison of the
two pointer models (`pointer_variants.png`) and a schematic of the implemented closed loop (`architecture.png`).

Usage:
    python plot_overview.py
    python plot_overview.py <output path> [<experiment>:<hydra override> ...]

The second form only plots the overview, with overrides applied to the config of one experiment, e.g.
`pointers:model=poster_simulation`.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra import compose, initialize_config_dir

from neural_pointers.competitive_queueing.experiment import run_experiment as run_cq
from neural_pointers.coupled.experiment import run_experiment as run_coupled
from neural_pointers.coupled.visualization import visualize_trial as plot_coupled
from neural_pointers.diagram import (
    plot_architecture,
    plot_broader_architecture,
    plot_pointer_mechanism,
)
from neural_pointers.monitoring.experiment import run_experiment as run_mon
from neural_pointers.pointers.experiment import run_experiment as run_pt
from neural_pointers.simulation import concatenate_stages

ROOT = Path(__file__).resolve().parent
HEATMAP_CMAP = sns.color_palette("rocket", as_cmap=True)
# One color per mental operation (in order of execution), shared by panels I and II.
OP_COLORS = plt.get_cmap("tab10").colors[:6]


def load_config(experiment, overrides=()):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(ROOT / "configs" / experiment)
    ):
        return compose(config_name="config", overrides=list(overrides))


def over_time(getter, h):
    return np.stack([np.ravel(getter(x)) for x in h], axis=1)


def heatmap(fig, ax, data, t, label, ylabel):
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=HEATMAP_CMAP,
        extent=[t[0], t[-1], data.shape[0] - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(range(1, data.shape[0] + 1))
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label=label, fraction=0.08, pad=0.03)


def label_peaks(ax, t, traces, colors):
    """Direct labels with the operation number above the peak of each trace."""
    for k, (trace, color) in enumerate(zip(traces, colors)):
        i = int(np.argmax(trace))
        ax.annotate(
            str(k + 1),
            (t[i], trace[i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.2",
        )


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def main(path, overrides=()):
    per_experiment = {}
    for override in overrides:
        experiment, _, value = override.partition(":")
        per_experiment.setdefault(experiment, []).append(value)
    unknown = set(per_experiment) - {
        "competitive_queueing",
        "pointers",
        "monitoring",
    }
    if unknown:
        raise ValueError("Unknown experiments in overrides: {}".format(sorted(unknown)))
    m_cq, res_cq = run_cq(
        load_config(
            "competitive_queueing", per_experiment.get("competitive_queueing", ())
        )
    )
    m_pt, res_pt = run_pt(load_config("pointers", per_experiment.get("pointers", ())))
    m_mon, res_mon, _ = run_mon(
        load_config("monitoring", per_experiment.get("monitoring", ()))
    )
    t_cq, h_cq = concatenate_stages(res_cq)
    t_pt, h_pt = concatenate_stages(res_pt)
    t_mon, h_mon = concatenate_stages(res_mon)

    # the competitive queueing network executes item 0 first
    y_rates = over_time(m_cq.get_neuron_rate_output_nodes, h_cq)
    y_states = over_time(m_cq.get_neuron_state_output_nodes, h_cq)
    pt_rates = over_time(m_pt.get_neuron_rates, h_pt)

    fig = plt.figure(figsize=(14, 7.5), dpi=150, layout="constrained")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1])

    # I: competitive queueing
    ax = fig.add_subplot(gs[0, 0:2])
    for k, trace in enumerate(y_rates):
        ax.plot(t_cq, trace, color=OP_COLORS[k], lw=1.5, label="mOp {}".format(k + 1))
    label_peaks(ax, t_cq, y_rates, OP_COLORS)
    ax.set_title(
        "I  Scheduling of mental operations\n(competitive queueing)", loc="left"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("output rate")
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="center right")
    style(ax)

    ax = fig.add_subplot(gs[1, 0])
    # input node activity at the end of the final stage, normalized as on the poster
    # planning layer at the end of the first stage, before the first operation is executed
    x = m_cq.get_neuron_state_input_nodes(res_cq[0].h[-1])
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    ax.bar(range(1, len(x) + 1), x_norm, color="0.35", width=0.6)
    ax.set_xticks(range(1, len(x) + 1))
    ax.set_title("Queueing", fontsize=10)
    ax.set_xlabel("mOp")
    ax.set_ylabel("relative activity")
    style(ax)

    ax = fig.add_subplot(gs[1, 1])
    heatmap(fig, ax, y_states, t_cq, "activity", "mOp")
    ax.set_title("Sequencing", fontsize=10)

    # II: neural pointers
    ax = fig.add_subplot(gs[0, 2:4])
    for k, trace in enumerate(pt_rates):
        ax.plot(t_pt, trace, color=OP_COLORS[k], lw=1.5, label="mOp {}".format(k + 1))
    label_peaks(ax, t_pt, pt_rates, OP_COLORS)
    ax.set_title(
        "II  Neural pointers to mental operations\n(binding, then read-out)", loc="left"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("domain rate")
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="best")
    style(ax)

    ax = fig.add_subplot(gs[1, 2])
    binding_end = [r for r in res_pt if r.name == "binding5"][0].h[-1]
    weights = m_pt.get_synaptic_weights(
        binding_end
    ).T  # rows: pointers, columns: domain
    im = ax.imshow(weights, cmap=HEATMAP_CMAP, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(6), range(1, 7))
    ax.set_yticks(range(6), range(1, 7))
    ax.set_xlabel("to (domain)")
    ax.set_ylabel("from (pointer)")
    fig.colorbar(im, ax=ax, label="weight", fraction=0.08, pad=0.03)
    ax.set_title("Pointer formation", fontsize=10)

    ax = fig.add_subplot(gs[1, 3])
    heatmap(fig, ax, pt_rates, t_pt, "activity", "mOp")
    ax.set_title("Retrieval", fontsize=10)

    # III: monitoring
    ax = fig.add_subplot(gs[0, 4:6])
    series = [
        (
            "difference",
            m_mon.get_neuron_rate_difference_nodes,
            dict(color="0.1", lw=1.5),
        ),
        (
            "prediction",
            m_mon.get_neuron_rate_prediction_nodes,
            dict(color="0.55", lw=1.5, ls="--"),
        ),
        (
            "summed input",
            lambda x: np.sum(m_mon.get_neuron_rate_input_nodes(x)),
            dict(color="0.75", lw=2.5),
        ),
        (
            "evidence",
            m_mon.get_neuron_rate_evidence_nodes,
            dict(color="#c0306a", lw=1.5),
        ),
    ]
    for name, getter, kw in series:
        ax.plot(
            t_mon, np.array([np.ravel(getter(x))[0] for x in h_mon]), label=name, **kw
        )
    ax.set_title(
        "III  Monitoring of operation execution\n(difference computation)", loc="left"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("rate")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)

    ax = fig.add_subplot(gs[1, 4:6])
    heatmap(
        fig,
        ax,
        over_time(m_mon.get_neuron_state_input_nodes, h_mon),
        t_mon,
        "activity",
        "input neuron",
    )
    ax.set_title("Systems dynamics monitoring", fontsize=10)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print("stored", path)


def plot_pointer_variants(path):
    """Compares the default pointer model with the structure used for the poster simulations."""
    variants = [
        ("default", "default (poster equations)"),
        ("poster_simulation", "poster simulation"),
    ]
    fig = plt.figure(figsize=(14, 7.5), dpi=150, layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 0.01])
    for row, (preset, title) in enumerate(variants):
        model, results = run_pt(load_config("pointers", ["model=" + preset]))
        t, h = concatenate_stages(results)
        rates = over_time(model.get_neuron_rates, h)

        ax = fig.add_subplot(gs[row, 0])
        for k, trace in enumerate(rates):
            ax.plot(t, trace, color=OP_COLORS[k], lw=1.5, label="mOp {}".format(k + 1))
        label_peaks(ax, t, rates, OP_COLORS)
        ax.set_title(title, loc="left")
        ax.set_xlabel("time")
        ax.set_ylabel("domain rate")
        ax.legend(frameon=False, fontsize=8, ncols=2, loc="best")
        style(ax)

        ax = fig.add_subplot(gs[row, 1])
        binding_end = [r for r in results if r.name == "binding5"][0].h[-1]
        im = ax.imshow(
            model.get_synaptic_weights(binding_end).T,
            cmap=HEATMAP_CMAP,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(range(6), range(1, 7))
        ax.set_yticks(range(6), range(1, 7))
        ax.set_xlabel("to (domain)")
        ax.set_ylabel("from (pointer)")
        fig.colorbar(im, ax=ax, label="weight", fraction=0.08, pad=0.03)
        ax.set_title("Pointer formation", fontsize=10)

        ax = fig.add_subplot(gs[row, 2])
        heatmap(fig, ax, rates, t, "activity", "mOp")
        ax.set_title("Retrieval", fontsize=10)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print("stored", path)


# Presets of the pointer model; each gets a full set of figures in docs/figures/<preset>/.
POINTER_PRESETS = ["default", "poster_simulation"]


def plot_coupled_preset(preset, path):
    """Simulates and plots the closed loop with the pointer model `preset`."""
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="coupled",
            overrides=["pointers/model@models.pointers=" + preset],
        )
    model, results, _ = run_coupled(cfg)
    t_go = [r.t[0] for r in results if r.name == "go"][0]
    plot_coupled(model, results, t_go, path, svg=False)
    print("stored", path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        main(args[0], args[1:])
    else:
        figures = ROOT / "docs" / "figures"
        for preset in POINTER_PRESETS:
            main(figures / preset / "overview.png", ["pointers:model=" + preset])
            plot_coupled_preset(preset, figures / preset / "coupled.png")
        plot_pointer_variants(figures / "pointer_variants.png")
        plot_architecture(figures / "architecture.png")
        print("stored", figures / "architecture.png")
        plot_broader_architecture(figures / "broader_architecture.png")
        print("stored", figures / "broader_architecture.png")
        plot_pointer_mechanism(figures / "pointer_mechanism.png")
        print("stored", figures / "pointer_mechanism.png")
