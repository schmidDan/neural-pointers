"""Plots an overview of the default simulations of all three components, arranged like the results panel of the
BonnBrain 2024 poster.

Usage:
    python plot_overview.py [output path, default: docs/figures/overview.png]
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
from neural_pointers.difference_computation.experiment import run_experiment as run_dc
from neural_pointers.pointers.experiment import run_experiment as run_pt
from neural_pointers.simulation import concatenate_stages

ROOT = Path(__file__).resolve().parent
HEATMAP_CMAP = sns.color_palette("rocket", as_cmap=True)
# One color per mental operation (in order of execution), shared by panels I and II.
OP_COLORS = plt.get_cmap("tab10").colors[:6]


def load_config(experiment):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(ROOT / "configs" / experiment)
    ):
        return compose(config_name="config")


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


def main(path):
    m_cq, res_cq = run_cq(load_config("competitive_queueing"))
    m_pt, res_pt = run_pt(load_config("pointers"))
    m_dc, res_dc, _ = run_dc(load_config("difference_computation"))
    t_cq, h_cq = concatenate_stages(res_cq)
    t_pt, h_pt = concatenate_stages(res_pt)
    t_dc, h_dc = concatenate_stages(res_dc)

    # the competitive queueing network executes the item with the highest index first
    cq_order = list(range(m_cq.n_neurons - 1, -1, -1))
    y_rates = over_time(m_cq.get_neuron_rate_output_nodes, h_cq)[cq_order]
    y_states = over_time(m_cq.get_neuron_state_output_nodes, h_cq)[cq_order]
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
    x = m_cq.get_neuron_state_input_nodes(res_cq[-1].h[-1])
    x_norm = (x - np.min(x)) / np.abs(np.max(x))
    ax.bar(range(1, len(x) + 1), x_norm, color="0.35", width=0.6)
    ax.set_xticks(range(1, len(x) + 1))
    ax.set_title("Queueing", fontsize=10)
    ax.set_xlabel("item")
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
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
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
        ("difference", m_dc.get_neuron_rate_exc_nodes, dict(color="0.1", lw=1.5)),
        (
            "prediction",
            m_dc.get_neuron_rate_inh_nodes,
            dict(color="0.55", lw=1.5, ls="--"),
        ),
        (
            "summed input",
            lambda x: np.sum(m_dc.get_neuron_rate_inp_nodes(x)),
            dict(color="0.75", lw=2.5),
        ),
        ("evidence", m_dc.get_neuron_rate_evdc_nodes, dict(color="#c0306a", lw=1.5)),
    ]
    for name, getter, kw in series:
        ax.plot(
            t_dc, np.array([np.ravel(getter(x))[0] for x in h_dc]), label=name, **kw
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
        over_time(m_dc.get_neuron_state_inp_nodes, h_dc),
        t_dc,
        "activity",
        "input neuron",
    )
    ax.set_title("Systems dynamics monitoring", fontsize=10)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print("stored", path)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "figures" / "overview.png"
    )
