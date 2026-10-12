"""Visualization of the closed-loop simulation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_pointers.simulation import concatenate_stages

OP_COLORS = plt.get_cmap("tab10").colors


def visualize_trial(model, results, t_go, path, svg=True):
    """Plots scheduling, retrieval, monitoring and control signals of the closed loop over time.

    Args:
        model: The simulated `CoupledModel`.
        results: List of `StageResult`s.
        t_go: Onset of the execution; the binding phase before is shaded.
        path: Output file path.
        svg: Whether to store an `.svg` version next to it.
    """
    t, h = concatenate_stages(results)
    parts = [model.split(x) for x in h]
    y = np.array([model.cq.get_neuron_rate_output_nodes(p[0]) for p in parts])
    r = np.array([model.pointers.get_neuron_rates(p[1]) for p in parts])
    d = np.array(
        [model.monitoring.get_neuron_rate_difference_nodes(p[2])[0] for p in parts]
    )
    signal = np.array([model.convergence_signal(p[2]) for p in parts])
    J = np.array([p[3] for p in parts])
    R = np.array([p[4] for p in parts])

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 9), dpi=150, sharex=True, layout="constrained"
    )
    for k in range(y.shape[1]):
        axes[0].plot(
            t, y[:, k], color=OP_COLORS[k], lw=1.5, label="mOp {}".format(k + 1)
        )
        axes[1].plot(t, r[:, k], color=OP_COLORS[k], lw=1.5)
    axes[0].set_ylabel("output rate")
    axes[0].set_title("Scheduling (competitive queueing)", loc="left")
    axes[0].legend(frameon=False, fontsize=8, ncols=6, loc="upper left")
    axes[1].set_ylabel("domain rate")
    axes[1].set_title("Retrieval (neural pointers)", loc="left")

    axes[2].plot(t, d, color="0.1", lw=1.5, label="difference")
    axes[2].plot(t, signal, color="#c0306a", lw=1.5, label="convergence signal")
    axes[2].set_ylabel("rate")
    axes[2].set_title("Monitoring (difference computation)", loc="left")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    on = signal >= 0.5
    if not np.any(on[1:] & ~on[:-1] & (t[1:] > t_go)):
        axes[2].text(
            0.55,
            0.5,
            "no convergence signal — the execution stalls",
            transform=axes[2].transAxes,
            ha="center",
            fontsize=10,
            color="#c0306a",
        )

    axes[3].plot(t, J, color="0.1", lw=1.5, label="arousal J")
    axes[3].plot(t, R, color="0.6", lw=1.5, ls="--", label="reset R")
    axes[3].set_ylim(0, max(1.0, 1.1 * float(np.max(J)), 1.1 * float(np.max(R))))
    axes[3].set_ylabel("activity")
    axes[3].set_title("Control", loc="left")
    axes[3].set_xlabel("time")
    axes[3].legend(frameon=False, fontsize=8, loc="upper left")

    for ax in axes:
        ax.axvspan(t[0], t_go, color="0.93", lw=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[3].text(
        t_go / 2,
        0.5,
        "binding",
        transform=axes[3].get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=8,
        color="0.4",
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if svg:
        fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
