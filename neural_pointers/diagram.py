"""Schematics: the implemented closed loop, the embedding architecture and the pointer mechanism
(`docs/figures/architecture.png`, `broader_architecture.png`, `pointer_mechanism.png`).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# colors of the components on the poster
COLORS = {"I": "#1a9e4b", "II": "#7a0f0f", "III": "#1aa3e0", "control": "#555555"}


def _box(ax, x, y, w, h, color, title, lines):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            lw=2,
            ec=color,
            fc="white",
        )
    )
    ax.text(
        x + 0.15, y + h - 0.2, title, color=color, fontsize=12, weight="bold", va="top"
    )
    for k, line in enumerate(lines):
        ax.text(
            x + 0.15,
            y + h - 0.75 - 0.42 * k,
            line,
            fontsize=9.5,
            va="top",
            color="0.15",
        )


def _arrow(
    ax,
    start,
    end,
    color,
    label=None,
    label_xy=None,
    style="-|>",
    ls="-",
    rad=0.0,
    ha="center",
):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=16,
            lw=1.8,
            color=color,
            ls=ls,
            connectionstyle="arc3,rad={}".format(rad),
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        ax.text(*label_xy, label, fontsize=9, color=color, ha=ha, va="center")


def plot_architecture(path):
    """Draws the components of the closed-loop simulation and their interactions.

    Args:
        path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(13, 7.6), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    _box(
        ax,
        0.3,
        4.1,
        5.4,
        3.0,
        COLORS["I"],
        "I  Scheduling: competitive queueing",
        [
            "planning nodes x: queue of operations",
            "   (graded input, operation 1 strongest)",
            "choice nodes y: compete while aroused (J > 0),",
            "   the winner y_k is the active operation",
            "feedback nodes w: suppress executed operations",
        ],
    )
    _box(
        ax,
        7.3,
        4.1,
        5.4,
        3.0,
        COLORS["II"],
        "II  Neural pointers to operations",
        [
            "pointer neurons p_j",
            "Hebbian synapses s_ij (formed during binding)",
            "domain (storage) neurons x_i: shunting",
            "   winner-take-all, modulated by bound pointers",
            "retrieved operation: domain rates r",
        ],
    )
    _box(
        ax,
        7.3,
        0.3,
        5.4,
        3.0,
        COLORS["III"],
        "III  Monitoring: difference computation",
        [
            "database u, v: recurrent processing of the pattern P r,",
            "   settling time specific to the operation",
            "difference d = input - prediction p",
            "evidence e: accumulates while d ≈ 0,",
            "   all-or-none convergence signal r_conv",
        ],
    )
    _box(
        ax,
        0.3,
        0.3,
        5.4,
        3.0,
        COLORS["control"],
        "Control",
        [
            "arousal J: driven by r_conv",
            "   -> the next operation is selected",
            "reset R: follows J with a delay (global inhibition)",
            "   -> clears domain and database for the hand-over",
            "go: external arousal starts the execution",
        ],
    )

    # I -> II: the selected operation activates its pointer
    _arrow(
        ax,
        (5.75, 5.6),
        (7.25, 5.6),
        COLORS["I"],
        "pointer input\nmin(g s(y_k), 1)",
        (6.5, 6.25),
    )
    # II -> III: the retrieved operation drives the processing
    _arrow(
        ax,
        (10.0, 4.05),
        (10.0, 3.35),
        COLORS["II"],
        "processing pattern P r",
        (10.15, 3.7),
        ha="left",
    )
    # III -> control: convergence signal
    _arrow(
        ax,
        (7.25, 1.8),
        (5.75, 1.8),
        COLORS["III"],
        "convergence\nsignal r_conv",
        (6.5, 2.45),
    )
    # control -> I: arousal
    _arrow(
        ax,
        (3.0, 3.35),
        (3.0, 4.05),
        COLORS["control"],
        "arousal J",
        (3.15, 3.7),
        ha="left",
    )
    # control -> II and III: delayed global reset
    _arrow(ax, (5.75, 1.0), (7.25, 0.9), COLORS["control"], ls="--")
    _arrow(
        ax,
        (5.75, 2.9),
        (7.25, 4.5),
        COLORS["control"],
        "reset R",
        (6.25, 3.85),
        ls="--",
    )

    ax.text(
        6.5,
        7.45,
        "Binding phase (scripted): operation patterns drive the domain while their pointers are active  →  Hebbian synapses",
        ha="center",
        fontsize=9.5,
        color=COLORS["II"],
        style="italic",
    )
    _arrow(ax, (10.0, 7.32), (10.0, 7.12), COLORS["II"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _mini_box(ax, x, y, w, h, color, text, dashed=False, fontsize=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            lw=1.5,
            ec=color,
            fc="white",
            ls="--" if dashed else "-",
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="0.15",
    )


def plot_broader_architecture(path):
    """Redraws the architecture presented on the poster and marks what this repository implements.

    Args:
        path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    grey = "0.55"

    # scheduler / planner
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 3.9),
            4.9,
            2.9,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            lw=2,
            ec=COLORS["I"],
            fc="white",
        )
    )
    ax.text(
        0.6,
        6.6,
        "Scheduler / planner",
        color=COLORS["I"],
        fontsize=12,
        weight="bold",
        va="top",
    )
    _mini_box(
        ax,
        0.8,
        5.35,
        4.1,
        0.75,
        COLORS["I"],
        "queue of operations (competitive queueing)",
    )
    _mini_box(
        ax, 0.8, 4.15, 4.1, 0.75, COLORS["I"], "active operation (program interface)"
    )
    _arrow(ax, (2.85, 5.3), (2.85, 4.95), COLORS["I"], "I", (3.05, 5.12), ha="left")

    # program storage
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 0.4),
            4.9,
            2.3,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            lw=2,
            ec=COLORS["II"],
            fc="white",
        )
    )
    ax.text(
        0.6,
        2.5,
        "Program storage",
        color=COLORS["II"],
        fontsize=12,
        weight="bold",
        va="top",
    )
    for k in range(6):
        filled = k == 2
        ax.add_patch(
            FancyBboxPatch(
                (0.95 + 0.63 * k, 0.75),
                0.5,
                1.1,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                lw=1.2,
                ec=COLORS["II"],
                fc=COLORS["II"] if filled else "white",
                alpha=0.85 if filled else 1.0,
            )
        )
    ax.text(
        2.75,
        0.56,
        "stored operation patterns",
        fontsize=9,
        color="0.15",
        ha="center",
    )

    # pointers between interface and storage
    _arrow(ax, (2.6, 3.85), (2.6, 2.75), COLORS["II"])
    _arrow(ax, (3.1, 2.75), (3.1, 3.85), COLORS["II"])
    ax.text(
        3.3,
        3.3,
        "neural pointers  II",
        fontsize=10,
        color=COLORS["II"],
        ha="left",
        va="center",
        weight="bold",
    )

    # task controller
    ax.add_patch(
        FancyBboxPatch(
            (7.7, 3.9),
            4.9,
            2.9,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            lw=2,
            ec=COLORS["III"],
            fc="white",
        )
    )
    ax.text(
        7.9,
        6.6,
        "Task controller",
        color=COLORS["III"],
        fontsize=12,
        weight="bold",
        va="top",
    )
    _mini_box(
        ax, 8.1, 5.35, 4.1, 0.75, COLORS["III"], "execution monitoring (convergence)"
    )
    _mini_box(
        ax, 8.1, 4.15, 4.1, 0.75, grey, "attentional control (not modeled)", dashed=True
    )
    ax.text(12.35, 5.95, "III", fontsize=11, color=COLORS["III"], weight="bold")

    # visual processor
    ax.add_patch(
        FancyBboxPatch(
            (7.7, 0.4),
            4.9,
            2.3,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            lw=2,
            ec=grey,
            fc="white",
            ls="--",
        )
    )
    ax.text(
        7.9, 2.5, "Visual processor", color=grey, fontsize=12, weight="bold", va="top"
    )
    ax.text(
        10.3,
        1.45,
        "evolves under the active operation\n(not modeled: operations are\nabstract activity patterns)",
        fontsize=9.5,
        color="0.35",
        ha="center",
        va="center",
    )

    # arrows between the modules
    _arrow(ax, (5.35, 6.0), (7.65, 6.0), COLORS["I"], "selected operation", (6.5, 6.25))
    _arrow(
        ax,
        (7.65, 5.6),
        (5.35, 5.6),
        COLORS["III"],
        "arousal:\nadvance the sequence",
        (6.5, 5.18),
    )
    _arrow(
        ax,
        (10.7, 3.85),
        (10.7, 2.75),
        grey,
        "attentional control",
        (10.9, 3.3),
        ha="left",
        ls="--",
    )
    _arrow(
        ax,
        (9.3, 2.75),
        (9.3, 3.85),
        COLORS["III"],
        "system\ndynamics",
        (9.15, 3.3),
        ha="right",
    )
    _arrow(ax, (5.35, 1.5), (7.65, 1.5), COLORS["II"], "retrieved pattern", (6.5, 1.75))

    ax.text(
        6.5,
        0.08,
        "colored: implemented in this repository (I competitive queueing, II neural pointers, "
        "III difference computation)  —  grey, dashed: not modeled",
        ha="center",
        fontsize=9,
        color="0.35",
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _inh_line(ax, start, end, color, lw=1.6):
    """Line with a filled circle at the end (inhibitory connection)."""
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-", lw=lw, color=color, shrinkA=2, shrinkB=7
        )
    )
    ax.plot([end[0]], [end[1]], marker="o", ms=6, color=color, zorder=5)


def plot_pointer_mechanism(path):
    """Circuit and equations of the default pointer model.

    Args:
        path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=150)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    red, grey, dark = COLORS["II"], "0.65", "0.15"

    px = [1.3, 2.5, 3.7]
    py, dy = 5.4, 2.3

    # pointer input
    for x in px:
        _arrow(ax, (x, 6.5), (x, py + 0.35), dark)
    ax.text(
        2.5,
        6.8,
        "pointer input $I^c_j$ (binding: external; execution: from the scheduler)",
        fontsize=10,
        ha="center",
        color=dark,
    )

    # all-to-all synapses, one highlighted
    for i, xs in enumerate(px):
        for j, xt in enumerate(px):
            hl = i == 1 and j == 1
            ax.plot(
                [xs, xt],
                [py - 0.3, dy + 0.3],
                color=red if hl else grey,
                lw=2.2 if hl else 0.9,
                zorder=2 if hl else 1,
            )
    ax.plot(
        [2.5], [(py + dy) / 2], marker="o", ms=9, mfc="white", mec=red, mew=2, zorder=3
    )
    ax.text(
        2.75,
        (py + dy) / 2,
        "$s_{ij}$  Hebbian synapse\n(plastic during binding)",
        fontsize=9.5,
        color=red,
        va="center",
    )

    # pointer and domain neurons
    for k, x in enumerate(px):
        ax.plot(
            [x],
            [py],
            marker="o",
            ms=22,
            mfc=red if k == 1 else "white",
            mec=red,
            mew=2,
            zorder=4,
        )
        ax.plot(
            [x],
            [dy],
            marker="o",
            ms=22,
            mfc="#f2e3e3" if k == 1 else "white",
            mec=dark,
            mew=2,
            zorder=4,
        )
    ax.text(
        0.5,
        py,
        "pointer\nneurons $y_j$",
        fontsize=10,
        ha="center",
        va="center",
        color=dark,
    )
    ax.text(
        0.5,
        dy,
        "domain\nneurons $x_i$",
        fontsize=10,
        ha="center",
        va="center",
        color=dark,
    )

    # pooled inhibition
    pool = (5.0, dy + 0.9)
    ax.plot([pool[0]], [pool[1]], marker="o", ms=14, color=dark, zorder=4)
    ax.text(
        pool[0] + 0.22,
        pool[1] + 0.05,
        "pooled\ninhibition",
        fontsize=9.5,
        va="center",
        color=dark,
    )
    for x in px:
        _arrow(
            ax, (x + 0.14, dy + 0.3), (pool[0] - 0.1, pool[1] - 0.1), grey, style="-|>"
        )
    for x in px:
        _inh_line(ax, (pool[0], pool[1] - 0.2), (x + 0.24, dy + 0.14), dark, lw=1.1)

    # driving input and reset
    for x in px:
        _arrow(ax, (x, 0.9), (x, dy - 0.35), dark)
    ax.text(
        2.5,
        0.55,
        "driving input $I_i$ (binding: operation pattern; execution: tonic)",
        fontsize=10,
        ha="center",
        color=dark,
    )
    _inh_line(ax, (5.6, 1.1), (4.0, dy - 0.2), grey)
    ax.text(5.72, 1.08, "reset $I^{inh}_i$", fontsize=10, ha="left", color="0.4")

    # equations of the default model
    ex = 6.6
    lines = [
        ("Pointer neurons", None),
        (None, r"$\tau_p\,\dot{y}_j = -y_j + I^c_j, \qquad p_j = [y_j]_+$"),
        ("Hebbian synapses", None),
        (
            None,
            r"$\dot{s}_{ij} = (s_{max} - s_{ij})\,c_{ij}/\tau_{rise} + (s_{min} - s_{ij})\,(1 - c_{ij})/\tau_{decay}$",
        ),
        (None, r"$c_{ij} = \mathrm{min}(p_j\,r_i,\; 1)$"),
        ("Domain neurons (shunting excitation and inhibition)", None),
        (
            None,
            r"$\tau\,\dot{x}_i = -\alpha x_i + (\beta - x_i)\,I_i\,(1 + \delta\,m_i) - (\eta + \gamma x_i)\,pool_i - \eta\,x_i\,I^{inh}_i$",
        ),
        (None, r"$m_i = {\sum}_j\, s_{ij}\,p_j, \qquad r_i = [x_i]_+$"),
        (None, r"$pool_i = (1 - k_{inh} - \kappa)\,r_i + k_{inh} {\sum}_j\, r_j$"),
    ]
    yk = 6.6
    for header, formula in lines:
        if header:
            ax.text(ex, yk, header, fontsize=10.5, weight="bold", color=dark)
            yk -= 0.44
        else:
            ax.text(ex + 0.25, yk, formula, fontsize=11, color=dark)
            yk -= 0.56
    yk -= 0.1
    ax.text(
        ex, yk, "Modes (default parameters)", fontsize=10.5, weight="bold", color=dark
    )
    yk -= 0.52
    rows = [
        ("", r"$\delta$", r"$\eta$", r"$\kappa = k_{inh}$", r"$\tau_{rise}$"),
        ("binding", "0.1", "1", "0", "1"),
        ("read-out", "10", "10", "1", r"$\infty$ (frozen)"),
    ]
    cols = [ex, ex + 1.7, ex + 2.9, ex + 4.0, ex + 5.6]
    for row in rows:
        for x, cell in zip(cols, row):
            ax.text(x, yk, cell, fontsize=10, color=dark)
        yk -= 0.48
    ax.text(
        ex,
        yk - 0.1,
        r"shared: $\alpha = 0$, $\beta = 1$, $\gamma = 1$, $\tau = 1$, $\tau_p = 0.2$,"
        r" $s_{min} = 0$, $s_{max} = 1$, $\tau_{decay} = 200$",
        fontsize=9.5,
        color="0.35",
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
