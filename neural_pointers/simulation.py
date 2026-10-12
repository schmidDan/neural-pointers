"""Stage-wise numerical integration shared by all experiments.

A simulation protocol is a sequence of stages. Each stage has a `name` (which selects the external input) and a
`duration`; the final state of one stage is the initial state of the next one.
"""

from collections import namedtuple

import numpy as np
from scipy.integrate import odeint

StageResult = namedtuple("StageResult", ["name", "t", "h"])


def simulate_stages(model, get_input, stages, dt, h_0, on_stage_start=None):
    """Integrates `model` over a sequence of stages.

    Stage `k` starts at `T_k`, the sum of the durations of all previous stages, and covers `[T_k, T_k + duration)` on
    the global grid `n * dt`. Each stage is integrated up to its end `T_k + duration`, which is the initial state of
    the next stage and is not part of the stage's samples. Concatenating all stages therefore yields the contiguous
    grid `0, dt, 2 dt, ...` without duplicated time points.

    Args:
        model: Callable `model(h, t, *inputs)` returning the temporal derivative of the state `h`.
        get_input: Callable mapping a stage name to the tuple of additional arguments passed to `model`.
        stages: Sequence of stage configurations, each with attributes `name` and `duration`.
        dt: Sampling interval of the returned trajectories. Stage durations must be multiples of `dt`.
        h_0: Initial state.
        on_stage_start: Optional callable invoked with the stage configuration before the stage is integrated, e.g.
            to switch the operation mode of the model.

    Returns:
        list: One `StageResult(name, t, h)` per stage, where `h` has shape `(len(t), len(h_0))`.

    Raises:
        ValueError: A stage duration is not a positive multiple of `dt`.
    """
    step = 0
    h_old = h_0
    results = []
    for stage in stages:
        n_steps = int(round(stage.duration / dt))
        if n_steps < 1 or not np.isclose(
            n_steps * dt, stage.duration, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "Duration of stage {} is not a positive multiple of dt={}, got: {}".format(
                    stage.name, dt, stage.duration
                )
            )
        t = (step + np.arange(n_steps + 1)) * dt
        if on_stage_start is not None:
            on_stage_start(stage)
        h = odeint(model, h_old, t, args=get_input(stage.name))
        h_old = h[-1]
        results.append(StageResult(stage.name, t[:-1], h[:-1]))
        step += n_steps
    return results


def stage_label(index, name):
    """Unique label of the `index`-th stage, e.g. for file names (stage names may repeat)."""
    return "{:02d}_{}".format(index, name)


def concatenate_stages(results):
    """Concatenates per-stage results into contiguous arrays.

    Args:
        results: List of `StageResult`s as returned by `simulate_stages`.

    Returns:
        tuple: Time points `t` and states `h` of the entire simulation.
    """
    t = np.concatenate([r.t for r in results])
    h = np.concatenate([r.h for r in results], axis=0)
    return t, h


def save_results(path, results, **extra):
    """Stores the simulated trajectories as a compressed `.npz` archive.

    The archive contains the concatenated time points `t` and states `h`, the `stage_names` and the number of samples
    per stage (`stage_lengths`), plus any additional arrays passed as keyword arguments.

    Args:
        path: Output file path.
        results: List of `StageResult`s as returned by `simulate_stages`.
        **extra: Additional arrays to store.
    """
    t, h = concatenate_stages(results)
    np.savez_compressed(
        path,
        t=t,
        h=h,
        stage_names=np.array([r.name for r in results]),
        stage_lengths=np.array([len(r.t) for r in results]),
        **extra,
    )
