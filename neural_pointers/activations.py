"""Activation functions mapping membrane potentials to firing rates.

All functions return a new array and never modify their input in place, since they are applied to views of the state
vector that is owned by the ODE solver.
"""

import numpy as np


def identity(h):
    """Linear activation."""
    return h


def relu(h):
    """Rectified linear activation, `max(h, 0)`."""
    r = np.copy(h)
    r[h < 0.0] = 0.0
    return r


def square(h):
    """Squared rectified linear activation, `max(h, 0)**2`."""
    return np.square(relu(h))


def threshold_linear(h, thresh):
    """Linear activation for `h >= thresh`, zero below.

    Args:
        h: Membrane potentials.
        thresh: Activation threshold.
    """
    r = np.copy(h)
    r[h < thresh] = 0.0
    return r


def step(h, thresh):
    """Binary step activation: 0 below `thresh`, 1 above. Values exactly at `thresh` are passed through unchanged.

    Args:
        h: Membrane potentials.
        thresh: Activation threshold.
    """
    r = np.copy(h)
    r[h < thresh] = 0.0
    r[h > thresh] = 1.0
    return r
