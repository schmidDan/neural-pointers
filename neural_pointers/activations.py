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


def logistic(h, thresh, width):
    """Logistic activation `1 / (1 + exp(-(h - thresh) / width))`, a smooth step from 0 to 1 at `thresh`.

    Args:
        h: Membrane potentials.
        thresh: Potential at which the activation is 0.5.
        width: Width of the transition, the activation is 0.27 and 0.73 at `thresh -/+ width`.
    """
    return 1.0 / (1.0 + np.exp(-(h - thresh) / width))
