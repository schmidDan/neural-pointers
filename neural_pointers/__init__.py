"""Neurodynamical models for sequencing and executing visual-mental operations.

The package contains three rate-based models that are simulated stage-wise with `scipy.integrate.odeint`:

* `neural_pointers.pointers`: synaptic neural pointers (binding and winner-take-all read-out).
* `neural_pointers.competitive_queueing`: competitive queueing for serial execution of a graded activity pattern.
* `neural_pointers.difference_computation`: progress monitoring by detecting convergence of an input representation.
"""

__version__ = "1.0.0.dev0"
