"""External inputs to the synaptic pointer model."""

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

_RGX_BINDING = re.compile(r"binding(\d+)")
_RGX_WTA = re.compile(r"WTA(\d+)")


class PointerInputs:
    """Inputs for the stages of a pointer binding and read-out protocol.

    For each item `k` a random input pattern and a pointer are generated. Pattern `k` drives domain neuron `k` most
    strongly (1.2) and does not drive neuron `k + 1` at all. Pointer `k` is a one-hot vector at position `k`. The serial
    order is not encoded by the pointers but by the order of the stages (or by the competitive queueing network).

    Stage names select the inputs:

    * `binding<k>`: pattern `k` drives the domain while pointer `k` is active.
    * `WTA<k>`: tonic, uniform drive while pointer `k` is active (read-out of item `k`).
    * `reset`: no drive, no pointer, strong external shunting inhibition.
    * `transition`: no input at all.
    """

    def __init__(self, n_neurons, rng):
        """Generates the input patterns and pointers.

        Args:
            n_neurons: Number of domain neurons (and items).
            rng: `numpy.random.Generator` used to draw the input patterns.
        """
        random_patterns = []
        pointers = []
        for idx_neur in range(n_neurons):
            rnd_pattern = rng.random(n_neurons)
            rnd_pattern[(idx_neur + 1) % n_neurons] = 0.0
            rnd_pattern[idx_neur] = 1.2
            random_patterns.append(rnd_pattern)

            pointer = np.zeros(n_neurons)
            pointer[idx_neur] = 1.0
            pointers.append(pointer)

        self.random_patterns = random_patterns
        self.pointers = pointers

        self.I_input_off = np.zeros(n_neurons)
        self.I_pointers_off = np.zeros(n_neurons)
        self.I_input_inh_off = np.zeros(n_neurons)
        self.I_input_tonic = np.ones(n_neurons) * 0.1
        self.I_input_inh_on = np.ones(n_neurons) * 100.0

    def get_input(self, stage):
        """Returns the model inputs for a simulation stage.

        Args:
            stage: Stage name, see the class documentation.

        Returns:
            tuple: Driving input, pointer activities and external inhibition.

        Raises:
            TypeError: `stage` is not a `str`.
            ValueError: Unrecognized stage.
        """
        if not isinstance(stage, str):
            raise TypeError(
                "Stage needs to be of type str, got: {}".format(type(stage))
            )

        if stage == "reset":
            return (self.I_input_off, self.I_pointers_off, self.I_input_inh_on)
        if stage == "transition":
            return (self.I_input_off, self.I_pointers_off, self.I_input_inh_off)
        match = _RGX_BINDING.fullmatch(stage)
        if match:
            idx = int(match.group(1))
            log.info("%s: binding item %d", stage, idx)
            return (self.random_patterns[idx], self.pointers[idx], self.I_input_inh_off)
        match = _RGX_WTA.fullmatch(stage)
        if match:
            idx = int(match.group(1))
            log.info("%s: reading out item %d", stage, idx)
            return (self.I_input_tonic, self.pointers[idx], self.I_input_inh_off)
        raise ValueError("Unrecognized stage, got: {}".format(stage))
