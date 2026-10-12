"""Some custom inputs to the model. To be extended and polished...
"""
import numpy as np

np.random.seed(42)


class Inputs:
    """Input class containing different kinds of potential inputs to the model."""

    def __init__(self, n_neurons, rng):
        """Setup for the inputs."""
        # define inputs driving the domain neurons
        self.I_input_off = np.zeros(n_neurons)
        #self.I_input_tonic = np.ones(n_neurons)
        self.I_input_tonic = np.ones(n_neurons) * 0.1
        self.I_input_on = rng.random(n_neurons)
        # set one input to be exactly zero, so that pointer accumulation shouldn't happen for this one
        self.I_input_on[4] = 0.0

        # define inputs from pointers modulating the domain neurons
        self.I_pointers_off = np.zeros(n_neurons)
        # choose one pointer to be active
        self.I_pointers_on = np.zeros(n_neurons)
        self.I_pointers_on[0] = 1.0

    def get_input(self, stage):
        """Method to get model input for certain simulation stage `stage` and a popuation of `n_neurons`.

        Args:
            stage: Stage for which the input should be computed.
            n_neurons: Number of neurons for which the input should be computed.

        Returns:
            list: Tuple of inputs with first element being the driving input to the domain neurons and the second
            element being the input from the pointers.

        Raises:
            TypeError: The input argument was not of type `str`.
            ValueError: Unrecognized stage.
        """
        if not isinstance(stage, str):
            raise TypeError(
                "Stage needs to be of type str, got: {}".format(type(stage))
            )
        if stage == "binding":
            I_input = self.I_input_on
            I_pointers = self.I_pointers_on
        elif stage == "transition":
            I_input = self.I_input_off
            I_pointers = self.I_pointers_off
        elif stage == "WTA":
            I_input = self.I_input_tonic
            I_pointers = self.I_pointers_on
        else:
            raise ValueError("Unrecognized stage, got: {}".format(stage))
        return (I_input, I_pointers)
