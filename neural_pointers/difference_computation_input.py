"""Some custom inputs to the model. To be extended and polished...
"""
import numpy as np
import re


class DifferenceComputationInputs:
    """Input class containing different kinds of potential inputs to the model."""

    def __init__(self, n_neurons, rng):
        """Setup for the inputs."""
        # define inputs driving the domain neurons
        patterns = []
        for _ in range(0,10):
            pattern0 = rng.uniform(0.0, 1.0, n_neurons*7)
            patterns.append(pattern0)
        self.patterns = patterns
        self.pattern_zeros = np.zeros(n_neurons*7)

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

        
        rgx = re.compile(r'pattern\d+')
        if stage == "pattern_zeros":
            I_input = self.pattern_zeros
        elif rgx.search(stage):
            idx = re.search(r'pattern(\d+)', stage).group(1)
            idx = int(idx)
            print("{}: {}".format(stage, idx))
            I_input = self.patterns[idx]
        else:
            raise ValueError("Unrecognized stage, got: {}".format(stage))
        return (I_input,)
