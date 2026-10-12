"""External inputs to the monitoring model."""

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

N_PATTERNS = 10

_RGX_PATTERN = re.compile(r"pattern(\d+)")


class MonitoringInputs:
    """Inputs for the stages of a monitoring protocol.

    `N_PATTERNS` input patterns are drawn uniformly from [0, 1). Stage names select the inputs:

    * `pattern<k>`: input pattern `k`.
    * `pattern_zeros`: no input.
    """

    def __init__(self, n_inputs, rng):
        """Generates the input patterns.

        Args:
            n_inputs: Number of input neurons.
            rng: `numpy.random.Generator` used to draw the input patterns.
        """
        self.patterns = [rng.uniform(0.0, 1.0, n_inputs) for _ in range(N_PATTERNS)]
        self.pattern_zeros = np.zeros(n_inputs)

    def get_input(self, stage):
        """Returns the model inputs for a simulation stage.

        Args:
            stage: Stage name, see the class documentation.

        Returns:
            tuple: The input pattern.

        Raises:
            TypeError: `stage` is not a `str`.
            ValueError: Unrecognized stage.
        """
        if not isinstance(stage, str):
            raise TypeError(
                "Stage needs to be of type str, got: {}".format(type(stage))
            )
        if stage == "pattern_zeros":
            return (self.pattern_zeros,)
        match = _RGX_PATTERN.fullmatch(stage)
        if match:
            idx = int(match.group(1))
            log.info("%s: input pattern %d", stage, idx)
            return (self.patterns[idx],)
        raise ValueError("Unrecognized stage, got: {}".format(stage))
