"""External inputs to the competitive queueing model."""

import numpy as np


class CompetitiveQueueingInputs:
    """Inputs for the stages of a competitive queueing protocol.

    The input pattern is a linear gradient from 1.0 to 0.1, i.e. item 0 has the strongest activation and is executed
    first.

    Stage names select the inputs:

    * `aroused`: input pattern on, arousal on (the output layer selects the next item).
    * `numbed`: input pattern on, arousal off (the selection is held).
    * `standby`: input pattern and arousal off.
    """

    def __init__(self, n_neurons, rng):
        """Generates the inputs.

        Args:
            n_neurons: Number of items.
            rng: Unused, kept for a uniform interface across experiments.
        """
        self.I_input_off = np.zeros(n_neurons)
        self.I_input_on = np.linspace(1.0, 0.1, n_neurons)

        self.I_arousal_off = 0.0
        self.I_arousal_on = 10.0

    def get_input(self, stage):
        """Returns the model inputs for a simulation stage.

        Args:
            stage: Stage name, see the class documentation.

        Returns:
            tuple: Input pattern and arousal signal.

        Raises:
            TypeError: `stage` is not a `str`.
            ValueError: Unrecognized stage.
        """
        if not isinstance(stage, str):
            raise TypeError(
                "Stage needs to be of type str, got: {}".format(type(stage))
            )
        if stage == "aroused":
            return (self.I_input_on, self.I_arousal_on)
        if stage == "standby":
            return (self.I_input_off, self.I_arousal_off)
        if stage == "numbed":
            return (self.I_input_on, self.I_arousal_off)
        raise ValueError("Unrecognized stage, got: {}".format(stage))
