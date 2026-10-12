"""Some custom inputs to the model. To be extended and polished...
"""
import numpy as np
import re

class Inputs:
    """Input class containing different kinds of potential inputs to the model."""

    def __init__(self, n_neurons, rng):

        random_patterns = []
        pointers = []
        for idx_neur in range(n_neurons):
            rnd_pattern = rng.random(n_neurons)
            # set one input to be exactly zero, so that pointer accumulation shouldn't happen for this one
            rnd_pattern[(idx_neur+1) % n_neurons] = 0.0
            rnd_pattern[idx_neur] = 1.2
            random_patterns.append(rnd_pattern)
                
            # define inputs from pointers modulating the domain neurons
            pointer =  np.zeros(n_neurons)
            # choose one pointer to be active
            pointer = np.zeros(n_neurons)
            # make pointer strength dependent on sequence position (to encode the sequential order)
#            pointer[idx_neur] = 1.0
            pointer[idx_neur] = 1.0 - 0.5 * (idx_neur/(n_neurons-1))
            ##pointer *= 10.0
            pointers.append(pointer)
            
        
        """Setup for the inputs."""
        # define inputs driving the domain neurons
        self.I_input_off = np.zeros(n_neurons)
        self.I_pointers_off = np.zeros(n_neurons)
        self.I_input_inh_off = np.zeros(n_neurons)
        self.I_input_tonic = np.ones(n_neurons) * 0.1 # WTA
        #self.I_input_tonic = np.ones(n_neurons) * 0.01
#        self.I_input_tonic = np.ones(n_neurons) * 1.0
        self.I_input_inh_on = np.ones(n_neurons) * 100.0
        
        # set defaults
        self.random_patterns = random_patterns
        self.pointers = pointers


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

        
        # cf. https://stackoverflow.com/a/9012064
        rgx_bind = re.compile(r'binding\d+')
        rgx_wta = re.compile(r'WTA\d+')
        
        if stage == "binding":
            I_input = self.random_patterns[0]
            I_pointers = self.pointers[4]
            I_input_inh = self.I_input_inh_off
        elif stage == "transition":
            I_input = self.I_input_off
            I_pointers = self.I_pointers_off
            I_input_inh = self.I_input_inh_off
        elif stage == "WTA":
            I_input = self.I_input_tonic
            I_pointers = self.pointers[4]
            I_input_inh = self.I_input_inh_off
        elif rgx_bind.search(stage):
            # cf. https://stackoverflow.com/a/4667014
            idx = re.search(r'binding(\d+)', stage).group(1)
            idx = int(idx)
            print("{}: {}".format(stage, idx))
            I_input = self.random_patterns[idx]
            I_pointers = self.pointers[idx]
            I_input_inh = self.I_input_inh_off
        elif rgx_wta.search(stage):
            idx = re.search(r'WTA(\d+)', stage).group(1)
            idx = int(idx)
            print("{}: {}".format(stage, idx))
            I_input = self.I_input_tonic
###            I_pointers = self.pointers[idx]
            I_pointers = self.pointers[idx]
            I_input_inh = self.I_input_inh_off
        elif stage == "reset":
            I_input = self.I_input_off
            I_pointers = self.I_pointers_off
            I_input_inh = self.I_input_inh_on
        else:
            raise ValueError("Unrecognized stage, got: {}".format(stage))
        return (I_input, I_pointers, I_input_inh)
