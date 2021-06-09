"""Components for a dynamical model of neural pointers.
"""

from collections import namedtuple
from enum import Enum, auto
import numpy as np


def relu(h):
    r = h
    r[h < 0.0] = 0.0
    return r


NeuronParams = namedtuple(
    "NeuronParams", ["alpha", "beta", "delta", "eta", "gamma", "tau"]
)
SynParams = namedtuple("SynParams", ["syn_min", "syn_max", "tau_rise", "tau_decay"])
ConnectionParams = namedtuple("ConnectionParams", ["kappa"])
ModelParams = namedtuple(
    "ModelParams", ["NeuronParams", "SynParams", "ConnectionParams"]
)


class DOMAIN_MODE(Enum):
    """Enum to encode the different modes of operation of the `SynapticPointerModel`."""

    BINDING = auto()
    WTA = auto()


class SynapticPointerModel:
    """A computational interpretation of the synaptic pointer proposal outlined in "Zylberberg et al. (2013). A
    neuronal device for the control of multi-step computations. Papers in Physics 5, 050006".

    The `SynapticPointerModel` so far implements the system dynamics exhibited by a population of "domain" neurons and
    by their synapses which relegate input from a "pointer" source to the domain neurons. The state space spanned by
    these components (i.e. neurons and synapses) is concatenated along axis 0 to form an overall state vector. The
    model domain neurons can be forced into two distinct modes (`DOMAIN_MODE`) by calling `set_mode`. Initially the
    model is set to `DOMAIN_MODE.BINDING`. Neuron activities are assumed to be computed from their potentials based on a
    `relu` mapping.
    """

    def __init__(self, n_neurons, config):
        """Model initialization. The ODE parameters necessary to perform computations in either `DOMAIN_MODE` are each
        read from a common `hydra` config. These parameters are parsed via `_set_bidning_params` and `_set_WTA_params`
        into `namedtuple`s (`NeuronParams`, `SynParams`, ConnectionParams`, which are combined into `ModelParams`).

        Args:
            n_neurons: Number of domain neurons to be modeled.
            config: `hydra` config containing all ODE parameters.
        """
        self._set_binding_params(config.binding_params)
        self._set_WTA_params(config.WTA_params)
        self._mode = DOMAIN_MODE.BINDING
        self.set_mode(self._mode)

        self._g = relu

        self._n_neurons = n_neurons
        self._input_weights = np.identity(n_neurons)
        self._pointer_weights = np.ones((n_neurons, n_neurons))
        self._lateral_weights = np.identity(n_neurons)

        h_0_syn = np.zeros(np.shape(self._pointer_weights))
        h_0_neur = np.zeros(n_neurons)
        self._h_0 = self.combine_state_space(h_0_syn, h_0_neur)

    def _set_params(self, config):
        nps = NeuronParams(**config.neuron_params)
        sps = SynParams(**config.syn_params)
        cps = ConnectionParams(**config.connection_params)
        return ModelParams(NeuronParams=nps, SynParams=sps, ConnectionParams=cps)

    def _set_binding_params(self, config):
        self._binding_params = self._set_params(config)

    def _set_WTA_params(self, config):
        self._WTA_params = self._set_params(config)

    def get_synaptic_state(self, h):
        """Getter method for the synaptic states from the combined state `h`.

        Args:
            h: Combined system state from which the synaptic state should be retrieved.
        """
        return np.reshape(h[: -self._n_neurons], (self._n_neurons, self._n_neurons))

    def get_synaptic_weights(self, h):
        """Getter method for the synaptic weights based on the models `_pointer_weights` and the synaptic state to be
        extracted from the combined state `h`.

        Args:
            h: Combined system state from which the synaptic state should be retrieved.
        """
        return self._pointer_weights * (1 + self.get_synaptic_state(h))

    def get_neuron_state(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[-self._n_neurons :]

    def get_neuron_rates(self, h):
        """Getter method for the neuron rates to be computed via the model's `_g` mapping and from the neuron states,
        which are extracted from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._g(self.get_neuron_state(h))

    def combine_state_space(self, h_syn, h_neur):
        """Combines the two subspaces of the state space via concatenation into an overall system state.

        Args:
            h_syn: Synaptic state.
            h_neur: Neuron state.

        Returns:
            np.array: Overall system state.
        """
        return np.concatenate((np.reshape(h_syn, (-1)), h_neur), axis=0)

    def set_mode(self, mode):
        """Method to force the model into a certain operation mode as defined by `DOMAIN_MODE`.

        Args:
            mode: Mode from the possible values of `DOMAIN_MODE` to which the model should be set.

        Raises:
            TypeError: The input argument was not of type `DOMAIN_MODE`.
        """
        if not isinstance(mode, DOMAIN_MODE):
            raise TypeError(
                "mode needs to be of type DOMAIN_MODE, got: {}".format(type(mode))
            )
        self._mode = mode

    def _get_mode_params(self):
        if self._mode == DOMAIN_MODE.BINDING:
            return self._binding_params
        if self._mode == DOMAIN_MODE.WTA:
            return self._WTA_params
        else:
            raise ValueError("Invalid mode, got: {}".format(self._mode))

    def _get_d_h_neur(self, h_neur, exc, mod, inh):
        ps = self._get_mode_params().NeuronParams
        d_h_neur = (
            -ps.alpha * h_neur
            + (ps.beta - h_neur) * exc * (1.0 + ps.delta * mod)
            - (ps.eta + ps.gamma * h_neur) * inh
        ) / ps.tau
        return d_h_neur

    def _get_d_h_syn(self, h_syn, h_neur, I_pointers):
        ps = self._get_mode_params().SynParams

        syn_corr = np.reshape(I_pointers, (1, -1)) * np.reshape(h_neur, (-1, 1))
        d_h_syn = (ps.syn_max - h_syn) * syn_corr / ps.tau_rise + (
            ps.syn_min - h_syn
        ) * (1 - syn_corr) / ps.tau_decay
        return d_h_syn

    def __call__(self, h, t, I_input, I_pointers):
        """Main method by which the model can be queried for a temporal derivative of its state values based on the
        current state `h`, a point in time `t`, driving inputs to the domain `I_input` and modulating input from
        pointers `I_pointers` and the internal model parameters and update equiations `_get_d_h_neur` and
        `_get_d_h_syn` implementing the respective ODEs.

        Parameters:
            h: Concatenation of states along axis 0. First part synaptic pointer states, last `n_neurons` entries
            neuron states.
            t: Point in time as required by the interface of `scipy.integrate.odeint`. So far unused as the ODE is time
            invariant.
            I_input: Driving input to the domain neurons.
            I_pointers: Modulating input to the domain neurons. Also used by the synapse dynamics to update their
            pointer values.

        Returns:
            np.array: Temporal derviative of the model's state values.
        """
        h_syn = self.get_synaptic_state(h)
        h_neur = self.get_neuron_state(h)

        # compute driving input
        exc = np.dot(self._input_weights, I_input)
        # compute modulating input
        mod = np.dot(self.get_synaptic_weights(h), I_pointers)
        # compute inhibition (competition)
        kappa = self._get_mode_params().ConnectionParams.kappa
        inh = sum(self.get_neuron_rates(h)) - kappa * self.get_neuron_rates(h)
        # compute pointer state update
        d_h_syn = self._get_d_h_syn(h_syn, h_neur, I_pointers)
        # compute neuron state update
        d_h_neur = self._get_d_h_neur(h_neur, exc, mod, inh)

        return self.combine_state_space(d_h_syn, d_h_neur)
