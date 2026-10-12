"""Dynamical model of synaptic neural pointers."""

from collections import namedtuple
from enum import Enum, auto

import numpy as np

from neural_pointers.activations import relu

NeuronParams = namedtuple(
    "NeuronParams", ["alpha", "beta", "delta", "eta", "gamma", "tau"]
)
SynParams = namedtuple("SynParams", ["syn_min", "syn_max", "tau_rise", "tau_decay"])
ConnectionParams = namedtuple("ConnectionParams", ["kappa", "k_inh"])
ModelParams = namedtuple(
    "ModelParams", ["NeuronParams", "SynParams", "ConnectionParams"]
)


class DOMAIN_MODE(Enum):
    """Modes of operation of the `SynapticPointerModel`."""

    BINDING = auto()
    WTA = auto()


class SynapticPointerModel:
    """A dynamical implementation of the synaptic pointer proposal of Zylberberg et al. (2013). A neuronal device for
    the control of multi-step computations. Papers in Physics 5, 050006.

    The model comprises `n` domain neurons with potentials `x_i` and rates `r_i = relu(x_i)`, and an `n x n` matrix of
    synaptic states `s_ij` of the connections from pointer `j` to domain neuron `i`. Pointer activities `p_j` are
    provided as external input. The dynamics read

        tau * dx_i/dt = -alpha * x_i + beta * I_i * (1 + delta * m_i) - eta * inh_i
        m_i           = sum_j (1 + s_ij) * p_j
        inh_i         = (1 - k_inh - kappa) * r_i + k_inh * sum_j r_j + x_i * I_inh_i
        ds_ij/dt      = (s_max - s_ij) * c_ij / tau_rise + (s_min - s_ij) * (1 - c_ij) / tau_decay
        c_ij          = p_j * r_i

    where `I` is the driving input to the domain and `I_inh` an external shunting inhibition used to reset the domain.
    Binding and read-out differ only in their parameters, which are switched via `set_mode` (`DOMAIN_MODE.BINDING`
    initially). With `kappa = k_inh = 1` the inhibition implements winner-take-all competition without
    self-inhibition. The parameter `gamma` is part of the configuration but does not enter the dynamics.

    The full state is the concatenation of the flattened synaptic states and the neuron potentials.
    """

    def __init__(self, n_neurons, config):
        """Model initialization.

        Args:
            n_neurons: Number of domain neurons (and pointers).
            config: Config with the parameter sets `binding_params` and `WTA_params`, each containing `neuron_params`,
                `syn_params` and `connection_params`.
        """
        self._binding_params = self._parse_params(config.binding_params)
        self._WTA_params = self._parse_params(config.WTA_params)
        self.set_mode(DOMAIN_MODE.BINDING)

        self._g = relu

        self._n_neurons = n_neurons
        self._input_weights = np.identity(n_neurons)
        self._pointer_weights = np.ones((n_neurons, n_neurons))

        h_0_syn = np.zeros(np.shape(self._pointer_weights))
        h_0_neur = np.zeros(n_neurons)
        self.h_0 = self.combine_state_space(h_0_syn, h_0_neur)

    @property
    def n_neurons(self):
        """Number of domain neurons (and pointers)."""
        return self._n_neurons

    @staticmethod
    def _parse_params(config):
        nps = NeuronParams(**config.neuron_params)
        sps = SynParams(**config.syn_params)
        cps = ConnectionParams(**config.connection_params)
        return ModelParams(NeuronParams=nps, SynParams=sps, ConnectionParams=cps)

    def get_synaptic_state(self, h):
        """Returns the `n x n` synaptic states (rows: domain neurons, columns: pointers) of the combined state `h`."""
        return np.reshape(h[: -self._n_neurons], (self._n_neurons, self._n_neurons))

    def get_synaptic_weights(self, h):
        """Returns the effective pointer-to-domain weights `w_ij = 1 + s_ij` of the combined state `h`."""
        return self._pointer_weights * (1.0 + self.get_synaptic_state(h))

    def get_neuron_state(self, h):
        """Returns the domain neuron potentials of the combined state `h`."""
        return h[-self._n_neurons :]

    def get_neuron_rates(self, h):
        """Returns the domain neuron rates of the combined state `h`."""
        return self._g(self.get_neuron_state(h))

    def combine_state_space(self, h_syn, h_neur):
        """Concatenates synaptic states and neuron potentials into the combined state.

        Args:
            h_syn: Synaptic states.
            h_neur: Neuron potentials.

        Returns:
            np.array: Combined state.
        """
        return np.concatenate((np.reshape(h_syn, (-1)), h_neur), axis=0)

    def set_mode(self, mode):
        """Switches the model to the parameter set of `mode`.

        Args:
            mode: A `DOMAIN_MODE`.

        Raises:
            TypeError: `mode` is not a `DOMAIN_MODE`.
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
        raise ValueError("Invalid mode, got: {}".format(self._mode))

    def _get_d_h_neur(self, h_neur, exc, mod, inh):
        ps = self._get_mode_params().NeuronParams
        d_h_neur = (
            -ps.alpha * h_neur + ps.beta * exc * (1.0 + ps.delta * mod) - ps.eta * inh
        ) / ps.tau
        return d_h_neur

    def _get_d_h_syn(self, h_syn, r_neur, I_pointers):
        ps = self._get_mode_params().SynParams
        syn_corr = np.reshape(I_pointers, (1, -1)) * np.reshape(r_neur, (-1, 1))
        d_h_syn = (ps.syn_max - h_syn) * syn_corr / ps.tau_rise + (
            ps.syn_min - h_syn
        ) * (1.0 - syn_corr) / ps.tau_decay
        return d_h_syn

    def __call__(self, h, t, I_input, I_pointers, I_input_inh):
        """Computes the temporal derivative of the combined state, following the interface of
        `scipy.integrate.odeint`.

        Args:
            h: Combined state (flattened synaptic states followed by the `n_neurons` neuron potentials).
            t: Point in time. Unused, the dynamics are time invariant.
            I_input: Driving input to the domain neurons.
            I_pointers: Pointer activities. Modulate the domain neurons and gate synaptic plasticity.
            I_input_inh: External shunting inhibition of the domain neurons.

        Returns:
            np.array: Temporal derivative of the combined state.
        """
        h_syn = self.get_synaptic_state(h)
        h_neur = self.get_neuron_state(h)
        r_neur = self.get_neuron_rates(h)

        # driving input
        exc = np.dot(self._input_weights, I_input)
        # modulation by the pointers
        mod = np.dot(self.get_synaptic_weights(h), I_pointers)
        # pooled inhibition (competition) and external shunting inhibition
        kappa = self._get_mode_params().ConnectionParams.kappa
        k_inh = self._get_mode_params().ConnectionParams.k_inh
        inh = (
            (1.0 - k_inh - kappa) * r_neur
            + k_inh * sum(r_neur)
            + h_neur * np.dot(self._input_weights, I_input_inh)
        )

        d_h_syn = self._get_d_h_syn(h_syn, r_neur, I_pointers)
        d_h_neur = self._get_d_h_neur(h_neur, exc, mod, inh)

        return self.combine_state_space(d_h_syn, d_h_neur)
