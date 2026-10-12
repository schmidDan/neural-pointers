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

# Structural variants of the model (see `SynapticPointerModel`); the first value of each entry is the default.
VARIANTS = {
    "excitation": ("additive", "shunting"),
    "inhibition": ("subtractive", "shunting"),
    "pointer_weights": ("offset", "synaptic"),
    "pointer_dynamics": (False, True),
}


class DOMAIN_MODE(Enum):
    """Modes of operation of the `SynapticPointerModel`."""

    BINDING = auto()
    WTA = auto()


class SynapticPointerModel:
    """A dynamical implementation of the synaptic pointer proposal of Zylberberg et al. (2013). A neuronal device for
    the control of multi-step computations. Papers in Physics 5, 050006.

    The model comprises `n` domain neurons with potentials `x_i` and rates `r_i = relu(x_i)`, `n` pointers with
    activities `p_j`, and an `n x n` matrix of synaptic states `s_ij` of the connections from pointer `j` to domain
    neuron `i`. The dynamics read

        tau * dx_i/dt = -alpha * x_i + E_i - H_i
        pool_i        = (1 - k_inh - kappa) * r_i + k_inh * sum_j r_j
        m_i           = sum_j w_ij * p_j
        ds_ij/dt      = (s_max - s_ij) * c_ij / tau_rise + (s_min - s_ij) * (1 - c_ij) / tau_decay
        c_ij          = min(p_j * r_i, 1)

    where `I` is the driving input to the domain and `I_inh` an external inhibition that shunts the potentials to zero
    to reset the domain.
    Binding and read-out differ only in their parameters, which are switched via `set_mode` (`DOMAIN_MODE.BINDING`
    initially). With `kappa = k_inh = 1` the inhibition implements winner-take-all competition without
    self-inhibition.

    The structure of the model is selected by the `variant` config (defaults first):

    * `excitation`: `additive`, `E_i = beta * I_i * (1 + delta * m_i)`, or `shunting`,
      `E_i = (beta - x_i) * I_i * (1 + delta * m_i)`.
    * `inhibition`: `subtractive`, `H_i = eta * (pool_i + x_i * I_inh_i)`, or `shunting`,
      `H_i = (eta + gamma * x_i) * pool_i + eta * x_i * I_inh_i`.
    * `pointer_weights`: `offset`, `w_ij = 1 + s_ij` (every pointer modulates every domain neuron), or `synaptic`,
      `w_ij = s_ij` (only bound pairs).
    * `pointer_dynamics`: `false`, the pointer activities equal the external pointer input `I_c`, or `true`, pointer
      neurons integrate the input as a control signal, `tau_p * dy_j/dt = -y_j + I_c_j`, `p_j = relu(y_j)`.

    The full state is the concatenation of the flattened synaptic states, the domain potentials and, with pointer
    dynamics, the pointer potentials `y`.
    """

    def __init__(self, n_neurons, config):
        """Model initialization.

        Args:
            n_neurons: Number of domain neurons (and pointers).
            config: Config with the parameter sets `binding_params` and `WTA_params`, each containing `neuron_params`,
                `syn_params` and `connection_params`, and optionally `variant` (see above) and `pointer_params`
                (`tau`, the time constant of the pointer neurons).

        Raises:
            ValueError: Unknown variant.
        """
        self._binding_params = self._parse_params(config.binding_params)
        self._WTA_params = self._parse_params(config.WTA_params)
        self.set_mode(DOMAIN_MODE.BINDING)

        variant = dict(config.get("variant") or {})
        self._variant = {}
        for key, options in VARIANTS.items():
            value = variant.pop(key, options[0])
            if value not in options:
                raise ValueError(
                    "Variant {} must be one of {}, got: {}".format(key, options, value)
                )
            self._variant[key] = value
        if variant:
            raise ValueError("Unknown variants: {}".format(sorted(variant)))
        self._tau_pointer = (config.get("pointer_params") or {}).get("tau", 1.0)

        self._g = relu

        self._n_neurons = n_neurons
        self._input_weights = np.identity(n_neurons)
        self._pointer_weights = np.ones((n_neurons, n_neurons))

        h_0_syn = np.zeros(np.shape(self._pointer_weights))
        h_0_neur = np.zeros(n_neurons)
        h_0_pointer = np.zeros(n_neurons if self.has_pointer_dynamics else 0)
        self.h_0 = self.combine_state_space(h_0_syn, h_0_neur, h_0_pointer)

    @property
    def variant(self):
        """The selected structural variant, see `VARIANTS`."""
        return dict(self._variant)

    @property
    def has_pointer_dynamics(self):
        """Whether the pointers are modeled as neurons with their own dynamics."""
        return self._variant["pointer_dynamics"]

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
        n = self._n_neurons
        return np.reshape(h[: n * n], (n, n))

    def get_synaptic_weights(self, h):
        """Returns the effective pointer-to-domain weights `w_ij` of the combined state `h`."""
        if self._variant["pointer_weights"] == "offset":
            return self._pointer_weights * (1.0 + self.get_synaptic_state(h))
        return self._pointer_weights * self.get_synaptic_state(h)

    def get_neuron_state(self, h):
        """Returns the domain neuron potentials of the combined state `h`."""
        n = self._n_neurons
        return h[n * n : n * n + n]

    def get_pointer_state(self, h):
        """Returns the pointer potentials `y` of the combined state `h` (empty without pointer dynamics)."""
        n = self._n_neurons
        return h[n * n + n :]

    def get_pointer_rates(self, h, I_pointers):
        """Returns the pointer activities for the combined state `h` and the external pointer input `I_pointers`."""
        if self.has_pointer_dynamics:
            return relu(self.get_pointer_state(h))
        return I_pointers

    def get_neuron_rates(self, h):
        """Returns the domain neuron rates of the combined state `h`."""
        return self._g(self.get_neuron_state(h))

    def combine_state_space(self, h_syn, h_neur, h_pointer=()):
        """Concatenates synaptic states, domain potentials and pointer potentials into the combined state.

        Args:
            h_syn: Synaptic states.
            h_neur: Domain neuron potentials.
            h_pointer: Pointer potentials (empty without pointer dynamics).

        Returns:
            np.array: Combined state.
        """
        return np.concatenate((np.reshape(h_syn, (-1)), h_neur, h_pointer), axis=0)

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

    def _get_d_h_neur(self, h_neur, exc, mod, pool, reset):
        ps = self._get_mode_params().NeuronParams
        if self._variant["excitation"] == "additive":
            excitation = ps.beta * exc * (1.0 + ps.delta * mod)
        else:
            excitation = (ps.beta - h_neur) * exc * (1.0 + ps.delta * mod)
        if self._variant["inhibition"] == "subtractive":
            inhibition = ps.eta * (pool + reset)
        else:
            # the reset is not scaled by the shunting factor, which would make it quadratic in the potential
            inhibition = (ps.eta + ps.gamma * h_neur) * pool + ps.eta * reset
        d_h_neur = (-ps.alpha * h_neur + excitation - inhibition) / ps.tau
        return d_h_neur

    def _get_d_h_syn(self, h_syn, r_neur, I_pointers):
        ps = self._get_mode_params().SynParams
        # coincidence of pointer and domain activity, bounded so that the states stay within [s_min, s_max]
        syn_corr = np.minimum(
            np.reshape(I_pointers, (1, -1)) * np.reshape(r_neur, (-1, 1)), 1.0
        )
        d_h_syn = (ps.syn_max - h_syn) * syn_corr / ps.tau_rise + (
            ps.syn_min - h_syn
        ) * (1.0 - syn_corr) / ps.tau_decay
        return d_h_syn

    def __call__(self, h, t, I_input, I_pointers, I_input_inh):
        """Computes the temporal derivative of the combined state, following the interface of
        `scipy.integrate.odeint`.

        Args:
            h: Combined state (flattened synaptic states, domain potentials and, with pointer dynamics, pointer
                potentials).
            t: Point in time. Unused, the dynamics are time invariant.
            I_input: Driving input to the domain neurons.
            I_pointers: External pointer input. The pointer activities modulate the domain neurons and gate synaptic
                plasticity.
            I_input_inh: External shunting inhibition of the domain neurons.

        Returns:
            np.array: Temporal derivative of the combined state.
        """
        h_syn = self.get_synaptic_state(h)
        h_neur = self.get_neuron_state(h)
        r_neur = self.get_neuron_rates(h)
        r_pointer = self.get_pointer_rates(h, I_pointers)

        # driving input
        exc = np.dot(self._input_weights, I_input)
        # modulation by the pointers
        mod = np.dot(self.get_synaptic_weights(h), r_pointer)
        # pooled inhibition (competition) and external reset
        kappa = self._get_mode_params().ConnectionParams.kappa
        k_inh = self._get_mode_params().ConnectionParams.k_inh
        pool = (1.0 - k_inh - kappa) * r_neur + k_inh * sum(r_neur)
        reset = h_neur * np.dot(self._input_weights, I_input_inh)

        d_h_syn = self._get_d_h_syn(h_syn, r_neur, r_pointer)
        d_h_neur = self._get_d_h_neur(h_neur, exc, mod, pool, reset)
        if self.has_pointer_dynamics:
            d_h_pointer = (-self.get_pointer_state(h) + I_pointers) / self._tau_pointer
        else:
            d_h_pointer = ()

        return self.combine_state_space(d_h_syn, d_h_neur, d_h_pointer)
