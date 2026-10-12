"""Dynamical model of competitive queueing."""

from collections import namedtuple

import numpy as np

from neural_pointers.activations import relu, square, threshold_linear

CQNeuronParams = namedtuple(
    "CQNeuronParams",
    [
        "A_x",
        "B_x",
        "C_x",
        "tau_x",
        "G",
        "epsilon",
        "A_y",
        "B_y",
        "C_y",
        "tau_y",
        "A_w",
        "B_w",
        "tau_w",
        "theta_t",
    ],
)
CQModelParams = namedtuple("CQModelParams", ["CQNeuronParams"])


class CompetitiveQueueingModel:
    """A competitive queueing network following Boardman & Bullock (1991). A neural network model of serial order
    recall from short-term memory. IJCNN-91-Seattle, vol. 2, 879-884.

    Three layers of `n` shunting neurons each:

    * input (planning) nodes `x` hold a graded activity pattern encoding the serial order of `n` items,
    * output (choice) nodes `y` compete in a winner-take-all fashion when the arousal signal `J` is on,
    * inhibitory (feedback) nodes `w` integrate above-threshold output activity and suppress the input node of the item
      that has been executed, so that the next item can win the competition.

    The full state is the concatenation of `x`, `y` and `w`.
    """

    def __init__(self, n_neurons, config):
        """Model initialization.

        Args:
            n_neurons: Number of items (neurons per layer).
            config: Config containing `neuron_params`, see `CQNeuronParams`.
        """
        self._params = CQModelParams(
            CQNeuronParams=CQNeuronParams(**config.neuron_params)
        )

        self._f = relu
        self._g = relu
        self._h = relu
        self._q = relu
        self._r = square
        self._s = square
        self._t = lambda x: threshold_linear(
            x, thresh=self._params.CQNeuronParams.theta_t
        )

        self._n_neurons = n_neurons

        h_0_neur_x = np.zeros(n_neurons)
        h_0_neur_y = np.zeros(n_neurons)
        h_0_neur_w = np.zeros(n_neurons)
        self.h_0 = self.combine_state_space(h_0_neur_x, h_0_neur_y, h_0_neur_w)

    @property
    def n_neurons(self):
        """Number of items (neurons per layer)."""
        return self._n_neurons

    def get_neuron_state_input_nodes(self, h):
        """Returns the input node potentials `x` of the combined state `h`."""
        return h[: self._n_neurons]

    def get_neuron_rate_input_nodes(self, h):
        """Returns the input node rates `f(x)` of the combined state `h`."""
        return self._f(h[: self._n_neurons])

    def get_neuron_state_output_nodes(self, h):
        """Returns the output node potentials `y` of the combined state `h`."""
        return h[self._n_neurons : 2 * self._n_neurons]

    def get_neuron_rate_output_nodes(self, h):
        """Returns the output node rates `s(y)` of the combined state `h`."""
        return self._s(h[self._n_neurons : 2 * self._n_neurons])

    def get_neuron_state_inhib_nodes(self, h):
        """Returns the inhibitory node potentials `w` of the combined state `h`."""
        return h[2 * self._n_neurons :]

    def get_neuron_rate_inhib_nodes(self, h):
        """Returns the inhibitory node rates `h(w)` of the combined state `h`."""
        return self._h(h[2 * self._n_neurons :])

    def combine_state_space(self, h_neur_x, h_neur_y, h_neur_w):
        """Concatenates input, output and inhibitory node potentials into the combined state.

        Args:
            h_neur_x: Input node potentials.
            h_neur_y: Output node potentials.
            h_neur_w: Inhibitory node potentials.

        Returns:
            np.array: Combined state.
        """
        return np.concatenate((h_neur_x, h_neur_y, h_neur_w), axis=0)

    def _get_d_h_neur_input_nodes(self, h_neur_x, h_neur_w, h_neur_I, h_neur_J):
        """Input node dynamics:

        tau_x * dx_i/dt = -A_x * x_i
                          + (B_x - x_i) * (f(x_i - I_i) / S + I_i)
                          - (C_x + x_i) * (sum_{j != i} g(x_j - I_j) + h(w_i))

        with `S = G / (epsilon + sum_j f(x_j))` while the arousal `J > 0` and `S = 1` otherwise.
        """
        ps = self._params.CQNeuronParams

        h_S = ps.G / (ps.epsilon + np.sum(self._f(h_neur_x))) * (h_neur_J > 0) + 1.0 * (
            h_neur_J <= 0
        )

        exc = self._f(h_neur_x - h_neur_I) / h_S + h_neur_I
        inh = (
            np.sum(self._g(h_neur_x - h_neur_I))
            - self._g(h_neur_x - h_neur_I)
            + self._h(h_neur_w)
        )

        d_h_neur_x = (
            -ps.A_x * h_neur_x  # passive decay
            + (ps.B_x - h_neur_x) * exc  # shunted excitation
            - (ps.C_x + h_neur_x) * inh  # shunted inhibition
        ) / ps.tau_x
        return d_h_neur_x

    def _get_d_h_neur_output_nodes(self, h_neur_y, h_neur_x, h_neur_J):
        """Output node dynamics:

        tau_y * dy_i/dt = -A_y * y_i + (B_y - y_i) * (s(y_i) + J * q(x_i)) - (C_y + y_i) * sum_{j != i} r(y_j)
        """
        ps = self._params.CQNeuronParams

        exc = self._s(h_neur_y) + h_neur_J * self._q(h_neur_x)
        inh = np.sum(self._r(h_neur_y)) - self._r(h_neur_y)

        d_h_neur_y = (
            -ps.A_y * h_neur_y  # passive decay
            + (ps.B_y - h_neur_y) * exc  # shunted excitation
            - (ps.C_y + h_neur_y) * inh  # shunted inhibition
        ) / ps.tau_y
        return d_h_neur_y

    def _get_d_h_neur_inhib_nodes(self, h_neur_y, h_neur_w):
        """Inhibitory node dynamics, with `t` being threshold-linear at `theta_t`:

        tau_w * dw_i/dt = -A_w * w_i + (B_w - w_i) * t(y_i)
        """
        ps = self._params.CQNeuronParams

        exc = self._t(h_neur_y)

        d_h_neur_w = (
            -ps.A_w * h_neur_w  # passive decay
            + (ps.B_w - h_neur_w) * exc  # shunted excitation
        ) / ps.tau_w
        return d_h_neur_w

    def __call__(self, h, t, I_input, I_J):
        """Computes the temporal derivative of the combined state, following the interface of
        `scipy.integrate.odeint`.

        Args:
            h: Combined state (`x`, `y` and `w` concatenated).
            t: Point in time. Unused, the dynamics are time invariant.
            I_input: Graded input pattern to the input nodes.
            I_J: Scalar arousal signal gating the competition of the output nodes.

        Returns:
            np.array: Temporal derivative of the combined state.
        """
        h_neur_x = self.get_neuron_state_input_nodes(h)
        h_neur_y = self.get_neuron_state_output_nodes(h)
        h_neur_w = self.get_neuron_state_inhib_nodes(h)

        d_h_neur_x = self._get_d_h_neur_input_nodes(h_neur_x, h_neur_w, I_input, I_J)
        d_h_neur_y = self._get_d_h_neur_output_nodes(h_neur_y, h_neur_x, I_J)
        d_h_neur_w = self._get_d_h_neur_inhib_nodes(h_neur_y, h_neur_w)

        return self.combine_state_space(d_h_neur_x, d_h_neur_y, d_h_neur_w)
