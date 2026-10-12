"""Dynamical model for monitoring the convergence of an input representation."""

from collections import namedtuple

import numpy as np

from neural_pointers.activations import identity, relu, step

DCNeuronParams = namedtuple(
    "DCNeuronParams",
    [
        "alpha",
        "beta",
        "gamma",
        "eta",
        "tau",
        "alpha_i",
        "beta_i",
        "tau_i",
        "alpha_inp",
        "beta_inp",
        "lambda_inp",
        "gamma_inp",
        "eta_inp",
        "tau_inp",
        "alpha_evdc",
        "beta_evdc",
        "tau_evdc",
        "thresh_evdc",
    ],
)
DCModelParams = namedtuple("DCModelParams", ["DCNeuronParams"])


class DifferenceComputationModel:
    """A monitoring network that signals when an input representation has settled.

    Five populations (concatenated in this order to form the full state):

    * difference neurons `d` (`exc` nodes): driven by the summed input activity, inhibited by a prediction,
    * prediction neurons `p` (`inh` nodes): integrate the difference neurons and catch up with the input,
    * input neurons `u` (`inp` nodes, `n_inputs_per_neuron * n` of them): driven by the external pattern, with random
      recurrent excitation and inhibition via `v`,
    * inhibitory input neurons `v` (`inp_inh` nodes): driven by random projections of `u`,
    * evidence neurons `e` (`evdc` nodes): accumulate while the difference is close to zero and signal convergence by
      crossing `thresh_evdc`.

    The recurrent weights are drawn from a dedicated generator seeded with `weight_seed`, independent of the
    experiment seed.

    Note:
        The evidence dynamics depend on two timers (`_t_start`, `_t_last`) that are updated inside `__call__`, i.e.
        at every evaluation by the ODE solver, which makes the solution depend on the solver's internal steps. The
        model only supports `n_neurons == 1`. See `docs/known_issues.md`.
    """

    def __init__(self, n_neurons, config):
        """Model initialization.

        Args:
            n_neurons: Number of difference neurons. Only 1 is supported.
            config: Config containing `neuron_params` (see `DCNeuronParams`), `n_inputs_per_neuron` and
                `weight_seed`.

        Raises:
            ValueError: `n_neurons` is not 1.
        """
        if n_neurons != 1:
            raise ValueError(
                "DifferenceComputationModel only supports n_neurons == 1, got: {}".format(
                    n_neurons
                )
            )
        self._params = DCModelParams(
            DCNeuronParams=DCNeuronParams(**config.neuron_params)
        )

        self._f = identity
        self._g = identity
        self._s = relu
        self._t = step

        self._n_neurons = n_neurons
        n_neurons_inp = config.n_inputs_per_neuron * n_neurons
        self._n_neurons_inp = n_neurons_inp

        weight_rng = np.random.default_rng(seed=config.weight_seed)
        # NOTE: dividing by the row sums without `keepdims` normalizes column j by the sum of row j
        # (see docs/known_issues.md).
        self._weights_exc = weight_rng.uniform(0, 1, (n_neurons_inp, n_neurons_inp))
        self._weights_exc = self._weights_exc / np.sum(self._weights_exc, axis=1)
        self._weights_inh = weight_rng.uniform(0, 1, (n_neurons_inp, n_neurons_inp))
        self._weights_inh = self._weights_inh / np.sum(self._weights_inh, axis=1)

        h_0_neur = np.zeros(n_neurons)
        h_0_neur_i = np.zeros(n_neurons)
        h_0_neur_inp = np.zeros(n_neurons_inp)
        h_0_neur_inp_inh = np.zeros(n_neurons_inp)
        h_0_neur_evdc = np.zeros(n_neurons)
        self.h_0 = self.combine_state_space(
            h_0_neur, h_0_neur_i, h_0_neur_inp, h_0_neur_inp_inh, h_0_neur_evdc
        )

        self._t_last = 0.0
        self._t_start = -1.0

    def _slice_exc(self):
        return slice(0, self._n_neurons)

    def _slice_inh(self):
        return slice(self._n_neurons, 2 * self._n_neurons)

    def _slice_inp(self):
        return slice(2 * self._n_neurons, 2 * self._n_neurons + self._n_neurons_inp)

    def _slice_inp_inh(self):
        start = 2 * self._n_neurons + self._n_neurons_inp
        return slice(start, start + self._n_neurons_inp)

    def _slice_evdc(self):
        return slice(2 * self._n_neurons + 2 * self._n_neurons_inp, None)

    def get_neuron_state_exc_nodes(self, h):
        """Returns the difference neuron potentials of the combined state `h`."""
        return h[self._slice_exc()]

    def get_neuron_rate_exc_nodes(self, h):
        """Returns the difference neuron rates of the combined state `h`."""
        return self._f(h[self._slice_exc()])

    def get_neuron_state_inh_nodes(self, h):
        """Returns the prediction neuron potentials of the combined state `h`."""
        return h[self._slice_inh()]

    def get_neuron_rate_inh_nodes(self, h):
        """Returns the prediction neuron rates of the combined state `h`."""
        return self._g(h[self._slice_inh()])

    def get_neuron_state_inp_nodes(self, h):
        """Returns the input neuron potentials of the combined state `h`."""
        return h[self._slice_inp()]

    def get_neuron_rate_inp_nodes(self, h):
        """Returns the input neuron rates of the combined state `h`."""
        return self._s(h[self._slice_inp()])

    def get_neuron_state_inp_inh_nodes(self, h):
        """Returns the inhibitory input neuron potentials of the combined state `h`."""
        return h[self._slice_inp_inh()]

    def get_neuron_rate_inp_inh_nodes(self, h):
        """Returns the inhibitory input neuron rates of the combined state `h`."""
        return self._s(h[self._slice_inp_inh()])

    def get_neuron_state_evdc_nodes(self, h):
        """Returns the evidence neuron potentials of the combined state `h`."""
        return h[self._slice_evdc()]

    def get_neuron_rate_evdc_nodes(self, h):
        """Returns the binary evidence neuron output of the combined state `h`."""
        return self._t(h[self._slice_evdc()], self._params.DCNeuronParams.thresh_evdc)

    def combine_state_space(
        self, h_neur, h_neur_i, h_neur_inp, h_neur_inp_inh, h_neur_evdc
    ):
        """Concatenates the potentials of all populations into the combined state.

        Args:
            h_neur: Difference neuron potentials.
            h_neur_i: Prediction neuron potentials.
            h_neur_inp: Input neuron potentials.
            h_neur_inp_inh: Inhibitory input neuron potentials.
            h_neur_evdc: Evidence neuron potentials.

        Returns:
            np.array: Combined state.
        """
        return np.concatenate(
            (h_neur, h_neur_i, h_neur_inp, h_neur_inp_inh, h_neur_evdc), axis=0
        )

    def _get_d_h_neur_exc_nodes(self, h_neur, h_neur_i, h_neur_inp):
        """Difference neuron dynamics:

        tau * dd/dt = -alpha * d + (beta - d) * sum_k |s(u_k)| - (eta + gamma * d) * g(p)
        """
        ps = self._params.DCNeuronParams

        exc = np.sum(np.abs(self._s(h_neur_inp)))
        inh = self._g(h_neur_i)

        d_h_neur = (
            -ps.alpha * h_neur  # passive decay
            + (ps.beta - h_neur) * exc  # shunted excitation
            - (ps.eta + ps.gamma * h_neur) * inh  # shunted inhibition
        ) / ps.tau
        return d_h_neur

    def _get_d_h_neur_inh_nodes(self, h_neur_i, h_neur):
        """Prediction neuron dynamics:

        tau_i * dp/dt = -alpha_i * p + (beta_i - p) * f(d)
        """
        ps = self._params.DCNeuronParams

        exc = self._f(h_neur)

        d_h_neur_i = (
            -ps.alpha_i * h_neur_i  # passive decay
            + (ps.beta_i - h_neur_i) * exc  # shunted excitation
        ) / ps.tau_i
        return d_h_neur_i

    def _get_d_h_neur_inp_nodes(self, h_neur_inp, I_inp, h_neur_inp_inh):
        """Input neuron dynamics:

        tau_inp * du/dt = -alpha_inp * u + (beta_inp - u) * (I + lambda_inp * W_exc s(u))
                          - (eta_inp + gamma_inp * u) * v
        """
        ps = self._params.DCNeuronParams

        exc = I_inp + ps.lambda_inp * np.dot(self._weights_exc, self._s(h_neur_inp))
        inh = h_neur_inp_inh

        d_h_neur_inp = (
            -ps.alpha_inp * h_neur_inp  # passive decay
            + (ps.beta_inp - h_neur_inp) * exc  # shunted excitation
            - (ps.eta_inp + ps.gamma_inp * h_neur_inp) * inh  # shunted inhibition
        ) / ps.tau_inp
        return d_h_neur_inp

    def _get_d_h_neur_inp_inh_nodes(self, h_neur_inp_inh, h_neur_inp):
        """Inhibitory input neuron dynamics:

        tau_inp * dv/dt = -alpha_inp * v + (beta_inp - v) * W_inh s(u)
        """
        ps = self._params.DCNeuronParams

        exc = np.dot(self._weights_inh, self._s(h_neur_inp))

        d_h_neur_inp_inh = (
            -ps.alpha_inp * h_neur_inp_inh  # passive decay
            + (ps.beta_inp - h_neur_inp_inh) * exc  # shunted excitation
        ) / ps.tau_inp
        return d_h_neur_inp_inh

    def _get_d_h_neur_evdc_nodes(self, h_neur_evdc, h_neur, t):
        """Evidence neuron dynamics, accumulating while `|d| < 0.001` once 20 time units have passed since the last
        convergence signal:

        tau_evdc * de/dt = -alpha_evdc * e + (beta_evdc - e) * [|f(d)| < 0.001] * [t > 20 + t_last]
        """
        ps = self._params.DCNeuronParams

        exc = (np.abs(self._f(h_neur)) < 0.001) * (t > (20.0 + self._t_last))

        d_h_neur_evdc = (
            -ps.alpha_evdc * h_neur_evdc  # passive decay
            + (ps.beta_evdc - h_neur_evdc) * exc  # shunted excitation
        ) / ps.tau_evdc
        return d_h_neur_evdc

    def __call__(self, h, t, I_input):
        """Computes the temporal derivative of the combined state, following the interface of
        `scipy.integrate.odeint`.

        Updates the evidence timers as a side effect: `_t_start` marks the onset of a convergence signal and `_t_last`
        is set once the signal has persisted for 10 time units.

        Args:
            h: Combined state (`d`, `p`, `u`, `v` and `e` concatenated).
            t: Point in time.
            I_input: External input pattern to the input neurons.

        Returns:
            np.array: Temporal derivative of the combined state.
        """
        h_neur = self.get_neuron_state_exc_nodes(h)
        h_neur_i = self.get_neuron_state_inh_nodes(h)
        h_neur_inp = self.get_neuron_state_inp_nodes(h)
        h_neur_inp_inh = self.get_neuron_state_inp_inh_nodes(h)
        h_neur_evdc = self.get_neuron_state_evdc_nodes(h)

        r_neur_evdc = self.get_neuron_rate_evdc_nodes(h)
        if r_neur_evdc and (self._t_start < self._t_last + 10.0):
            self._t_start = t
        if r_neur_evdc and (t > (self._t_start + 10.0)):
            self._t_last = t

        d_h_neur = self._get_d_h_neur_exc_nodes(h_neur, h_neur_i, h_neur_inp)
        d_h_neur_i = self._get_d_h_neur_inh_nodes(h_neur_i, h_neur)
        d_h_neur_inp = self._get_d_h_neur_inp_nodes(h_neur_inp, I_input, h_neur_inp_inh)
        d_h_neur_inp_inh = self._get_d_h_neur_inp_inh_nodes(h_neur_inp_inh, h_neur_inp)
        d_h_neur_evdc = self._get_d_h_neur_evdc_nodes(h_neur_evdc, h_neur, t)

        return self.combine_state_space(
            d_h_neur, d_h_neur_i, d_h_neur_inp, d_h_neur_inp_inh, d_h_neur_evdc
        )


def convergence_time(model, h, t):
    """Determines when the evidence neurons first signal convergence within a stage.

    Args:
        model: The simulated `DifferenceComputationModel`.
        h: Array of states for the time points `t`.
        t: Time points of the stage.

    Returns:
        tuple: Stage onset, time of the first convergence signal and their difference. The latter two are `nan` if no
        convergence was signaled.
    """
    r_evdc = np.array([model.get_neuron_rate_evdc_nodes(x) for x in h])
    idxs = np.where(r_evdc)[0]
    if idxs.size == 0:
        return t[0], np.nan, np.nan
    return t[0], t[idxs[0]], t[idxs[0]] - t[0]
