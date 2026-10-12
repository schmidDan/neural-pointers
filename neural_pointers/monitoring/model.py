"""Dynamical model for monitoring the convergence of an input representation."""

from collections import namedtuple

import numpy as np

from neural_pointers.activations import identity, logistic, relu

MonitoringNeuronParams = namedtuple(
    "MonitoringNeuronParams",
    [
        "alpha",
        "beta",
        "gamma",
        "eta",
        "tau",
        "alpha_pred",
        "beta_pred",
        "tau_pred",
        "alpha_input",
        "beta_input",
        "lambda_input",
        "gamma_input",
        "eta_input",
        "tau_input",
        "theta_d",
        "alpha_evidence",
        "beta_evidence",
        "tau_evidence",
        "thresh_evidence",
        "width_evidence",
        "w_self_evidence",
        "gamma_adapt",
        "tau_adapt",
        "kappa_adapt",
    ],
)
MonitoringModelParams = namedtuple("MonitoringModelParams", ["MonitoringNeuronParams"])


class MonitoringModel:
    """A monitoring network that signals when an input representation has settled.

    Six populations of `n` neurons each, except for the input layers with `n_inputs_per_neuron * n` neurons
    (concatenated in this order to form the full state):

    * difference neurons `d` (`exc` nodes): driven by the summed activity of their group of input neurons, inhibited
      by a prediction,
    * prediction neurons `p` (`inh` nodes): integrate the difference and catch up with the input, so that `d`
      returns to zero once the input has settled,
    * input neurons `u` (`inp` nodes): driven by the external pattern, with random recurrent excitation and
      inhibition via `v`. They act as the database holding the processed pattern: the recurrent interactions dominate
      the approach to the pattern's attractor, so the settling time is specific to the pattern,
    * inhibitory input neurons `v` (`inp_inh` nodes): driven by random projections of `u`,
    * evidence neurons `e` (`evidence` nodes): accumulate while the difference is close to zero; once their output switches
      on, self-excitation drives it to saturation,
    * adaptation variables `a` (`adapt` nodes): terminate the convergence signal and re-enable the evidence
      accumulation when the difference deviates from zero again, i.e. when the input changes.

    The output of the evidence neurons, `r = logistic((e - thresh_evidence) / width_evidence)`, signals convergence with an
    all-or-none pulse once per change of the input; the pulse also ends when the input changes. The recurrent weights of the input layer are drawn from a dedicated generator seeded with
    `weight_seed`, independent of the experiment seed. The dynamics are time invariant.
    """

    def __init__(self, n_neurons, config):
        """Model initialization.

        Args:
            n_neurons: Number of difference neurons.
            config: Config containing `neuron_params` (see `MonitoringNeuronParams`), `n_inputs_per_neuron` and
                `weight_seed`.
        """
        self._params = MonitoringModelParams(
            MonitoringNeuronParams=MonitoringNeuronParams(**config.neuron_params)
        )

        self._f = identity
        self._g = identity
        self._s = relu

        self._n_neurons = n_neurons
        self._n_inputs_per_neuron = config.n_inputs_per_neuron
        n_neurons_input = config.n_inputs_per_neuron * n_neurons
        self._n_neurons_input = n_neurons_input

        weight_rng = np.random.default_rng(seed=config.weight_seed)
        # the incoming weights of each neuron sum to one
        self._weights_exc = weight_rng.uniform(0, 1, (n_neurons_input, n_neurons_input))
        self._weights_exc /= np.sum(self._weights_exc, axis=1, keepdims=True)
        self._weights_inh = weight_rng.uniform(0, 1, (n_neurons_input, n_neurons_input))
        self._weights_inh /= np.sum(self._weights_inh, axis=1, keepdims=True)

        h_0_neur = np.zeros(n_neurons)
        h_0_neur_i = np.zeros(n_neurons)
        h_0_neur_input = np.zeros(n_neurons_input)
        h_0_neur_input_inh = np.zeros(n_neurons_input)
        h_0_neur_evidence = np.zeros(n_neurons)
        h_0_adapt = np.zeros(n_neurons)
        self.h_0 = self.combine_state_space(
            h_0_neur,
            h_0_neur_i,
            h_0_neur_input,
            h_0_neur_input_inh,
            h_0_neur_evidence,
            h_0_adapt,
        )

    @property
    def n_neurons(self):
        """Number of difference neurons."""
        return self._n_neurons

    def _slice_difference(self):
        return slice(0, self._n_neurons)

    def _slice_inh(self):
        return slice(self._n_neurons, 2 * self._n_neurons)

    def _slice_input(self):
        return slice(2 * self._n_neurons, 2 * self._n_neurons + self._n_neurons_input)

    def _slice_input_inh(self):
        start = 2 * self._n_neurons + self._n_neurons_input
        return slice(start, start + self._n_neurons_input)

    def _slice_evidence(self):
        start = 2 * self._n_neurons + 2 * self._n_neurons_input
        return slice(start, start + self._n_neurons)

    def _slice_adapt(self):
        start = 3 * self._n_neurons + 2 * self._n_neurons_input
        return slice(start, start + self._n_neurons)

    def get_neuron_state_difference_nodes(self, h):
        """Returns the difference neuron potentials of the combined state `h`."""
        return h[self._slice_difference()]

    def get_neuron_rate_difference_nodes(self, h):
        """Returns the difference neuron rates of the combined state `h`."""
        return self._f(h[self._slice_difference()])

    def get_neuron_state_prediction_nodes(self, h):
        """Returns the prediction neuron potentials of the combined state `h`."""
        return h[self._slice_inh()]

    def get_neuron_rate_prediction_nodes(self, h):
        """Returns the prediction neuron rates of the combined state `h`."""
        return self._g(h[self._slice_inh()])

    def get_neuron_state_input_nodes(self, h):
        """Returns the input neuron potentials of the combined state `h`."""
        return h[self._slice_input()]

    def get_neuron_rate_input_nodes(self, h):
        """Returns the input neuron rates of the combined state `h`."""
        return self._s(h[self._slice_input()])

    def get_neuron_state_input_inh_nodes(self, h):
        """Returns the inhibitory input neuron potentials of the combined state `h`."""
        return h[self._slice_input_inh()]

    def get_neuron_rate_input_inh_nodes(self, h):
        """Returns the inhibitory input neuron rates of the combined state `h`."""
        return self._s(h[self._slice_input_inh()])

    def get_neuron_state_evidence_nodes(self, h):
        """Returns the evidence neuron potentials of the combined state `h`."""
        return h[self._slice_evidence()]

    def get_neuron_rate_evidence_nodes(self, h):
        """Returns the evidence neuron output (convergence signal, between 0 and 1) of the combined state `h`."""
        ps = self._params.MonitoringNeuronParams
        return logistic(
            h[self._slice_evidence()], ps.thresh_evidence, ps.width_evidence
        )

    def get_adaptation_state(self, h):
        """Returns the adaptation variables of the combined state `h`."""
        return h[self._slice_adapt()]

    def get_closeness(self, h):
        """Returns the closeness of the difference to zero, `1 / (1 + (d / theta_d)^2)`, of the combined state `h`."""
        return self._closeness(self.get_neuron_rate_difference_nodes(h))

    def _closeness(self, r_neur):
        return 1.0 / (
            1.0 + np.square(r_neur / self._params.MonitoringNeuronParams.theta_d)
        )

    def combine_state_space(
        self,
        h_neur,
        h_prediction,
        h_neur_input,
        h_neur_input_inh,
        h_neur_evidence,
        h_adapt,
    ):
        """Concatenates the states of all populations into the combined state.

        Args:
            h_neur: Difference neuron potentials.
            h_prediction: Prediction neuron potentials.
            h_neur_input: Input neuron potentials.
            h_neur_input_inh: Inhibitory input neuron potentials.
            h_neur_evidence: Evidence neuron potentials.
            h_adapt: Adaptation variables.

        Returns:
            np.array: Combined state.
        """
        return np.concatenate(
            (
                h_neur,
                h_prediction,
                h_neur_input,
                h_neur_input_inh,
                h_neur_evidence,
                h_adapt,
            ),
            axis=0,
        )

    def _get_d_h_neur_difference_nodes(self, h_neur, h_prediction, h_neur_input):
        """Difference neuron dynamics, with `U_i` the summed rate of the input group of neuron `i`:

        tau * dd_i/dt = -alpha * d_i + (beta - d_i) * U_i - (eta + gamma * d_i) * g(p_i)
        """
        ps = self._params.MonitoringNeuronParams

        exc = np.sum(
            np.reshape(
                self._s(h_neur_input), (self._n_neurons, self._n_inputs_per_neuron)
            ),
            axis=1,
        )
        inh = self._g(h_prediction)

        d_h_neur = (
            -ps.alpha * h_neur  # passive decay
            + (ps.beta - h_neur) * exc  # shunted excitation
            - (ps.eta + ps.gamma * h_neur) * inh  # shunted inhibition
        ) / ps.tau
        return d_h_neur

    def _get_d_h_neur_prediction_nodes(self, h_prediction, h_neur):
        """Prediction neuron dynamics:

        tau_pred * dp/dt = -alpha_pred * p + (beta_pred - p) * f(d)
        """
        ps = self._params.MonitoringNeuronParams

        exc = self._f(h_neur)

        d_h_neur_i = (
            -ps.alpha_pred * h_prediction  # passive decay
            + (ps.beta_pred - h_prediction) * exc  # shunted excitation
        ) / ps.tau_pred
        return d_h_neur_i

    def _get_d_h_neur_input_nodes(self, h_neur_input, I_input, h_neur_input_inh, I_inh):
        """Input neuron dynamics, with an external shunting inhibition `I_inh` that resets the database:

        tau_input * du/dt = -alpha_input * u + (beta_input - u) * (I + lambda_input * W_exc s(u))
                          - (eta_input + gamma_input * u) * v - u * I_inh
        """
        ps = self._params.MonitoringNeuronParams

        exc = I_input + ps.lambda_input * np.dot(
            self._weights_exc, self._s(h_neur_input)
        )
        inh = h_neur_input_inh

        d_h_neur_input = (
            -ps.alpha_input * h_neur_input  # passive decay
            + (ps.beta_input - h_neur_input) * exc  # shunted excitation
            - (ps.eta_input + ps.gamma_input * h_neur_input) * inh  # shunted inhibition
            - h_neur_input * I_inh  # external reset
        ) / ps.tau_input
        return d_h_neur_input

    def _get_d_h_neur_input_inh_nodes(self, h_neur_input_inh, h_neur_input, I_inh):
        """Inhibitory input neuron dynamics, reset together with the input neurons:

        tau_input * dv/dt = -alpha_input * v + (beta_input - v) * W_inh s(u) - v * I_inh
        """
        ps = self._params.MonitoringNeuronParams

        exc = np.dot(self._weights_inh, self._s(h_neur_input))

        d_h_neur_input_inh = (
            -ps.alpha_input * h_neur_input_inh  # passive decay
            + (ps.beta_input - h_neur_input_inh) * exc  # shunted excitation
            - h_neur_input_inh * I_inh  # external reset
        ) / ps.tau_input
        return d_h_neur_input_inh

    def _get_d_h_neur_evidence_nodes(
        self, h_neur_evidence, closeness, h_adapt, r_evidence
    ):
        """Evidence neuron dynamics, with closeness `c = 1 / (1 + (d / theta_d)^2)` and output `r`:

        tau_evidence * de/dt = -alpha_evidence * e + (beta_evidence - e) * c * (1 + w_self_evidence * r) * (1 - a)
                           - gamma_adapt * a * e
        """
        ps = self._params.MonitoringNeuronParams

        exc = closeness * (1.0 + ps.w_self_evidence * r_evidence) * (1.0 - h_adapt)

        d_h_neur_evidence = (
            -ps.alpha_evidence * h_neur_evidence  # passive decay
            + (ps.beta_evidence - h_neur_evidence) * exc  # shunted excitation
            - ps.gamma_adapt
            * h_adapt
            * h_neur_evidence  # shunting inhibition by the adaptation
        ) / ps.tau_evidence
        return d_h_neur_evidence

    def _get_d_h_adapt(self, h_adapt, r_evidence, closeness):
        """Adaptation dynamics: charged by the convergence signal `r`, discharged while the difference deviates from
        zero:

        tau_adapt * da/dt = (1 - a) * r - kappa_adapt * a * (1 - c)
        """
        ps = self._params.MonitoringNeuronParams
        return (
            (1.0 - h_adapt) * r_evidence - ps.kappa_adapt * h_adapt * (1.0 - closeness)
        ) / ps.tau_adapt

    def __call__(self, h, t, I_input, I_inh=0.0):
        """Computes the temporal derivative of the combined state, following the interface of
        `scipy.integrate.odeint`.

        Args:
            h: Combined state (`d`, `p`, `u`, `v`, `e` and `a` concatenated).
            t: Point in time. Unused, the dynamics are time invariant.
            I_input: External input pattern to the input neurons.
            I_inh: External shunting inhibition of the input layer, resets the database (e.g. between operations in
                the closed loop). The comparator and the evidence system are not inhibited; they follow the reset
                input on their own.

        Returns:
            np.array: Temporal derivative of the combined state.
        """
        h_neur = self.get_neuron_state_difference_nodes(h)
        h_prediction = self.get_neuron_state_prediction_nodes(h)
        h_neur_input = self.get_neuron_state_input_nodes(h)
        h_neur_input_inh = self.get_neuron_state_input_inh_nodes(h)
        h_neur_evidence = self.get_neuron_state_evidence_nodes(h)
        h_adapt = self.get_adaptation_state(h)

        closeness = self._closeness(self._f(h_neur))
        r_evidence = self.get_neuron_rate_evidence_nodes(h)

        return self.combine_state_space(
            self._get_d_h_neur_difference_nodes(h_neur, h_prediction, h_neur_input),
            self._get_d_h_neur_prediction_nodes(h_prediction, h_neur),
            self._get_d_h_neur_input_nodes(
                h_neur_input, I_input, h_neur_input_inh, I_inh
            ),
            self._get_d_h_neur_input_inh_nodes(h_neur_input_inh, h_neur_input, I_inh),
            self._get_d_h_neur_evidence_nodes(
                h_neur_evidence, closeness, h_adapt, r_evidence
            ),
            self._get_d_h_adapt(h_adapt, r_evidence, closeness),
        )


def convergence_time(model, h, t):
    """Determines when the evidence neurons signal convergence within a stage.

    A neuron signals convergence when its output rises above 0.5. Signals that are already on at the stage onset
    stem from the previous stage and are ignored. With several difference neurons, the input has converged once
    every neuron has signaled.

    Args:
        model: The simulated `MonitoringModel`.
        h: Array of states for the time points `t`.
        t: Time points of the stage.

    Returns:
        tuple: Stage onset, time of the convergence signal and their difference. The latter two are `nan` if
        convergence was not signaled.
    """
    on = np.array([model.get_neuron_rate_evidence_nodes(x) for x in h]) >= 0.5
    first_rise = []
    for on_i in on.T:
        rises = np.where(on_i[1:] & ~on_i[:-1])[0] + 1
        if rises.size == 0:
            return t[0], np.nan, np.nan
        first_rise.append(rises[0])
    t_conv = t[max(first_rise)]
    return t[0], t_conv, t_conv - t[0]
