"""Closed-loop coupling of competitive queueing, neural pointers and execution monitoring."""

from collections import namedtuple

import numpy as np

CouplingParams = namedtuple(
    "CouplingParams",
    [
        "pointer_gain",
        "arousal_gain",
        "arousal_threshold",
        "tau_arousal",
        "reset_gain",
        "reset_gain_processing",
        "reset_threshold",
        "tau_reset",
        "go_arousal",
    ],
)


class CoupledModel:
    """Sequential execution of mental operations by the three components in a closed loop.

    * The competitive queueing network (CQ) selects the next operation whenever the arousal signal `J` is on.
    * The output of the selected operation activates its pointer, `I_c_j = min(pointer_gain * s(y_j), 1)`, and the
      pointer re-instantiates the bound operation in the pointer domain (read-out).
    * The retrieved operation drives the monitoring network with its processing pattern,
      `I_DC = P r_domain`, where column `k` of `P` is the input pattern of operation `k`.
    * Once the processing has settled, the monitoring network signals convergence. The signal drives the arousal neuron, which makes
      the CQ select the next operation, and with a delay a reset neuron. The reset is a global inhibition: it clears
      the pointer domain, so that the next operation can win the read-out, and the input layer of the monitoring
      network (the database), so
      that the next operation is instantiated into a defined state and its processing time is set by its own
      dynamics:

          tau_arousal * dJ/dt = -J + arousal_gain * max(r_conv - arousal_threshold, 0) / (1 - arousal_threshold)
          tau_reset * dR/dt   = -R + J
          reset               = max(R - reset_threshold, 0)
          I_inh_domain        = reset_gain * reset
          I_inh_database      = reset_gain_processing * reset

    The loop is closed during the stages with `closed_loop = True`; before that, the operations are bound to the
    pointers by externally scripted stages. The full state is the concatenation of the states of the CQ, the pointer
    model, the monitoring network, `J` and `R`.
    """

    def __init__(self, cq, pointers, monitoring, processing_patterns, config):
        """Model initialization.

        Args:
            cq: `CompetitiveQueueingModel`.
            pointers: `SynapticPointerModel` with one domain neuron per operation.
            monitoring: `MonitoringModel`.
            processing_patterns: Array `P` of shape `(n_inputs, n_operations)`, the pattern processed by the monitoring network for
                each operation.
            config: Config with the fields of `CouplingParams`.
        """
        self.cq, self.pointers, self.monitoring = cq, pointers, monitoring
        self._patterns = np.asarray(processing_patterns)
        self._params = CouplingParams(
            **{field: config[field] for field in CouplingParams._fields}
        )
        sizes = [len(cq.h_0), len(pointers.h_0), len(monitoring.h_0), 1, 1]
        self._bounds = np.cumsum([0] + sizes)
        self.h_0 = np.concatenate([cq.h_0, pointers.h_0, monitoring.h_0, [0.0, 0.0]])

    def split(self, h):
        """Splits the combined state into the states of the CQ, the pointer model and the monitoring network, `J` and `R`."""
        b = self._bounds
        h_cq, h_pt, h_mon, J, R = (h[b[i] : b[i + 1]] for i in range(5))
        return h_cq, h_pt, h_mon, J[0], R[0]

    def convergence_signal(self, h_mon):
        """Convergence signal of the monitoring network; with several difference neurons, all of them have to signal."""
        return np.min(self.monitoring.get_neuron_rate_evidence_nodes(h_mon))

    def pointer_input(self, h_cq):
        """Input to the pointers from the output of the CQ."""
        return np.minimum(
            self._params.pointer_gain * self.cq.get_neuron_rate_output_nodes(h_cq), 1.0
        )

    def __call__(
        self, h, t, cq_input, pt_input, pt_pointers, pt_inh, arousal, closed_loop
    ):
        """Computes the temporal derivative of the combined state, following the interface of
        `scipy.integrate.odeint`.

        Args:
            h: Combined state.
            t: Point in time. Unused, the dynamics are time invariant.
            cq_input: Input pattern of the CQ (the queue of operations).
            pt_input: Driving input to the pointer domain.
            pt_pointers: External pointer input (ignored in the closed loop).
            pt_inh: External reset of the pointer domain (ignored in the closed loop).
            arousal: External arousal of the CQ, added to `J`.
            closed_loop: Whether the components are coupled.

        Returns:
            np.array: Temporal derivative of the combined state.
        """
        ps = self._params
        h_cq, h_pt, h_mon, J, R = self.split(h)
        if closed_loop:
            pointer_input = self.pointer_input(h_cq)
            reset = max(R - ps.reset_threshold, 0.0)
            domain_reset = np.full(len(pt_input), ps.reset_gain * reset)
            database_reset = ps.reset_gain_processing * reset
            monitoring_input = self._patterns @ self.pointers.get_neuron_rates(h_pt)
            drive = (
                ps.arousal_gain
                * max(self.convergence_signal(h_mon) - ps.arousal_threshold, 0.0)
                / (1.0 - ps.arousal_threshold)
            )
        else:
            pointer_input, domain_reset = pt_pointers, pt_inh
            database_reset = 0.0
            monitoring_input = np.zeros(self._patterns.shape[0])
            drive = 0.0

        return np.concatenate(
            [
                self.cq(h_cq, t, cq_input, arousal + J),
                self.pointers(h_pt, t, pt_input, pointer_input, domain_reset),
                self.monitoring(h_mon, t, monitoring_input, database_reset),
                [(-J + drive) / ps.tau_arousal, (-R + J) / ps.tau_reset],
            ]
        )
