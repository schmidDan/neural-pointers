"""Components for a dynamical model of neural pointers.
"""

from collections import namedtuple
from enum import Enum, auto
import numpy as np


def relu(h):
    #r = h
    r = np.copy(h) # fixing the bug!
    r[h < 0.0] = 0.0
    return r

def sigmoid(h, f_0=1.0, beta=1.0, theta=0.5):
#def sigmoid(h, f_0=1.0, beta=5.0, theta=0.5):
    return f_0 / (1.0 + np.exp(-beta * (h - theta)))

def square(h):
    r = np.square(relu(h))
    return r

#def type2(h, thresh=0.1):
#def type2(h, thresh=0.25):
def type2(h, thresh):
    #r = h
    r = np.copy(h) # fixing the bug!
    r[h < thresh] = 0.0
    return r

CQNeuronParams = namedtuple(
    "CQNeuronParams", ["A_x", "B_x", "C_x", "tau_x", "G", "epsilon", "A_y", "B_y", "C_y", "tau_y", "A_w", "B_w", "tau_w", "theta_t"]
)
CQModelParams = namedtuple(
    "ModelParams", ["CQNeuronParams"]
)


class CompetitiveQueueingModel:
    """... cf. Boardman & Bullock, IJCNN, 1991
    """

    def __init__(self, n_neurons, config):
        """...

        Args:
            n_neurons: Number of domain neurons to be modeled.
            config: `hydra` config containing all ODE parameters.
        """
        self._set_params(config)

        self._f = relu
        self._g = relu
        self._h = relu
        self._q = relu
        self._r = square
        self._s = square
        self._t = lambda x: type2(x, thresh=self._get_params().CQNeuronParams.theta_t)

        self._n_neurons = n_neurons

        h_0_neur_x = np.zeros(n_neurons)
        h_0_neur_y = np.zeros(n_neurons)
        h_0_neur_w = np.zeros(n_neurons)
        self._h_0 = self.combine_state_space(h_0_neur_x, h_0_neur_y, h_0_neur_w)

    def _set_params(self, config):
        nps = CQNeuronParams(**config.neuron_params)
        mps = CQModelParams(CQNeuronParams=nps)
        self._params = mps
        return mps

    def get_neuron_state_input_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[:self._n_neurons]
    
    
    def get_neuron_rate_input_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._f(h[:self._n_neurons])

    def get_neuron_state_output_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[self._n_neurons:2*self._n_neurons]

    def get_neuron_rate_output_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._s(h[self._n_neurons:2*self._n_neurons])

    def get_neuron_state_inhib_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[2*self._n_neurons:]

    def get_neuron_rate_inhib_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._h(h[2*self._n_neurons:])

    def combine_state_space(self, h_neur_x, h_neur_y, h_neur_w):
        """Combines the two subspaces of the state space via concatenation into an overall system state.

        Args:
            h_syn: Synaptic state.
            h_neur: Neuron state.

        Returns:
            np.array: Overall system state.
        """
        return np.concatenate((h_neur_x, h_neur_y, h_neur_w), axis=0)

    def _get_params(self):
        return self._params

    def _get_d_h_neur_input_nodes(self, h_neur_x, h_neur_w, h_neur_I, h_neur_J):
        """
        dx_i = -A_x * x_i # passive decay
               + (B_x - x_i)*(f(x_i - I_i) / S + I_i) # shunted excitation
               - (C_x + x_i)*(sum_N_i=!=j(g(x_j-I_j)+h(w_i))) # shunted inhibition
        """

        ps = self._get_params().CQNeuronParams

        h_S = ps.G / (ps.epsilon + np.sum(self._f(h_neur_x))) * (h_neur_J > 0) + 1.0  * (h_neur_J <= 0)

        print(h_S)
        
######        exc = self._f(h_neur_x - h_neur_I) / h_S + h_neur_I
        exc = self._f(h_neur_x - h_neur_I) / h_S + h_neur_I
#        exc = self._f(h_neur_x - h_neur_I) * h_S + h_neur_I
        #exc = self._f(h_neur_x) * h_S + h_neur_I
        
        #inh = np.sum(self._g(h_neur_x - h_neur_I) + self._h(h_neur_w)) - self._g(h_neur_x - h_neur_I)
        inh = np.sum(self._g(h_neur_x - h_neur_I)) - self._g(h_neur_x - h_neur_I) + self._h(h_neur_w)
#        inh = np.sum(self._g(h_neur_x - h_neur_I)) - self._g(h_neur_x - h_neur_I) # testing w/o feedback

        d_h_neur_x = (
            -ps.A_x * h_neur_x          # passive decay
            + (ps.B_x - h_neur_x) * exc # shunted excitation
            - (ps.C_x + h_neur_x) * inh # shunted inhibition
        ) / ps.tau_x
        return d_h_neur_x

    def _get_d_h_neur_output_nodes(self, h_neur_y, h_neur_x, h_neur_J):
        """
        dx_i = -A_y * y_i # passive decay
               + (B_y - y_i)*(s(y_i) + J * q(x_i)) # shunted excitation
               - (C_y + y_i)*(sum_N_i=!=j(r(y_i))) # shunted inhibition
        """

        ps = self._get_params().CQNeuronParams

        exc = self._s(h_neur_y) + h_neur_J * self._q(h_neur_x)
        
        inh = np.sum(self._r(h_neur_y)) - self._r(h_neur_y)

        d_h_neur_y = (
            -ps.A_y * h_neur_y          # passive decay
            + (ps.B_y - h_neur_y) * exc # shunted excitation
            - (ps.C_y + h_neur_y) * inh # shunted inhibition
        ) / ps.tau_y
        return d_h_neur_y

    def _get_d_h_neur_inhib_nodes(self, h_neur_y, h_neur_w):
        """
        dx_i = -A_w * w_i # passive decay
               + (B_w - w_i)*(t(y_i)) # shunted excitation
        """

        ps = self._get_params().CQNeuronParams

        exc = self._t(h_neur_y)
        
        d_h_neur_w = (
            -ps.A_w * h_neur_w          # passive decay
            + (ps.B_w - h_neur_w) * exc # shunted excitation
        ) / ps.tau_w
        return d_h_neur_w
 
    def __call__(self, h, t, I_input, I_J):
        """Main method by which the model can be queried for a temporal derivative of its state values based on the
        current state `h`, a point in time `t`, driving inputs to the domain `I_input` and modulating input from
        pointers `I_pointers` and the internal model parameters and update equations `_get_d_h_neur` and
        `_get_d_h_syn` implementing the respective ODEs.

        Parameters:
            h: Concatenation of states along axis 0. First part synaptic pointer states, last `n_neurons` entries
            neuron states.
            t: Point in time as required by the interface of `scipy.integrate.odeint`. So far unused as the ODE is time
            invariant.
            I_J: Arousal state.
            I_pointers: Modulating input to the domain neurons. Also used by the synapse dynamics to update their
            pointer values.

        Returns:
            np.array: Temporal derivative of the model's state values.
        """
        h_neur_x = self.get_neuron_state_input_nodes(h)
        h_neur_y = self.get_neuron_state_output_nodes(h)
        h_neur_w = self.get_neuron_state_inhib_nodes(h)

        # compute neuron state update
        d_h_neur_x = self._get_d_h_neur_input_nodes(h_neur_x, h_neur_w, I_input, I_J)
        d_h_neur_y = self._get_d_h_neur_output_nodes(h_neur_y, h_neur_x, I_J)
        d_h_neur_w = self._get_d_h_neur_inhib_nodes(h_neur_y, h_neur_w)

        return self.combine_state_space(d_h_neur_x, d_h_neur_y, d_h_neur_w)
