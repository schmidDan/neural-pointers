"""Components for a dynamical model of neural pointers.
"""

from collections import namedtuple
from enum import Enum, auto
import numpy as np

def id(h):
    r = h
    return r

def relu(h):
    r = np.copy(h)
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
    r = np.copy(h)
    r[h < thresh] = 0.0
    return r

def step(h, thresh):
    r = np.copy(h)
    r[h < thresh] = 0.0
    r[h > thresh] = 1.0
    return r

DCNeuronParams = namedtuple(
    "DCNeuronParams", ["alpha", "beta", "gamma", "eta", "tau", "alpha_i", "beta_i", "tau_i", "alpha_inp", "beta_inp", "lambda_inp", "gamma_inp", "eta_inp", "tau_inp", "alpha_evdc", "beta_evdc", "tau_evdc", "thresh_evdc"]
)
DCModelParams = namedtuple(
    "ModelParams", ["DCNeuronParams"]
)


class DifferenceComputationModel:
    """... computes temporal differences
    """

    def __init__(self, n_neurons, config):
        """...

        Args:
            n_neurons: Number of domain neurons to be modeled.
            config: `hydra` config containing all ODE parameters.
        """
        self._set_params(config)

        #self._f = relu
        self._f = id
        #self._g = relu
        self._g = id
        self._s = relu
        self._t = step
        #self._h = rect
        #self._t = lambda x: type2(x, thresh=self._get_params().CQNeuronParams.theta_t)

        self._n_neurons = n_neurons

        n_neurons_inp = 7*n_neurons
        self._n_neurons_inp = n_neurons_inp

        weight_rng = np.random.default_rng(seed=132)
        
        self._weights_exc = weight_rng.uniform(0,1,(n_neurons_inp,n_neurons_inp))
        self._weights_exc = self._weights_exc / np.sum(self._weights_exc, axis=1)
        self._weights_inh = weight_rng.uniform(0,1,(n_neurons_inp,n_neurons_inp))
        self._weights_inh = self._weights_inh / np.sum(self._weights_inh, axis=1)

        h_0_neur = np.zeros(n_neurons)
        h_0_neur_i = np.zeros(n_neurons)
        h_0_neur_inp = np.zeros(n_neurons_inp)
        h_0_neur_inp_inh = np.zeros(n_neurons_inp)
        h_0_neur_evdc = np.zeros(n_neurons)
        self._h_0 = self.combine_state_space(h_0_neur, h_0_neur_i, h_0_neur_inp, h_0_neur_inp_inh, h_0_neur_evdc)

        self._t_last = 0.0
        self._t_start = -1.0

    def _set_params(self, config):
        nps = DCNeuronParams(**config.neuron_params)
        mps = DCModelParams(DCNeuronParams=nps)
        self._params = mps
        return mps

    def get_neuron_state_exc_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[:self._n_neurons]
    
    
    def get_neuron_rate_exc_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._f(h[:self._n_neurons])

    def get_neuron_state_inh_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[self._n_neurons:2*self._n_neurons]

    def get_neuron_rate_inh_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._g(h[self._n_neurons:2*self._n_neurons])

    
    def get_neuron_state_inp_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[2*self._n_neurons:2*self._n_neurons+self._n_neurons_inp]
    
    
    def get_neuron_rate_inp_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._s(h[2*self._n_neurons:2*self._n_neurons+self._n_neurons_inp])

    
    def get_neuron_state_inp_inh_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[2*self._n_neurons+self._n_neurons_inp:2*self._n_neurons+2*self._n_neurons_inp]
    
    
    def get_neuron_rate_inp_inh_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._s(h[2*self._n_neurons+self._n_neurons_inp:2*self._n_neurons+2*self._n_neurons_inp])

    def get_neuron_state_evdc_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return h[2*self._n_neurons+2*self._n_neurons_inp:]
    
    def get_neuron_rate_evdc_nodes(self, h):
        """Getter method for the neuron states from the combined state `h`.

        Args:
            h: Combined system state from which the neuron state should be retrieved.
        """
        return self._t(h[2*self._n_neurons+2*self._n_neurons_inp:], self._get_params().DCNeuronParams.thresh_evdc)
    
    
    def combine_state_space(self, h_neur, h_neur_i, h_neur_inp, h_neur_inp_inh, h_neur_evdc):
        """Combines the two subspaces of the state space via concatenation into an overall system state.

        Args:
            h_syn: Synaptic state.
            h_neur: Neuron state.

        Returns:
            np.array: Overall system state.
        """
        return np.concatenate((h_neur, h_neur_i, h_neur_inp, h_neur_inp_inh, h_neur_evdc), axis=0)

    def _get_params(self):
        return self._params

    def _get_d_h_neur_exc_nodes(self, h_neur, h_neur_i, h_neur_inp):
        """
        dx_i = ...
        """

        ps = self._get_params().DCNeuronParams

        exc = np.sum(np.abs(self._s(h_neur_inp)))
        
        inh = self._g(h_neur_i)

        d_h_neur = (
            -ps.alpha * h_neur          # passive decay
            + (ps.beta - h_neur) * exc # shunted excitation
            - (ps.eta + ps.gamma * h_neur) * inh # shunted inhibition
        ) / ps.tau
        return d_h_neur

    def _get_d_h_neur_inh_nodes(self, h_neur_i, h_neur):
        """
        dx_i = ...
        """

        ps = self._get_params().DCNeuronParams

        exc = self._f(h_neur)
        
        d_h_neur_i = (
            -ps.alpha_i * h_neur_i          # passive decay
            + (ps.beta_i - h_neur_i) * exc # shunted excitation
        ) / ps.tau_i
        return d_h_neur_i

    def _get_d_h_neur_inp_nodes(self, h_neur_inp, I_inp, h_neur_inp_inh):
        """
        dx_i = ...
        """

        ps = self._get_params().DCNeuronParams

        #exc = I_inp + ps.lambda_inp * self._s(h_neur_inp)
        exc = I_inp + ps.lambda_inp * np.dot(self._weights_exc, self._s(h_neur_inp))
        
        #inh = np.sum(self._s(h_neur_inp))
        #inh = np.dot(self._weights_inh, self._s(h_neur_inp))
        inh = h_neur_inp_inh
        
        #print("------------------")
        #print(h_neur_inp)
        
        #print((ps.eta_inp + ps.gamma_inp * h_neur_inp) * inh)
        
        d_h_neur_inp = (
            -ps.alpha_inp * h_neur_inp          # passive decay
            + (ps.beta_inp - h_neur_inp) * exc # shunted excitation
            - (ps.eta_inp + ps.gamma_inp * h_neur_inp) * inh # shunted inhibition
        ) / ps.tau_inp

        #print(d_h_neur_inp)
        #print("___________________")
        
        return d_h_neur_inp

    def _get_d_h_neur_inp_inh_nodes(self, h_neur_inp_inh, h_neur_inp):
        """
        dx_i = ...
        """

        ps = self._get_params().DCNeuronParams

        #exc = I_inp + ps.lambda_inp * self._s(h_neur_inp)
        exc = np.dot(self._weights_inh, self._s(h_neur_inp))
                        
        d_h_neur_inp_inh = (
            -ps.alpha_inp * h_neur_inp_inh          # passive decay
            + (ps.beta_inp - h_neur_inp_inh) * exc # shunted excitation
            #- (ps.eta_inp + ps.gamma_inp * h_neur_inp_inh) * inh # shunted inhibition
        ) / ps.tau_inp
        
        return d_h_neur_inp_inh

    def _get_d_h_neur_evdc_nodes(self, h_neur_evdc, h_neur, t):
        """
        dx_i = ...
        """

        ps = self._get_params().DCNeuronParams

        #print(t)

        exc = (np.abs(self._f(h_neur)) < 0.001) * (t > (20.0 + self._t_last))
        
        d_h_neur_evdc = (
            -ps.alpha_evdc * h_neur_evdc          # passive decay
            + (ps.beta_evdc - h_neur_evdc) * exc # shunted excitation
        ) / ps.tau_evdc
        return d_h_neur_evdc

    def __call__(self, h, t, I_input):
        """Main method by which the model can be queried for a temporal derivative of its state values based on the
        current state `h`, a point in time `t`, driving inputs to the domain `I_input` and modulating input from
        pointers `I_pointers` and the internal model parameters and update equations `_get_d_h_neur` and
        `_get_d_h_syn` implementing the respective ODEs.

        Parameters:
            h: Concatenation of states along axis 0. First part synaptic pointer states, last `n_neurons` entries
            neuron states.
            t: Point in time as required by the interface of `scipy.integrate.odeint`. So far unused as the ODE is time
            invariant.
            .....

        Returns:
            np.array: Temporal derivative of the model's state values.
        """
        h_neur = self.get_neuron_state_exc_nodes(h)
        h_neur_i = self.get_neuron_state_inh_nodes(h)
        h_neur_inp = self.get_neuron_state_inp_nodes(h)
        h_neur_inp_inh = self.get_neuron_state_inp_inh_nodes(h)
        h_neur_evdc = self.get_neuron_state_evdc_nodes(h)

        
        r_neur_evdc = self.get_neuron_rate_evdc_nodes(h)

#        if r_neur_evdc:
#            print("********")
#            print(h_neur_evdc)
#            print(r_neur_evdc)
#            print("********")
        
        if r_neur_evdc and (self._t_start < self._t_last + 10.0):
            self._t_start = t
#            print("xxxxxxxxxxxxx")
#            print(self._t_start)
#            print("xxxxxxxxxxxxx")
        
        if r_neur_evdc and (t > (self._t_start + 10.0)):
#            print("....")
#            print(self._t_last)
            self._t_last = t            
#            print(self._t_last)
#            print("....")
#        print("++++++++++++++++++++++++++")
#        print(h)
#        print(h_neur_inp)
#        print("//////////////////////////")

        # compute neuron state 
        d_h_neur = self._get_d_h_neur_exc_nodes(h_neur, h_neur_i, h_neur_inp)
        d_h_neur_i = self._get_d_h_neur_inh_nodes(h_neur_i, h_neur)
        d_h_neur_inp = self._get_d_h_neur_inp_nodes(h_neur_inp, I_input, h_neur_inp_inh)
        d_h_neur_inp_inh = self._get_d_h_neur_inp_inh_nodes(h_neur_inp_inh, h_neur_inp)
        d_h_neur_evdc = self._get_d_h_neur_evdc_nodes(h_neur_evdc, h_neur, t)
        

        return self.combine_state_space(d_h_neur, d_h_neur_i, d_h_neur_inp, d_h_neur_inp_inh, d_h_neur_evdc)
