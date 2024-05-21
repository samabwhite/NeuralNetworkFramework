import numpy as np
from layers.activation_layer import ActivationLayer

class SoftMaxActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        exp_values = np.exp(self.input - np.max(self.input, keepdims=True))
        self.output = exp_values / np.sum(exp_values, keepdims=True)
        return self.output

    def backwardPass(self, dA):
        n = np.size(self.output)
        tmp = np.reshape(np.tile(self.output, n), (n, n))
        return np.dot(tmp * (np.identity(n) - tmp.T), dA)
