import numpy as np
from layers import ActivationLayer

class SoftMaxActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        if self.input.ndim == 1:
            exp_values = np.exp(self.input - np.max(self.input))
            self.output = exp_values / np.sum(exp_values)
        else:
            exp_values = np.exp(self.input - np.max(self.input, axis=1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backwardPass(self, dA):
        if self.input.ndim == 1:
            single_output = self.output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            return np.dot(jacobian_matrix, dA)
        else:
            batch_size = self.input.shape[0]
            dZ = np.empty_like(dA)
            for i in range(batch_size):
                single_output = self.output[i].reshape(-1, 1)
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                dZ[i] = np.dot(jacobian_matrix, dA[i])
            return dZ