import numpy as np
from layers.activation_layer import ActivationLayer

class ELUActivation(ActivationLayer):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forwardPass(self, input):
        self.input = input
        self.output = np.where(self.input > 0, self.input, self.alpha * (np.exp(self.input) - 1))
        return self.output

    def backwardPass(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.input <= 0] = dZ[self.input <= 0] * (self.output[self.input <= 0] + self.alpha)
        return dZ