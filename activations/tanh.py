import numpy as np
from layers import ActivationLayer

class TanhActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        self.output = np.tanh(self.input)
        return self.output

    def backwardPass(self, dA):
        return dA * (1 - np.power(self.output, 2))
