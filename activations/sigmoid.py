import numpy as np
from layers import ActivationLayer

class SigmoidActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backwardPass(self, dA):
        sigmoid_derivative = self.output * (1 - self.output)
        return dA * sigmoid_derivative