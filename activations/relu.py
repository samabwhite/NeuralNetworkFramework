import numpy as np
from layers import ActivationLayer

class ReLUActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output

    def backwardPass(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.input <= 0] = 0
        return dZ
