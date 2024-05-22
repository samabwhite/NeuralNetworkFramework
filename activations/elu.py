import numpy as np
from layers import ActivationLayer

class ELUActivation(ActivationLayer):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forwardPass(self, input):
        self.input = input
        self.output = np.where(self.input > 0, self.input, self.alpha * (np.exp(self.input) - 1))
        print("Forward Pass:")
        print("Input:", self.input)
        print("Output:", self.output)
        return self.output

    def backwardPass(self, dA):
        dZ = np.array(dA, copy=True)
        print("\nBackward Pass:")
        print("dA:", dA)
        print("dZ before modification:", dZ)
        print("Input dimensions:", self.input.ndim)
        
        if self.input.ndim == 1:
            indices = self.input <= 0
            print("Indices (1D):", indices)
            dZ[indices] *= (self.alpha + self.output[indices])
        else:
            for i in range(self.input.shape[0]):
                indices = self.input[i] <= 0
                print(f"Indices (batch {i}):", indices)
                dZ[i][indices] *= (self.alpha + self.output[i][indices])
        
        print("dZ after modification:", dZ)
        return dZ
