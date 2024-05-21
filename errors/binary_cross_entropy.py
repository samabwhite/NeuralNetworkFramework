import numpy as np
from layers.error_layer import ErrorLayer

class BinaryCrossEntropy(ErrorLayer):
    def forwardPass(self, predicted, labels):
        self.networkOutput = predicted
        self.labels = labels
        self.error = self.binaryCrossEntropy(predicted, labels)
        return self.error

    def backwardPass(self):
        self.dA = self.derivBCE(self.networkOutput, self.labels)

    def binaryCrossEntropy(self, predicted, labels):
        return -np.mean(labels * np.log(predicted + 1e-20) + (1 - labels) * np.log(1 - predicted + 1e-20))

    def derivBCE(self, predicted, labels):
        return (predicted - labels) / ((predicted + 1e-20) * (1 - predicted + 1e-20))
