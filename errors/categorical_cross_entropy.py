import numpy as np
from layers import ErrorLayer

class CategoricalCrossEntropy(ErrorLayer):
    def forwardPass(self, predicted, labels):
        self.networkOutput = predicted
        self.labels = labels
        self.error = self.categoricalCrossEntropy(predicted, labels)
        return self.error

    def backwardPass(self):
        self.dA = self.derivCCE(self.networkOutput, self.labels)

    def categoricalCrossEntropy(self, predicted, labels):
        return -np.sum(labels * np.log(predicted + 1e-20))

    def derivCCE(self, predicted, labels):
        return -labels / (predicted + 1e-20)
