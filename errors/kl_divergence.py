import numpy as np
from layers import ErrorLayer

class KLDivergence(ErrorLayer):
    def forwardPass(self, predicted, labels):
        self.networkOutput = predicted
        self.labels = labels
        self.error = self.klDivergence(predicted, labels)
        return self.error

    def backwardPass(self):
        self.dA = self.derivKL(self.networkOutput, self.labels)

    def klDivergence(self, predicted, labels):
        return np.sum(labels * np.log(labels / (predicted + 1e-20) + 1e-20))

    def derivKL(self, predicted, labels):
        return -labels / (predicted + 1e-20)
