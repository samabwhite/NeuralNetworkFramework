import numpy as np
from layers.error_layer import ErrorLayer

class MeanSquaredError(ErrorLayer):
    def forwardPass(self, predicted, labels):
        self.networkOutput = predicted
        self.labels = labels
        self.error = self.meanSquaredError(predicted, labels)
        return self.error

    def backwardPass(self):
        self.dA = self.derivMSE(self.networkOutput, self.labels)

    def meanSquaredError(self, predicted, labels):
        return np.mean((predicted - labels) ** 2)

    def derivMSE(self, predicted, labels):
        return 2 * (predicted - labels) / labels.size
