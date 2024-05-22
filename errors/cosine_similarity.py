import numpy as np
from layers import ErrorLayer

class CosineSimilarityLoss(ErrorLayer):
    def forwardPass(self, predicted, labels):
        self.networkOutput = predicted
        self.labels = labels
        self.error = self.cosineSimilarity(predicted, labels)
        return self.error

    def backwardPass(self):
        self.dA = self.derivCosine(self.networkOutput, self.labels)

    def cosineSimilarity(self, predicted, labels):
        dot_product = np.sum(predicted * labels, axis=-1)
        norm_predicted = np.linalg.norm(predicted, axis=-1)
        norm_labels = np.linalg.norm(labels, axis=-1)
        return 1 - dot_product / (norm_predicted * norm_labels + 1e-20)

    def derivCosine(self, predicted, labels):
        dot_product = np.sum(predicted * labels, axis=-1, keepdims=True)
        norm_predicted = np.linalg.norm(predicted, axis=-1, keepdims=True)
        norm_labels = np.linalg.norm(labels, axis=-1, keepdims=True)
        return (labels / (norm_predicted * norm_labels + 1e-20)) - \
               (dot_product * predicted / (norm_predicted**3 * norm_labels + 1e-20))
