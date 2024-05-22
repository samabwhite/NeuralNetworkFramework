import numpy as np
from layers import HiddenLayer
from activations import SigmoidActivation
from activations import SoftMaxActivation
from activations import ReLUActivation
from activations import TanhActivation
from activations import ELUActivation
from errors import MeanSquaredError
from errors import CategoricalCrossEntropy
from errors import BinaryCrossEntropy
from errors import KLDivergence
from errors import CosineSimilarityLoss

class NeuralNetwork:
    def __init__(self, dataset, labels, epochs, alpha, batch_size=32):
        self.dataset = dataset
        self.labels = labels
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = []
        self.errorLayer = None
        self.complete = False
        self.errors = []

    def addHiddenLayer(self, inputSize, outputSize):
        if self.complete:
            print("Can't add layers after the error layer.")
            return
        newLayer = HiddenLayer(inputSize, outputSize)
        if len(self.layers) == 0:
            self.layers.append(newLayer)
        else:
            lastLayerOutputSize = self.layers[-1].weights.shape[1]
            if newLayer.weights.shape[0] != lastLayerOutputSize:
                print(f"Incompatible layer size. Expected input size {lastLayerOutputSize}, got {newLayer.weights.shape[0]}.")
                return
            self.layers.append(newLayer)

    def addActivation(self, layerNumber, activationType):
        if 0 <= layerNumber < len(self.layers):
            if activationType == "Sigmoid":
                self.layers[layerNumber].setActivation(SigmoidActivation())
            elif activationType == "Softmax":
                self.layers[layerNumber].setActivation(SoftMaxActivation())
            elif activationType == "ReLU":
                self.layers[layerNumber].setActivation(ReLUActivation())
            elif activationType == "Tanh":
                self.layers[layerNumber].setActivation(TanhActivation())
            elif activationType == "ELU":
                self.layers[layerNumber].setActivation(ELUActivation())
            else:
                print("Unknown activation type")
        else:
            print(f"Layer number {layerNumber} is out of range.")

    def addError(self, errorType):
        if self.complete:
            print("Error layer already added.")
            return
        if errorType == "Mean Squared Error":
            self.errorLayer = MeanSquaredError()
        elif errorType == "Categorical Cross Entropy":
            self.errorLayer = CategoricalCrossEntropy()
        elif errorType == "Binary Cross Entropy":
            self.errorLayer = BinaryCrossEntropy()
        elif errorType == "KLDivergence":
            self.errorLayer = KLDivergence()
        elif errorType == "Cosine Similarity":
            self.errorLayer = CosineSimilarityLoss()
        else:
            print("Unknown error type")
        self.complete = True

    def removeLayer(self, layerNumber):
        if layerNumber == 0:
            print("Can't remove the input layer from the network.")
        elif 0 <= layerNumber < len(self.layers):
            del self.layers[layerNumber]
        else:
            print(f"Layer number {layerNumber} is out of range.")

    def forwardPass(self, data, label=None):
        output = data
        for layer in self.layers:
            output = layer.forwardPass(output)
        if label is not None:
            self.errorLayer.forwardPass(output, label)
        return output

    def backwardPass(self):
        self.errorLayer.backwardPass()
        dA = self.errorLayer.dA
        for layer in reversed(self.layers):
            dA = layer.backwardPass(dA)

    def updateWeights(self):
        for layer in self.layers:
            layer.update(self.alpha)

    def train(self):
        if not self.complete:
            print("Error Layer Missing: Network not complete.")
            return
        num_batches = int(np.ceil(len(self.dataset) / self.batch_size))
        for epoch in range(self.epochs):
            epoch_error = 0
            for batch_index in range(num_batches):
                batch_start = batch_index * self.batch_size
                batch_end = min((batch_index + 1) * self.batch_size, len(self.dataset))
                batch_data = self.dataset[batch_start:batch_end]
                batch_labels = self.labels[batch_start:batch_end]
                
                self.forwardPass(batch_data, batch_labels)
                self.backwardPass()
                self.updateWeights()
                epoch_error += self.errorLayer.error
            self.errors.append(epoch_error / num_batches)
            print(f"Epoch {epoch+1}/{self.epochs}, Error: {self.errors[-1]}")

    def displayNetworkError(self):
        print(f"Error: {self.errorLayer.error}")
