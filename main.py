import numpy as np
from layers.hidden_layer import HiddenLayer
from activations.sigmoid import SigmoidActivation
from activations.softmax import SoftMaxActivation
from activations.relu import ReLUActivation
from activations.tanh import TanhActivation
from activations.elu import ELUActivation
from errors.mean_squared_error import MeanSquaredError
from errors.categorical_cross_entropy import CategoricalCrossEntropy
from errors.binary_cross_entropy import BinaryCrossEntropy
from errors.kl_divergence import KLDivergence
from errors.cosine_similarity import CosineSimilarityLoss

class NeuralNetwork:
    def __init__(self, dataset, labels, epochs, alpha):
        self.dataset = dataset
        self.labels = labels
        self.alpha = alpha
        self.epochs = epochs
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
        for epoch in range(self.epochs):
            epoch_error = 0
            for data, label in zip(self.dataset, self.labels):
                self.forwardPass(data, label)
                self.backwardPass()
                self.updateWeights()
                epoch_error += self.errorLayer.error
            self.errors.append(epoch_error / len(self.dataset))
            print(f"Epoch {epoch+1}/{self.epochs}, Error: {self.errors[-1]}")

    def displayNetworkError(self):
        print(f"Error: {self.errorLayer.error}")

if __name__ == "__main__":

    # Create some dummy data
    dataset = np.random.randn(100, 3)  # 100 samples, 3 features each
    labels = np.random.randn(100, 1)  # 100 samples, 1 label each

    # Initialize neural network
    nn = NeuralNetwork(dataset, labels, epochs=10, alpha=0.01)

    # Add layers
    nn.addHiddenLayer(inputSize=3, outputSize=5)
    nn.addActivation(0, "ReLU")
    nn.addHiddenLayer(inputSize=5, outputSize=1)
    nn.addActivation(1, "Sigmoid")

    # Add error layer
    nn.addError("Mean Squared Error")

    # Train the network
    nn.train()

    # Display final network error
    nn.displayNetworkError()
