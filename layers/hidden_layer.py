import numpy as np
from layers.layer import Layer

class HiddenLayer(Layer):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.weights = np.random.randn(inputSize, outputSize)
        self.bias = np.random.randn(outputSize)
        self.activationLayer = None

    def forwardPass(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        if self.activationLayer:
            self.output = self.activationLayer.forwardPass(self.output)
        return self.output

    def backwardPass(self, dA):
        if self.activationLayer:
            dA = self.activationLayer.backwardPass(dA)
        dA_prev = np.dot(dA, self.weights.T)
        self.dW = np.dot(self.input.T.reshape(-1, 1), dA.reshape(1, -1))
        self.dB = np.sum(dA, axis=0, keepdims=True)
        return dA_prev

    def update(self, alpha):
        self.weights -= alpha * self.dW
        self.bias -= alpha * self.dB

    def setActivation(self, activationLayer):
        self.activationLayer = activationLayer
