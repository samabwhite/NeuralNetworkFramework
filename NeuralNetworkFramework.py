import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forwardPass(self, input):
        pass

    def backwardPass(self, dA):
        pass

    def update(self, alpha):
        pass

class ActivationLayer(Layer):
    def forwardPass(self, input):
        pass

    def backwardPass(self, dA):
        pass

class ErrorLayer(Layer):
    def forwardPass(self, predicted, labels):
        pass

    def backwardPass(self):
        pass

class SigmoidActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backwardPass(self, dA):
        sigmoid_derivative = self.output * (1 - self.output)
        return dA * sigmoid_derivative

class SoftMaxActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        exp_values = np.exp(self.input - np.max(self.input, keepdims=True))
        self.output = exp_values / np.sum(exp_values, keepdims=True)
        return self.output

    def backwardPass(self, dA):
        n = np.size(self.output)
        tmp = np.reshape(np.tile(self.output, n), (n, n))
        return np.dot(tmp * (np.identity(n) - tmp.T), dA)

class ReLUActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output

    def backwardPass(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.input <= 0] = 0
        return dZ

class LeakyReLUActivation(ActivationLayer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forwardPass(self, input):
        self.input = input
        self.output = np.where(self.input > 0, self.input, self.alpha * self.input)
        return self.output

    def backwardPass(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.input <= 0] = self.alpha
        return dZ

class TanhActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        self.output = np.tanh(self.input)
        return self.output

    def backwardPass(self, dA):
        return dA * (1 - np.power(self.output, 2))

class ELUActivation(ActivationLayer):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forwardPass(self, input):
        self.input = input
        self.output = np.where(self.input > 0, self.input, self.alpha * (np.exp(self.input) - 1))
        return self.output

    def backwardPass(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.input <= 0] = dZ[self.input <= 0] * (self.output[self.input <= 0] + self.alpha)
        return dZ

class SwishActivation(ActivationLayer):
    def forwardPass(self, input):
        self.input = input
        self.output = self.input / (1 + np.exp(-self.input))
        return self.output

    def backwardPass(self, dA):
        sigma = 1 / (1 + np.exp(-self.input))
        return dA * (self.output + sigma * (1 - self.output))

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

class HuberLoss(ErrorLayer):
    def __init__(self, delta=1.0):
        self.delta = delta

    def forwardPass(self, predicted, labels):
        self.networkOutput = predicted
        self.labels = labels
        self.error = self.huberLoss(predicted, labels)
        return self.error

    def backwardPass(self):
        self.dA = self.derivHuber(self.networkOutput, self.labels)

    def huberLoss(self, predicted, labels):
        error = predicted - labels
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * error**2
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def derivHuber(self, predicted, labels):
        error = predicted - labels
        is_small_error = np.abs(error) <= self.delta
        return np.where(is_small_error, error, self.delta * np.sign(error))

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
            elif activationType == "LeakyReLU":
                self.layers[layerNumber].setActivation(LeakyReLUActivation())
            elif activationType == "Tanh":
                self.layers[layerNumber].setActivation(TanhActivation())
            elif activationType == "ELU":
                self.layers[layerNumber].setActivation(ELUActivation())
            elif activationType == "Swish":
                self.layers[layerNumber].setActivation(SwishActivation())
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
        elif errorType == "Huber Loss":
            self.errorLayer = HuberLoss()
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
