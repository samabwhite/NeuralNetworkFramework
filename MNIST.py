import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from layers.hidden_layer import HiddenLayer
from activations.sigmoid import SigmoidActivation
from activations.softmax import SoftMaxActivation
from errors.categorical_cross_entropy import CategoricalCrossEntropy

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
            else:
                print("Unknown activation type")
        else:
            print(f"Layer number {layerNumber} is out of range.")

    def addError(self, errorType):
        if self.complete:
            print("Error layer already added.")
            return
        if errorType == "Categorical Cross Entropy":
            self.errorLayer = CategoricalCrossEntropy()
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
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data: Flatten the images and normalize the pixel values
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Parameters
    epochs = 10
    alpha = 0.1

    # Create a neural network instance
    nn = NeuralNetwork(x_train, y_train, epochs, alpha)

    # Add layers
    nn.addHiddenLayer(inputSize=784, outputSize=128)  # First hidden layer with 784 inputs and 128 outputs
    nn.addActivation(layerNumber=0, activationType="Sigmoid")

    nn.addHiddenLayer(inputSize=128, outputSize=64)  # Second hidden layer with 128 inputs and 64 outputs
    nn.addActivation(layerNumber=1, activationType="Sigmoid")

    nn.addHiddenLayer(inputSize=64, outputSize=10)  # Output layer with 64 inputs and 10 outputs
    nn.addActivation(layerNumber=2, activationType="Softmax")

    # Add error layer
    nn.addError(errorType="Categorical Cross Entropy")

    # Train the network
    nn.train()

    # Plot the error over epochs
    plt.plot(range(epochs), nn.errors)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training Error over Epochs')
    plt.show()

    # Display final network output and error on a subset of test data
    for data, label in zip(x_test[:10], y_test[:10]):
        output = nn.forwardPass(data)
        print(f"Predicted: {np.argmax(output)}, Actual: {np.argmax(label)}")

    # Calculate accuracy on test data and print guesses vs correct answers
    correct_predictions = 0
    print("\nDetailed Output on Test Data:")
    for data, label in zip(x_test, y_test):
        output = nn.forwardPass(data)
        predicted_label = np.argmax(output)
        actual_label = np.argmax(label)
        print(f"Predicted: {predicted_label}, Actual: {actual_label}")
        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(x_test)
    print(f"\nAccuracy on test data: {accuracy * 100:.2f}%")

    # Output results on a subset of test data
    print("\nFinal network outputs on test data:")
    for data, label in zip(x_test[:10], y_test[:10]):
        output = nn.forwardPass(data)
        predicted_label = np.argmax(output)
        actual_label = np.argmax(label)
        print(f"Predicted: {predicted_label}, Actual: {actual_label}")
