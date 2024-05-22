import numpy as np
import tensorflow as tf
from models import NeuralNetwork

# Loading MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Parameters
epochs = 10
alpha = 0.1
batch_size = 32

# Create neural network object
nn = NeuralNetwork(x_train, y_train, epochs, alpha, batch_size)

# Add hidden and activation layers
nn.addHiddenLayer(inputSize=784, outputSize=128)
nn.addActivation(layerNumber=0, activationType="Sigmoid")

nn.addHiddenLayer(inputSize=128, outputSize=64)
nn.addActivation(layerNumber=1, activationType="Sigmoid")

nn.addHiddenLayer(inputSize=64, outputSize=10)
nn.addActivation(layerNumber=2, activationType="Softmax")

# Add error layer
nn.addError(errorType="Categorical Cross Entropy")

# Train the network
nn.train()

# Plot the error as a graph
import matplotlib.pyplot as plt
plt.plot(range(epochs), nn.errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Training Error over Epochs')
plt.show()

# Calculate accuracy
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
