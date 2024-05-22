# Neural Network Framework
A simple neural network framework based on the Builder Design Pattern, implemented from scratch using Python and NumPy. This framework contains various types of layers, activation functions, and error functions, which makes it easy to build and train neural networks.

## Features
- Support for different layer types (Hidden Layer, Activation Layer, Error Layer)
- Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax, ELU)
- Multiple error functions (Mean Squared Error, Categorical Cross-Entropy, Binary Cross-Entropy, KL Divergence, Cosine Similarity Loss)
- Training with batch processing

## Usage
### Basic Implementation
```python
import numpy as np
from models import NeuralNetwork

# Sample dataset
dataset = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

epochs = 10000
alpha = 0.1

# Create neural network object
nn = NeuralNetwork(dataset, labels, epochs, alpha)

# Add hidden and activation layers
nn.addHiddenLayer(inputSize=2, outputSize=2)
nn.addActivation(layerNumber=0, activationType="Sigmoid")

nn.addHiddenLayer(inputSize=2, outputSize=1)
nn.addActivation(layerNumber=1, activationType="Sigmoid")

# Add error layer
nn.addError(errorType="Mean Squared Error")

# Training network
nn.train()

# Display output and error
for data, label in zip(dataset, labels):
    output = nn.forwardPass(data)
    print(f"Input: {data}, Predicted: {output}, Actual: {label}")
```

### Batch Processing
```python
# Parameters
epochs = 50
alpha = 0.01
batch_size = 32

# Create neural network object
nn = NeuralNetwork(dataset, one_hot_labels, epochs, alpha, batch_size)
```

### Using Different Activation Functions
```python
# Add hidden and activation layers
nn.addHiddenLayer(inputSize=10, outputSize=20)
nn.addActivation(layerNumber=0, activationType="ReLU")

nn.addHiddenLayer(inputSize=20, outputSize=10)
nn.addActivation(layerNumber=1, activationType="Tanh")

nn.addHiddenLayer(inputSize=10, outputSize=5)
nn.addActivation(layerNumber=2, activationType="ELU")

nn.addHiddenLayer(inputSize=5, outputSize=1)
nn.addActivation(layerNumber=3, activationType="Sigmoid")
```

### Using Different Error Functions
```python
# Add error layer
nn.addError(errorType="Categorical Cross Entropy")
# or
nn.addError(errorType="Binary Cross Entropy")
# or
nn.addError(errorType="Mean Squared Error")
# or
nn.addError(errorType="KLDivergence")
# or
nn.addError(errorType="Cosine Similarity")
```