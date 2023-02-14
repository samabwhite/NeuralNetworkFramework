import NeuralNetworkFramework as nnf
import numpy as np
import pandas as pd


dataset = np.array([1, 2, 3, 4])
labels = np.array([0.01, 0.99])


nn = nnf.NeuralNetwork(dataset, labels, 1, 0.1)

nn.addHiddenLayer(4, 3, 1)
nn.addHiddenLayer(3, 2, 2)


nn.addActivation(1, "Sigmoid")
nn.addActivation(2, "Softmax")

nn.layers[1].weights = np.array([[0.05, 0.25, 0.45], [0.1, 0.3, 0.5], [0.15, 0.35, 0.55], [0.2, 0.4, 0.6]])
nn.layers[1].bias = np.array([0.25, 0.5, 0.75])

nn.layers[2].weights = np.array([[0.65, 0.80], [0.70, 0.85], [0.75, 0.90]])
nn.layers[2].bias = np.array([0.25, 0.5])


layer1Output = np.dot(dataset, nn.layers[1].weights)
print(layer1Output)
layer1Activation = 1 / (1 + np.exp(-layer1Output))
print(layer1Activation)
layer2Output = np.dot(layer1Activation, nn.layers[2].weights)
print(layer2Output)
layer2Adjust = layer2Output - np.max(layer2Output)
print(layer2Adjust)
layer2Activation = np.exp(layer2Adjust) / np.sum(np.exp(layer2Adjust))
print(layer2Activation)
error = labels - layer2Activation
print(error)



























'''
dataset = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
oneHotLabels = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])

network = nnf.NeuralNetwork(dataset, oneHotLabels, len(dataset) - 1, 1)
network.addHiddenLayer(6, 5, 1)
network.addActivation(1, "Sigmoid")
network.addHiddenLayer(5, 4, 2)
network.addActivation(2, "Sigmoid")
network.addHiddenLayer(4, 3, 3)
network.addActivation(3, "Softmax")
network.addError()

network.train() 

'''


'''
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


data = pd.read_csv("C:\\Users\\samwh\\Desktop\\Neural Network\\train.csv")
data = np.array(data)
data = data[0:1000].T
Y = one_hot(data[0]).T
X = data[1:].T 



network = nnf.NeuralNetwork(X, Y, 1000, 0.01)

network.addHiddenLayer(784, 10, 1)
network.addActivation(1, "Sigmoid")
network.addHiddenLayer(10, 10, 2)
network.addActivation(2, "Softmax")
network.addError()

network.train()
'''



