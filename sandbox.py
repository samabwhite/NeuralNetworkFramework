import NeuralNetworkFramework as nnf
import numpy as np

dataset = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 4, 5], [6, 5, 4, 3], [1, 3, 5, 1]])
oneHotLabels = np.array([[0, 1], [0, 1], [1, 0], [0, 1], [0, 1]])

network = nnf.NeuralNetwork(dataset, oneHotLabels, 5, 0.1)
network.addHiddenLayer(4, 3, 1)
network.addActivation(1, "Sigmoid")
network.addHiddenLayer(3, 3, 2)
network.addActivation(2, "Sigmoid")
network.addHiddenLayer(3, 2, 3)
network.addActivation(3, "Softmax")
network.addError()

network.train() 





