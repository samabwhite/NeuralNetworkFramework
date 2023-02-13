import NeuralNetworkFramework as nnf
import numpy as np
import pandas as pd

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




