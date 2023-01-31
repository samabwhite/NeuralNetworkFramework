import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None 
    def forwardPass():
        pass
    def backwardPass():
        pass
    def displayActivity(self):
        print("Layer Input: " + str(self.input))
        print("Layer Output: " + str(self.output) + "\n")
        


class HiddenLayer(Layer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(outputSize, inputSize)
        self.bias = np.random.randn(outputSize)
    
    def forwardPass(self, input):
        self.input = input
        self.output = np.dot(self.weights, self.input) + self.bias.T
        return self.output

    def backwardPass():
        pass

    def displayActivity(self):
        print("Hidden Layer Weights: " + str(self.weights))
        print("Hidden Layer Bias: " + str(self.bias))
        print("Hidden Layer Input: " + str(self.input))
        print("Hidden Layer Output: " + str(self.output) + "\n")



class SigmoidActivation(Layer):
    def __init__(self):
        pass

    def forwardPass(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backwardPass():
        pass

    def displayActivity(self):
        print("Sigmoid Input: " + str(self.input))
        print("Sigmoid Output: " + str(self.output) + "\n")



class SoftMaxActivation(Layer):
    def __init__(self):
        pass

    def forwardPass(self, input):
        self.input = input
        self.output = []
        for i in input:
            self.output.append(np.exp(i) / np.sum(np.exp(input)))
        return self.output

    def backwardPass():
        pass

    def displayActivity(self):
        print("SoftMax Input: " + str(self.input))
        print("SoftMax Output: " + str(self.output) + "\n")



class Error(Layer):
    def __init__(self):
        pass

    def forwardPass(self, networkOutput, labels):
        self.networkOutput = networkOutput
        self.labels = labels

        self.totalError = self.totalSquaredError(self.networkOutput, self.labels)
        self.totalErrorDerivative = self.squaredErrorDerivative(self.networkOutput, labels)

    def backwardPass():
        pass

    def totalSquaredError(self, predicted, labels):
        return np.sum((1/2) * np.power(np.subtract(target, predicted), 2))

    def squaredErrorDerivative(self, predicted, labels):
        return np.subtract(predicted, labels)

    def displayActivity(self):
        print("Network Output: " + str(self.networkOutput))
        print("Labels: " + str(self.labels))
        print("Total Error: " + str(self.totalError))
        print("Error Derivative: " + str(self.totalErrorDerivative) + "\n")


#____________________________________________________________________________
# Sandbox Area for Testing Framework

target = [0, 1]
alpha = 0.1

inputs1 = np.array([0.05, 0.10])

hiddenLayer1 = HiddenLayer(2,2)
sigmoidLayer = SigmoidActivation()
outputLayer = HiddenLayer(2,2)
softMaxLayer = SoftMaxActivation()
error = Error()


hiddenLayer1.forwardPass(inputs1) 
sigmoidLayer.forwardPass(hiddenLayer1.output)

outputLayer.forwardPass(sigmoidLayer.output)
softMaxLayer.forwardPass(outputLayer.output)
error.forwardPass(softMaxLayer.output, target)

error.displayActivity()







