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
        pass # inputs from the previous

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
        pass # netOutput * (1 - netOutput)

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
        return np.sum((1/2) * np.power(np.subtract(labels, predicted), 2))

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

def foil(dZ, input):
    dW = []
    for i in dZ:
        for k in input:
            dW.append(i * k)
    return dW

def backwardPropagation(inputs, labels, hiddenLayer, sigmoidLayer, outputLayer, softMaxLayer, errorLayer):
    errorDeriv = np.subtract(outputLayer.output, labels) # output - labels 
    softMaxDeriv = softMaxLayer.output * (np.subtract(1, softMaxLayer.output)) # softMaxOutput * (1 - softMaxOutput)
    sigmoidDeriv = sigmoidLayer.output * (np.subtract(1, sigmoidLayer.output))
    
    dZ2 = errorDeriv * softMaxDeriv
    dB2 = dZ2 
    dW2 = np.array(foil(dZ2, hiddenLayer.output))

    dZ1 = np.dot(dZ2, outputLayer.weights) * sigmoidDeriv
    dB1 = dZ1
    dW1 = np.array(foil(dZ1, inputs))

    return [dW1, dB1, dW2, dB2]


def updateValues(hiddenLayer, outputLayer, dW1, dB1, dW2, dB2):
    hiddenLayer.weights += (-alpha * np.reshape(dW1, (2,2)))
    hiddenLayer.bias += (-alpha * dB1)
    outputLayer.weights += (-alpha * np.reshape(dW1, (2,2)))
    outputLayer.bias += (-alpha * dB2)
    
    
    


dW1, dB1, dW2, dB2 = backwardPropagation(inputs1, target, hiddenLayer1, sigmoidLayer, outputLayer, softMaxLayer, error)

print("Hidden Layer Weights: ")
print(str(hiddenLayer1.weights) + "\n")
print("Hidden Layer Biases: ")
print(str(hiddenLayer1.bias) + "\n")

print("Output Layer Weights: ")
print(str(outputLayer.weights) + "\n")
print("Output Layer Biases: ")
print(str(outputLayer.bias) + "\n")


updateValues(hiddenLayer1, outputLayer, dW1, dB1, dW2, dB2)
    





