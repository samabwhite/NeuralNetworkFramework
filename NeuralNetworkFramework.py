import numpy as np

class NeuralNetwork(object):
    def __init__(self, dataset, labels, epochs, alpha):
        self.dataset = dataset
        self.labels = labels
        self.alpha = alpha
        self.epochs = epochs
        self.layers = [None] # first layer value is None as it represents the input layer
        self.errorLayer = None

    def addHiddenLayer(self, inputSize, outputSize, layerNumber):   # add first layer or automatically checks compatibility and adds layer to end '
        newLayer = HiddenLayer(inputSize, outputSize) 
        if not self.checkLayerCompatibility(newLayer, layerNumber):
            return
        self.layers.insert(layerNumber, newLayer)

    def addActivation(self, layerNumber, type):
        self.layers[layerNumber].setActivation(type)

    def addError(self): # figure out a way to prevent hidden layers to be added after error layer
        self.errorLayer = Error()
        self.layers.append(self.errorLayer)

    def forwardPass(self):
        network = self.layers
        network[1].forwardPass(self.dataset)
        for index in range(2, len(network) - 1):
            network[index].forwardPass(network[index-1].output)
        network[-1].forwardPass(network[-2].output, self.labels)

    def backwardPass(self):
        pass

    def displayNetwork(self):
        for layer in self.layers[1:]:
            print(str(layer.output))

    def displayNetworkOutput(self):
        print("Error: " + str(self.errorLayer.totalError))




    def checkLayerCompatibility(self, newLayer, layerNumber):

        # if layer number is specified as 0, return error
        if layerNumber == 0:
            print("Layer number cannot be 0. The 0th index is reserved for the inputs of the neural network.")
            return False

        def checkFit(layer1, layer2):
            if layer1.weights.shape[1] == layer2.weights.shape[0]:
                return True
            else: 
                print("New layer with shape " + str(newLayer.weights.shape) +  " is NOT compatable with " + str(self.layers[-1].weights.shape))
                return False

        # if there are no layers, return true
        if len(self.layers) == 1:
            try:
                result = self.dataset.shape[1] == newLayer.weights.shape[0]
            except:
                result = self.dataset.shape[0] == newLayer.weights.shape[0]

            if result:
                return True
            else:
                print(str("Inputs with shape " + str(self.dataset.shape) + " does not fit the new layer shape " + str(newLayer.weights.shape)))
                return False
        # if layer is being placed in first index, check rear compatibiltiy only
        elif layerNumber == 1 and (self.layers[layerNumber] is not None):
            return checkFit(newLayer, self.layers[layerNumber])
        # if layer is being placed at end, check front compatibility only
        elif layerNumber == len(self.layers):
            return checkFit(self.layers[-1], newLayer)
        # if layer is being placed in the middle, check both sides compatibility
        else:
            if checkFit(self.layers[layerNumber-1], newLayer) and checkFit(newLayer, self.layers[layerNumber]):
                return True
            return False


        





class Layer(NeuralNetwork):
    def __init__(self):
        self.input = None
        self.output = None 
        self.nextLayer = None
        self.prevLayer = None

    def forwardPass(self):
        pass

    def backwardPass(self):
        pass

    def update(self):
        pass

        


class HiddenLayer(Layer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(inputSize, outputSize)
        self.bias = np.random.randn(outputSize)
        #self.nextLayer = nextLayer
        #self.prevLayer = prevLayer
        self.activationLayer = None
    
    def forwardPass(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias.T
        return self.output


    def backwardPass(self, errorDeriv, softMaxDeriv, sigmoidDeriv):
        def foil(dZ, input):
            dW = []
            for i in dZ:
                for k in input:
                    dW.append(i * k)
            return dW
        if self.nextLayer == None: 
            self.dZ = errorDeriv * softMaxDeriv
            self.dB = self.dZ
            self.dW = np.array(foil(self.dZ, self.prevLayer.output))
        else:   
            self.dZ = np.dot(self.nextLayer.dZ, self.nextLayer.weights) * sigmoidDeriv
            self.dB = self.dZ
            self.dW = np.array(foil(self.dZ, self.input))

    def update(self):
        self.weights += (-self.alpha * np.reshape(self.dW, (2,2)))
        self.bias += (-self.alpha * self.dB)


    def setActivation(self, type):
        if type == "Sigmoid":
            self.activationLayer = SigmoidActivation()
        elif type == "Softmax":
            self.activationLayer = SoftMaxActivation()
        else:
            print("Unknown Activation Type")
        


class SigmoidActivation(Layer):
    def __init__(self):
        pass

    def forwardPass(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backwardPass(self):
        self.derivative = self.output * (np.subtract(1, self.output))

    def update(self):
        pass




class SoftMaxActivation(Layer):
    def __init__(self):
        pass

    def forwardPass(self, input):
        self.input = input
        self.output = []
        for i in input:
            self.output.append(np.exp(i) / np.sum(np.exp(input)))
        return self.output

    def backwardPass(self):
        self.derivative = self.output * (np.subtract(1, self.output)) # softMaxOutput * (1 - softMaxOutput)

    def update(self):
        pass



class Error(Layer):
    def __init__(self):
        pass

    def forwardPass(self, networkOutput, labels):
        self.networkOutput = networkOutput
        self.labels = labels

        self.totalError = self.totalSquaredError(self.networkOutput, self.labels)
        self.totalErrorDerivative = self.squaredErrorDerivative(self.networkOutput, labels)

        self.output = self.totalError

    def backwardPass(self):
        self.derivative = np.subtract(self.networkOutput, self.labels) # output - labels 

    def totalSquaredError(self, predicted, labels):
        return np.sum((1/2) * np.power(np.subtract(labels, predicted), 2))

    def squaredErrorDerivative(self, predicted, labels):
        return np.subtract(predicted, labels)




#____________________________________________________________________________
# Sandbox Area for Testing Framework

inputs = np.array([0.1, 0.2, 0.5, 0.7, 1.0])
labels = np.array([0.01, 0.99])

nn = NeuralNetwork(inputs, labels, 1, 0.01)

nn.addHiddenLayer(5, 2, 1)
nn.addActivation(1, "Sigmoid")
nn.addHiddenLayer(2, 2, 2)
nn.addActivation(2, "Softmax")
nn.addError()

nn.forwardPass()







