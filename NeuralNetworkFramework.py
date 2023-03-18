import numpy as np

class NeuralNetwork(object):
    def __init__(self, dataset, labels, epochs, alpha):
        self.dataset = dataset #[0:epochs]
        self.labels = labels
        self.alpha = alpha
        self.epochs = epochs
        self.layers = [None] # first layer value is None as it represents the input layer
        self.errorLayer = None
        self.complete = False


    def addHiddenLayer(self, inputSize, outputSize, layerNumber):   # add first layer or automatically checks compatibility and adds layer to end '
        if layerNumber >= len(self.layers) and self.complete:
            print("Can't add dense layers after error.")
            return
        newLayer = HiddenLayer(inputSize, outputSize) 
        if not self.checkLayerCompatibility(newLayer, layerNumber):
            return
        self.layers.insert(layerNumber, newLayer)

    def addActivation(self, layerNumber, type):
        self.layers[layerNumber].setActivation(type)

    def addError(self, type): # figure out a way to prevent hidden layers to be added after error layer
        self.errorLayer = Error(type)
        self.layers.append(self.errorLayer)
        self.complete = True

    def removeLayer(self, layerNumber):
        if layerNumber == 0:
            print("Can't remove input layer from network.")
            return
        del self.complete[layerNumber]

    def forwardPass(self, data, label):
        layer = self.layers
        layer[1].forwardPass(data)
        layer[1].activationLayer.forwardPass(layer[1].output)
        if len(layer) > 3:
            for index in range(2, len(layer) - 1):
                layer[index].forwardPass(layer[index-1].activationLayer.output)
                layer[index].activationLayer.forwardPass(layer[index].output)
        self.errorLayer.forwardPass(layer[-2].activationLayer.output, label)

    def backwardPass(self, label):
        layer = self.layers
        # loop through each layer backward passing
        for index in range(2, len(self.layers)):
            layer[-index].backwardPass(self.layers, self.errorLayer.networkOutput, label)
        # update parameters
        

    def updateWeights(self):
        layer = self.layers
        for index in range(2, len(self.layers)):
            layer[-index].update(self.alpha)

    def displayNetworkOutput(self):
        print(self.errorLayer.networkOutput)

    def displayNetworkError(self):
        print("Error: " + str(self.errorLayer.error))


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
        # if layer is being placed in an empty section of the array
        elif layerNumber >= len(self.layers):
            print("WARNING: Layer was added in empty section of network array!")
            return True
        # if layer is being placed in the middle, check both sides compatibility
        else:
            if checkFit(self.layers[layerNumber-1], newLayer) and checkFit(newLayer, self.layers[layerNumber]):
                return True
            return False

    def train(self):
        if not self.complete:
            print("Error Layer Missing: Network not complete")


        for data, label in zip(self.dataset, self.labels):
            self.forwardPass(data, label)
            self.displayNetworkError()
            self.backwardPass(label)
            self.updateWeights()
            

        
            
            

            
            
         
            
            
        
        





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
        self.output = np.dot(input, self.weights) + self.bias.T
        return self.output


    def backwardPass(self, network, networkOutput, label):
        layerIndex = network.index(self)
        # need to add sum and 1/m to average out derivatives for batchsize
        if layerIndex == len(network)-2: 
            errorDeriv = network[-1].backwardPass(networkOutput, label)
            self.dZ = network[layerIndex].activationLayer.backwardPass(errorDeriv) 
            self.dB = self.dZ # * 1
            self.dW = []
            for i in self.input:
                self.dW.append(i * self.dZ)
            self.dW = np.array(self.dW) 
        else:
            self.dZ = []
            for dZ, weights in zip(network[layerIndex+1].dZ, network[layerIndex].weights):
                self.dZ.append(np.sum(dZ * weights))
            self.dZ *= network[layerIndex].activationLayer.backwardPass()
            self.dB = self.dZ # * 1
            self.dW = []
            for i in self.input:
                self.dW.append(i * self.dZ)
            self.dW = np.array(self.dW)
            
            


        '''
        def foil(dZ, input):
            dW = []
            for i in dZ:
                for k in input:
                    dW.append(i * k)
            return dW

        # if this layer is the output layer, 
        
        if network[-2] == self:
            self.dZ = errorDerivative * self.activationLayer.backwardPass()
            self.dB = self.dZ
            self.dW = np.array(foil(self.dZ, network[-3].output))
        else:
            layerIndex = network.index(self)
            self.dZ = np.dot(network[layerIndex+1].dZ, network[layerIndex+1].weights.T) * self.activationLayer.backwardPass() # second dot value rotated
            self.dB = self.dZ
            self.dW = np.array(foil(self.dZ, self.input))
        '''




    def update(self, alpha):
        self.weights += -(alpha * np.reshape(self.dW, (self.weights.shape)))
        self.bias += -(alpha * self.dB)


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
        #self.derivative = (np.exp(-self.output))/((np.exp(-self.output)+1)**2)
        self.derivative = self.output * (np.subtract(1, self.output))
        return self.derivative  

    def update(self):
        pass




class SoftMaxActivation(Layer):
    def __init__(self):
        pass

    def forwardPass(self, input):
        self.input = input
        self.output = []
        # TO DO! MAKE AN IF ELSE DEPENDING ON SHAPE OF INPUTS - BATCHES WILL BE USED
        self.expValues = np.exp(self.input - np.max(self.input)) # axis=1, keepdims=True 
        self.output = self.expValues / np.sum(self.expValues) # axis=1, keepdims=True
        '''
        for i in input:
            self.output.append(np.exp(i) / np.sum(np.exp(input)))
            '''
        return self.output

    def backwardPass(self, errorDeriv):
        n = np.size(self.output)
        tmp = np.reshape(np.tile(self.output, n), (10,10))
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), errorDeriv)

    def update(self):
        pass



class Error(Layer):
    def __init__(self, type):
        self.errorType = type
        self.total = 1
        self.correct = 1


    def forwardPass(self, networkOutput, labels):
        self.networkOutput = networkOutput
        self.labels = labels

        if self.errorType == "Categorical Cross Entropy":
            self.error = self.categoricalCrossEntropy(self.networkOutput, self.labels)
        elif self.errorType == "Mean Squared Error":
            self.error = self.totalSquaredError(self.networkOutput, self.labels)
        else:
            print("Error Type: " + str(self.errorType + " is unknown. Try again."))
        #self.meanError = self.totalSquaredError(self.networkOutput, self.labels)
        #self.totalErrorDerivative = self.squaredErrorDerivative(self.networkOutput, labels)
        if np.argmax(networkOutput) == np.argmax(labels):
            self.correct += 1
        self.total += 1
        print(self.correct / self.total)
        print(networkOutput)
        self.output = self.error

    def backwardPass(self, predicted, labels):
        if self.errorType == "Categorical Cross Entropy":
            return self.derivCCE(predicted, labels)
        elif self.errorType == "Mean Squared Error":
            #return squaredErrorDerivative(predicted, labels)
            pass
        else:
            print("Error Type: " + str(self.errorType + " is unknown. Try again."))

    def totalSquaredError(self, predicted, labels):
        print("Target Output: " + str(labels))
        print("System Output: " + str(predicted))
        # needs to divide by number of predictions, in this case it's 1 
        return (1/len(predicted))*np.sum(np.power(np.subtract(labels, predicted), 2))

    def squaredErrorDerivative(self, predicted, labels):
        pass

    
    def categoricalCrossEntropy(self, predicted, labels):
        return -np.sum(labels * np.log(predicted + (10**-20)))


    def derivCCE(self, predicted, labels):
        return -labels/(predicted + (10**-20))












