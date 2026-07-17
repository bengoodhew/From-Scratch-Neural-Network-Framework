import numpy as np
import framework.functions.activation as activation
import framework.functions.derivatives as derivatives


class Layer:
    '''
        Template class for layers

        Child classes must define both forward and backward methods

        If a layer has weights call super().__init__(weightsIncluded = True)
        Define the getWeights, setWeights, getBias and setBias methods
    '''
    def __init__(self, weightsIncluded = False):
        self.weightsIncluded = weightsIncluded

    def forward(self, x, trainMode = None):
        """
            Performs the forward pass

            Parameters
            -----------
            x : np.ndarray -- input 
            trainMode : bool -- used as some operation need changing in inference

            Returns
            --------
            y : np.ndarray
        """
        raise NotImplementedError("Layer must implement a forward()")

    def backward(self, delta, lr):
        '''
            Performs the backward pass with delta update if required

            Parameters
            -----------
            delta : np.ndarray -- error signal (delta) from next layer
            lr : float -- learning rate
           
            Returns
            --------
            delta : np.ndarray -- updated error signal (delta) for previous layer
        '''
        raise NotImplementedError("Layer must implemet a backward()")
        

    def hasWeights(self):
        return self.weightsIncluded
    
    def save(self, file):
        np.save(file, self.getWeights())
        np.save(file, self.getBias())
    
    def load(self, file):
        self.setWeights(np.load(file))
        self.setBias(np.load(file))








class FullyConnected(Layer):
    '''
        Fully connected (dense) neural network layer

        Computes:
        y = x*w + b


        Where:
        x : Input features
        w : Learnable weight matrix
        b : Learnable bias vector

       
        Weight initialisation supports HE, GLOROT and LECUN each with both uniform and normal distribution options

        Biases are all initialised to 0


        During the forward pass the inputs are transformed into the output feature space using a linear transform

        During backward pass the error (delta) is received, used to:
         -  calculate gradients for weights and biases
         -  update weights and biases
         -  calculate delta for the previous layer and pass back to continue backpropagation
    '''

    def __init__(self, inSize, outSize, initMethod = "HE", initDist = "uniform"):
        '''
            Initialise weights and biases

            Parameters
            ----------
            inSize : int -- used to create weight matrix and initialisation
            outSize : int -- used to create weight matrix and initialisation
            initMethod : str(option - HE, GLOROT, LECUN) -- option for initialisation type
            initDist : str(option - UNIFORM, NORMAL) -- option for initialisation distribution

            Creates
            --------
            self.weights : np.ndarray -- shape(inSize, outSize)
            self.bias : np.ndarray -- shape(1, outSize)
        '''
        super().__init__(weightsIncluded = True)
        self.inSize = inSize
        self.outSize = outSize


        ## initialisation type
        if initMethod.upper() == "HE":
            var = 2/inSize
            interval = np.sqrt(6/inSize)
        elif initMethod.upper() == "GLOROT":
            var = 2/(outSize+inSize)
            interval = np.sqrt(6/(inSize+outSize))
        elif initMethod.upper() == "LECUN":
            var = 1/inSize
            interval = np.sqrt(3/inSize)

        ## initialisation distribution
        if initDist.upper() == "NORMAL":
            self.weights = np.random.normal(0, np.sqrt(var), size = (inSize, outSize)) 
        elif initDist.upper() == "UNIFORM":
            self.weights = np.random.uniform(-1*interval, interval, size = (inSize, outSize)) 

        ## Biases initialisation
        self.bias = np.zeros(shape = (1, outSize))


        ## Stores input in forward pass used to calculate weight gradient in backward
        self.input = None



    def forward(self, x, trainMode = None):
        """
            Performs the forward pass

            Parameters
            -----------
            x : np.ndarray -- input of shape (batchSize, inSize)
            trainMode : bool -- unused - kept for consistency with other layers

            Returns
            --------
            y : np.ndarray -- output of shape (batchSize, outSize)
        """
        self.input = x  ## Used to calculate weight gradients

        y = np.dot(self.input, self.weights) + self.bias
        
        return y



    def backward(self, delta, lr):
        '''
            Performs the backward pass with weight update

            Parameters
            -----------
            delta : np.ndarray -- error signal (delta) from next layer
            lr : float -- learning rate for this update
           
            Returns
            --------
            delta : np.ndarray -- updated error signal (delta) for previous layer
        '''
        weightGrad = np.dot(self.input.T, delta)
        biasGrad = np.sum(delta, axis=0, keepdims=True)

        
        self.weights -= lr * weightGrad
        self.bias -= lr * biasGrad

        delta = np.dot(delta, self.weights.T)


        return delta


    def getWeights(self):
        return self.weights
    
    def setWeights(self, weights):
        self.weights = weights


    def getBias(self):
        return self.bias
    
    def setBias(self, bias):
        self.bias = bias









class ActivationLayer(Layer):
    '''
        Activation function layer is a wrapper to implement activation function within a model
       
        Retrieve both activation and derivative function by name (e.g ReLU)
       
        Forward applies the activation

        Backward updates the delta with the derivative
    '''
    def __init__(self, function):
        '''
            Retrieves both the activation function and the activation function derivative

            Parameters
            ----------
            activation function : str
                supported options:
                    - SIGMOID
                    - RELU
                    - LEAKYRELU
                    - TANH
                    - SOFTMAXWITHCROSSENTROPY
        '''
        super().__init__()
        self.input = None ## Stores input in forward pass used to calculate gradient in backward
        self.activation, self.activationDerivative = self.findActivation(function)
               

    def findActivation(self, activationChoice):
        '''
            Maps activation name to activation function and the activation function derivative
        '''

        activationChoice = activationChoice.upper()
        self.activationChoice = activationChoice

        if(activationChoice == "SIGMOID"):
            return activation.sigmoid, derivatives.derivedSigmoid
        
        elif(activationChoice == "RELU"):
            return activation.ReLU, derivatives.derivedReLU
        
        elif(activationChoice == "LEAKYRELU"):
            return activation.leakyReLU, derivatives.derivedLeakReLU
        
        elif(activationChoice == "TANH"):
            return activation.tanh, derivatives.derivedTanh
        
        elif(activationChoice == "SOFTMAXWITHCROSSENTROPY"):
            return activation.softmax, derivatives.derivedSoftmaxWithCrossEntropy
        
        else:
            raise Exception("Not A Valid Activation")
        
        
    def forward(self, x, trainMode = None):
        '''
            Applies activation

            Parameters
            ----------
            x : np.ndarray -- input of shape (batchSize, previous output size)
            trainMode : bool -- unused kept for consistency

            Returns
            --------
            np.ndarray -- output of shape (batchSize, previous output size)
        '''
        self.input = x.copy()  ## Used to calculate gradients and update delta
        temp = self.activation(x)
        
        return temp
    
    def backward(self, delta, lr):
        '''
            Updates delta with derivative

            Parameters
            ----------
            delta : np.ndarray -- error signal from the following layer
            lr : float -- unused kept for consistency

            Returns
            --------
            np.ndarray -- updated delta for the previous layer
        '''
        return self.activationDerivative(self.input, delta)









class DropoutLayer(Layer):
    '''
        DropoutLayer applies a mask to forward inputs and backward gradients

        The mask is calculated based on Bernoulli distribution with probability 1-p (where p is dropout probability)

        The mask is not applied if in inference mode

        
        Numpys binomial distribution is used with n (trials) = 1, which is equivalent to the Bernoulli distribution
    '''
    def __init__(self, p):
        '''
            Initialise state variables

            Parameters
            ----------
            size : int -- size of the mask
            p : float -- probability used to create the mask
        '''
        super().__init__()

        self.p = p

        self.mask = None

    def generateMask(self, size):
        '''
            Uses the keep probability (1 - self.p) and size (self.size) to generate the mask

            Scaled mask to maintain the magnitude of the outputs in forward pass and gradients in backward pass
        '''
        mask = np.random.binomial(1, (1-self.p), size)
        
        self.mask = mask / (1-self.p)

    def forward(self, x, trainMode):
        '''
            A mask is generated and applied only if trainMode is on

            If train mode is not applied layer just forwards inputs

            Parameters
            ----------
            x : np.ndarray -- input to layer
            trainMode : bool -- determines whether or not to apply the mask

            Returns
            --------
            np.ndarray -- either x with mask applied or just x depending on trainMode
        '''
        if trainMode:
            self.generateMask(x.shape)

            return x * self.mask
        return x

    def backward(self, delta, lr = None):
        '''
            Applies the stored mask to the gradients

            The same mask is used as in forward pass so gradients are applied only to active neurons

            Parameters
            ----------
            delta : np.ndarray -- error signal from the following layer
            lr : float -- unused kept for consistency

            Returns
            --------
            np.ndarray -- updated(scaled) delta for the previous layer
        '''
        return delta * self.mask










class BatchNormLayer(Layer):
    '''
        Batch normalisation layer

        Normalises inputs using batch mean and variance during training and calculated global mean and variance in inference.

        Normalised inputs are then scaled by the factor gamma and shifted by beta.
    '''
    def __init__(self, size, e = 0.0001):
        '''
            Initialise BatchNorm state variables and learnable parameters

            Parameters
            ----------
            size : int -- number of features to normalise
            e : float -- small value added to variance for numerical stability

            Creates
            --------
            globalMean : np.ndarray -- running mean used during inference
            globalVar : np.ndarray -- running variance used during inference
            means : np.ndarray -- storage for the batch mean
            var : np.ndarray -- storage for the batch variance
            gamma : np.ndarray -- learnable scale parameter
            beta : np.ndarray -- learnable shift parameter
            xHat : np.ndarray -- storage for normalised inputs from forward to be used in backward pass
        '''
        super().__init__(weightsIncluded = True)

        self.globalMean = None
        self.globalVar = None

        self.means = np.zeros(size)
        self.var = np.zeros(size)
        
        self.gamma = np.ones(size)
        self.beta = np.zeros(size)

        self.e = e

        self.xHat = None

        

    def updateGlobalMeanVar(self):
        '''
            Updates running mean and variance used during inference

            The first batch initialises the global statistics.
            Following batches update the stored statistics using the
            current batch statistics.
        '''
        if self.globalMean is None:
            self.globalMean = self.means
            self.globalVar = self.var
            return
        
        self.globalMean = (self.means + self.globalMean)/2
        self.globalVar = (self.var + self.globalVar)/2
        



    def findMeanAndVar(self, x):
        '''
            Calculates mean and variance from current batch

            Parameters
            ----------
            x : np.ndarray -- input batch

            Stores
            ------
            self.means : np.ndarray -- current batch mean
            self.var : np.ndarray -- current batch variance
        '''

        self.means = np.mean(x, axis = 0)
        self.var =  np.var(x, axis=0)

        ## Updates running statistics for inference
        self.updateGlobalMeanVar()
        
    
    def testNormalise(self, x):
        '''
            Normalises using stored global statistics

            Used during inference when batch statistics are not available
        '''
        return ((x - self.globalMean))/np.sqrt(self.globalVar + self.e)


    def normalise(self, x):
        '''
            Normalises using current batch statistics

            Used during training
        '''
        return ((x - self.means))/np.sqrt(self.var + self.e)


    def transform(self, x):
        '''
            Applies learnable scales and shift using gamma and beta
        '''
        return x*self.gamma - self.beta


    def forward(self, x, trainMode):
        '''
            Training:
                Calculates batch mean and variance and normalises using them

            Inference:
                Uses stored global mean and variance
            
                
            Normalised inputs are then scaled and shifted

            
            Parameters
            ----------
            x : np.ndarray -- input activations
            trainMode : bool -- determines training or inference

            Returns
            --------
            np.ndarray -- normalised and transformed activations
        '''
        if trainMode:
            ## Calculate and store current mean and variance of batch
            self.findMeanAndVar(x)
            ## Normalise using current mean and variance of batch
            self.xHat = self.normalise(x)

        else:
            ## Normalise using stored global mean and variance
            self.xHat = self.testNormalise(x)

        ## Apply scale and shift
        out = self.transform(self.xHat)

        return out
    


    def backward(self, delta, lr):
        '''
            Uses delta to calculate the gradients of beta and gamma

            Delta for the previous layer is calculated and delta is updated

            Updates learnable parameters and returns gradient for the previous layer.

            Parameters
            ----------
            delta : np.ndarray -- error signal from following layer
            lr : float -- learning rate

            Returns
            --------
            np.ndarray -- updated delta for previous layer
        '''
        n = delta.shape[0] ## batchSize

        ## Gradients for learnable parameters
        dbeta = np.sum(delta, axis = 0)
        dgamma = np.sum(delta * self.xHat, axis = 0)


        ## Gradient through scale parameter
        deltaHat = delta * self.gamma

        ## Inverse standard deviation
        invStd = 1 / np.sqrt(self.var + self.e) #x-mu


        ## Calculate input gradient
        deltaX = ((invStd / n)
                  * (n * deltaHat - np.sum(deltaHat, axis=0)
                     - self.xHat * np.sum(deltaHat * self.xHat, axis=0)))

        ## Apply update
        self.beta -= lr * dbeta
        self.gamma -= lr * dgamma

        return deltaX
    




    def getWeights(self):
        return np.array([self.gamma, self.beta])
    
    def setWeights(self, weights):
        self.gamma = weights[0]
        self.beta = weights[1]

    def getBias(self):
        return np.array([self.globalMean, self.globalVar])
    
    def setBias(self, bias):
        self.globalMean = bias[0]
        self.globalVar = bias[1]





