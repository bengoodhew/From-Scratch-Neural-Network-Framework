import framework.functions.lossFuncs as lossFuncs


class Model:
    '''
        Template class for Models

        To create a model:
            -  Extend this class
            -  Call super().__init__(lossName)
            -  Define self.model as an ordered list of layers


        This class also works as the wrapper for the loss functions 
        To call loss run Model.loss(true, pred) -- will calculate loss and return loss

        To make a prediction run Model.forward()
        To update parameters run Model.backward()

        To switch to train mode call Model.trainMode()
        To switch to inference mode call Model.inferenceMode()



    '''
    def __init__(self, loss):
        '''
            Retrieves loss and the derivative for that loss

            Initialises the train state for train vs inference modes and last batch to scale loss derivative

            Parameter
            ---------
            loss : str(option - MSE, CROSSENTROPYLOSSWITHSOFTMAX) -- option for loss function
        '''
        self.loss, self.lossDerivative = self.findLoss(loss)
        self.train = True


    def findLoss(self, loss):
        '''
            Maps loss name to loss function and the loss function derivative
        '''
        loss = loss.upper()

        if(loss == "MSE"):
            return lossFuncs.MSE_loss, lossFuncs.MSE_loss_derivative
        
        if(loss == "CROSSENTROPYLOSSWITHSOFTMAX"):
            return lossFuncs.crossEntropyLoss, lossFuncs.crossEntropyLossWithSoftmaxAct
        
        else:
            raise Exception("Not A Valid Loss")
        

    def trainMode(self):
        '''
            Switch model to train mode uses training behaviour for batchNorm and dropout
        '''
        self.train = True
        
    def inferenceMode(self):
        '''
            Switch model to inference mode uses inference behaviour for batchNorm and dropout
        '''
        self.train = False


    def forward(self, x):
        '''
            Applies each layer sequentially passing output to next layer as input

            Parameters
            -----------
            x : np.ndarray -- inputs for the model (must be in shape (batchSize, firstLayerInputSize))

            Returns
            -------
            np.ndarray -- output of the model after all layers (will be in shape (batchSize, lastLayerOutputSize))
        '''
        next = x
        for layer in self.model:
            next = layer.forward(next, self.train)
        return next
    


    def backward(self, yTrue, yPred, lr):
        '''
            First calculates the error (delta) from loss derivate then applies each layer sequentially in reverse passing delta back

            Parameters
            -----------
            yTrue : np.ndarray -- the true label for the model
            yPred : np.ndarray -- the predicted value from the model
            lr : float -- learning rate used to update the model
        '''
        delta = self.lossDerivative(yTrue, yPred)
        delta = delta / delta.shape[0]
        
        for layer in reversed(self.model):
            delta = layer.backward(delta, lr)

        

    def saveModel(self, fileName):
        '''
            Will save model weights

            Parameter
            ----------
            fileName : str -- the name of the save file
        '''
        with open(fileName, 'wb') as f:
            for layer in self.model:
                if layer.hasWeights() == 1:
                    layer.save(f)


    def loadModel(self, fileName):
        '''
            Will load model weights

            Must be from the a model with same architecture

            Parameter
            ----------
            fileName : str -- the name of the file to load from
        '''
        with open(fileName, 'rb') as f:
            for layer in self.model:
                if layer.hasWeights() == 1:
                    layer.load(f)

