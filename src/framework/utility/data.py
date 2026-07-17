import numpy as np


def shuffle(x, y):
    '''
        Randomly shuffles inputs and corresponding labels

        The same random ordering is applied to both x and y to preserve
        the relationship between samples and labels

        
        Parameters
        ----------
        x : np.ndarray -- Input data with samples along axis 0
        y : np.ndarray -- Labels corresponding to each sample

        
        Returns
        -------
        x : np.ndarray -- Shuffled input data
        y : np.ndarray -- Shuffled labels
    '''
    xSwap = np.zeros(shape=x.shape[1])

    if y[0].shape == (): ## Handles scalar labels
        ySwap = 0

    else: ## Handles vectors for labels
        ySwap = np.zeros(shape=y.shape[1])
        

    for i in range(x.shape[0]-2):
        choice = np.random.randint(i, x.shape[0]-1)
        xSwap = x[i].copy()
        x[i] = x[choice]
        x[choice] = xSwap

        ySwap = y[i].copy()
        y[i] = y[choice]
        y[choice]= ySwap
    
    return x, y




def dataSplit(x, y, trainFraction):
    '''
        Splits a dataset into training and validation subsets.

        The data is shuffled before splitting to ensure samples are
        randomly distributed between subsets.

        Parameters
        ----------
        x : np.ndarray -- Input data
        y : np.ndarray -- Labels corresponding to input data
        trainFraction : float -- Fraction of data assigned to the training set

        Returns
        -------
        xTrain : np.ndarray -- Training inputs
        yTrain : np.ndarray -- Training labels
        xTest : np.ndarray -- Testing inputs
        yTest : np.ndarray -- Testing labels
    '''
    trainNum = int(x.shape[0] * trainFraction)
    valNum = int(x.shape[0] - trainNum)

    shuffle(x, y)

    return  x[0:trainNum], y[0:trainNum], x[trainNum:trainNum+valNum], y[trainNum:trainNum+valNum]



class dataLoader:
    def __init__(self, x, y, batchSize):
        '''
            Creates mini-batches for training.

            The dataset is shuffled before batching. 
            
            If the number of samples is not divisible by batch size,
            the remaining samples are returned as an additional smaller batch

            Parameters
            ----------
            xin : np.ndarray -- Input dataset
            yin : np.ndarray -- Labels corresponding to the dataset
            self.batchSize : int -- Number of samples per batch

            Returns
            -------
            list[tuple] -- List containing (input batch, label batch) pairs
        '''
        self.x = x
        self.y = y
        self.batchSize = batchSize
        self.numbatches = x.shape[0] // self.batchSize

    def __iter__(self):    
        shuffle(self.x, self.y)

        # dataset = []
    
        for batch in range(self.numbatches):
            # dataset.append((x[batch*self.batchSize:(batch+1)*self.batchSize],y[batch*self.batchSize:(batch+1)*self.batchSize]))

            yield (self.x[batch*self.batchSize : (batch+1)*self.batchSize], self.y[batch*self.batchSize : (batch+1)*self.batchSize])

        # Include remaining samples when dataset size is not divisible by batch size
        # dataset.append((x[numbatches*self.batchSize: x.shape[0]],y[numbatches*self.batchSize:x.shape[0]])) 

        yield (self.x[self.numbatches*self.batchSize : self.x.shape[0]], self.y[self.numbatches*self.batchSize : self.x.shape[0]])
        
        # return dataset

