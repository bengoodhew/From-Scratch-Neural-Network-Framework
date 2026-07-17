import numpy as np


def accuracy(yTrue, yPred):
    '''
        Calculates classification accuracy

        Accuracy represents the proportion of predictions that exactly match the true labels

        Parameters
        ----------
        yTrue : np.ndarray -- Ground truth labels
        yPred : np.ndarray -- Model predictions

        Returns
        -------
        float -- Fraction of correctly classified samples
    '''
    counter = np.sum(yTrue == yPred)
    total = yTrue.shape[0]
    
    return counter/total

