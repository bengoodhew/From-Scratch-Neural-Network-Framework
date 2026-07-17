import numpy as np


def MSE_loss(actual, pred):
    '''
        Calculates Mean Squared Error loss

        Measures the average squared difference between predictions and target values

        Formula:
            MSE = mean((actual - prediction)^2)

        Parameters
        ----------
        actual : np.ndarray -- Target values
        pred : np.ndarray -- Model predictions

        Returns
        -------
        float -- Mean squared error value
    '''
    return np.mean((actual - pred) ** 2)


def MSE_loss_derivative(actual, pred):
    '''
        Calculates the gradient of Mean Squared Error loss

        Formula:
            dMSE/dx  = prediction - actual

        Parameters
        ----------
        actual : np.ndarray -- Target values
        pred : np.ndarray -- Model predictions

        Returns
        -------
        np.ndarray -- Error signal used to start backpropagation
    '''
    return pred - actual



def crossEntropyLoss(actual, pred):
    '''
        Calculates numerically stable cross entropy loss

        Formula:
            Loss = -mean(actual * log(prediction))

        Parameters
        ----------
        actual : np.ndarray -- One-hot encoded target labels
        pred : np.ndarray -- Predicted class probabilities
        e : float -- Numerical stability constant

        Returns
        -------
        float -- Cross entropy loss value
    '''
    maxPred = np.max(pred, axis=1, keepdims=True)
    logSumExp = maxPred + np.log(np.sum(np.exp(pred - maxPred), axis=1, keepdims=True))
    logSoftmax = pred - logSumExp
    return -np.mean(np.sum(actual * logSoftmax, axis=1))

def crossEntropyLossWithSoftmaxAct(actual, pred):
    '''
        Calculates gradient for softmax activation combined with cross entropy loss

        The derivative simplifies to:
            delta = prediction - actual

        
        Parameters
        ----------
        actual : np.ndarray -- One-hot encoded target labels
        pred : np.ndarray -- Model output before or after softmax depending on the implementation

        Returns
        -------
        np.ndarray -- Error signal used during backpropagation
    '''
    return pred - actual

