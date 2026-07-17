import numpy as np


def step(x):
    '''
        Binary step activation function

        Returns 1 for non-negative inputs and 0 otherwise

        Parameters
        ----------
        x : float -- Input value

        Returns
        -------
        int -- Binary activation output
    '''
    if x>= 0:
        return 1
    return 0



def leakyReLU(x, leak = 0.01):
    '''
        Leaky rectified linear activation function

        Allows a small gradient for negative inputs to reduce the dying ReLU problem

        Formula:
            f(x) = x if x > 0
            f(x) = leak*x if x <= 0

        Parameters
        ----------
        x : np.ndarray -- Input values
        leak : float -- Gradient factor applied to negative values

        Returns
        -------
        np.ndarray -- Activated values
    '''
    mask = np.where(x<=0, leak, 1)
    return mask*x


def ReLU(x):
    '''
        Rectified linear activation function

        Sets negative values to zero while keeping positive values unchanged

        Formula:
            f(x) = max(0, x)

        Parameters
        ----------
        x : np.ndarray -- Input values

        Returns
        -------
        np.ndarray -- Activated values
    '''
    return np.maximum(0, x)



def sigmoid(x):
    '''
        Sigmoid activation function

        Maps inputs to the range (0, 1)

        Formula:
            f(x) = 1 / (1 + exp(-x))

        Parameters
        ----------
        x : np.ndarray -- Input values

        Returns
        -------
        np.ndarray -- Activated values
    '''
    return 1 / (1 + np.exp(-x))



def tanh(x):
    '''
        Hyperbolic tangent activation

        Maps inputs to the range (-1, 1)

        Parameters
        ----------
        x : np.ndarray -- Input values

        Returns
        -------
        np.ndarray -- Activated values
    '''
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x):
    '''
        Softmax activation function

        Converts logits into a probability distribution where each row sums to 1

        A max value subtraction is applied before exponentiationto improve numerical stability and reduce overflow risk

        Parameters
        ----------
        x : np.ndarray -- Model logits with shape (batch_size, number_of_classes)

        Returns
        -------
        np.ndarray -- Class probabilities
    '''
    return(np.exp(x-np.max(x, axis=1,keepdims=True)))/ np.sum(np.exp(x-np.max(x, axis=1,keepdims=True)), axis=1, keepdims=True)