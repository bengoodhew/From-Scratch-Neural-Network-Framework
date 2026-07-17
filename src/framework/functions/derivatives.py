import numpy as np
import framework.functions.activation as activation


'''
File contains derivatives of activation functions.

Every function receives:
    x -- inputs stored in forward pass
    delta -- error signal from following layer

The derivative with respect to x is calculated and used to update delta
The updated delta is returned to continue backpropagation
'''

def derivedSigmoid(x, delta):
    '''
        Calculates sigmoid activation gradient during backpropagation

        Formula:
            sigmoid'(x) = sigmoid(x)(1 - sigmoid(x))

        Parameters
        ----------
        x : np.ndarray -- Input values from the forward pass
        delta : np.ndarray -- Error gradient received from the following layer

        Returns
        -------
        np.ndarray -- Updated gradient passed to the previous layer
    '''
    s = activation.sigmoid(x)
    return (s * (1 - s)) * delta



def derivedReLU(x, delta):
    '''
        Calculates ReLU activation gradient during backpropagation

        The gradient is 1 for positive inputs and 0 for negative inputs

        Parameters
        ----------
        x : np.ndarray -- Input values from the forward pass
        delta : np.ndarray -- Error gradient received from the following layer

        Returns
        -------
        np.ndarray -- Updated gradient passed to the previous layer
    '''
    return np.where(x >= 0, 1, 0) * delta



def derivedLeakReLU(x, delta, leak = 0.01):
    '''
        Calculates Leaky ReLU activation gradient

        Unlike standard ReLU, negative inputs retain a small gradient

        Parameters
        ----------
        x : np.ndarray -- Input values from the forward pass
        delta : np.ndarray -- Error gradient received from the following layer
        leak : float -- Gradient value for negative inputs

        Returns
        -------
        np.ndarray -- Updated gradient passed to the previous layer
    '''
    return np.where(x > 0, 1, leak) * delta



def derivedTanh(x, delta):
    '''
        Calculates tanh activation gradient during backpropagation

        Formula:
            tanh'(x) = 1 - tanh(x)^2

        Parameters
        ----------
        x : np.ndarray -- Input values from the forward pass
        delta : np.ndarray -- Error gradient received from the following layer

        Returns
        -------
        np.ndarray -- Updated gradient passed to the previous layer
    '''
    y = activation.tanh(x)
    return (1 - y**2) * delta

def derivedSoftmaxWithCrossEntropy(x, delta):
    '''
        Gradient for softmax activation combined with cross entropy loss
        The derivative of softmax and cross entropy simplifies to:
            delta = prediction - target

        Therefore the gradient calculation is already performed inside the cross entropy loss function and is returned unchanged here

        Parameters
        ----------
        x : np.ndarray -- Input logits unused kept for consistency
        delta : np.ndarray -- Gradient calculated by the loss function

        Returns
        -------
        np.ndarray -- Unmodified gradient
    '''
    return delta

