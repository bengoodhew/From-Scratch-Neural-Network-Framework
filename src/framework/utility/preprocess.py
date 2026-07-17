import numpy as np


class Normaliser:
    '''
        Normaliser calculates the mean and variance of each feature during fitting,
        then uses these values to normalise new data

        Applies:
            xNormalised = (x - mean) / sqrt(variance + epsilon)
    '''
    def __init__(self, e = 0.00001):
        ''' 
            Initialises epsilon
        
            Parameters
            ----------
            e : float -- Small value added to variance for numerical stability
        '''
        self.mean = None
        self.var = None
        self.e = e

    def fit(self, x):
        '''
            Calculates and stores feature mean and variance

            Parameters
            ----------
            x : np.ndarray -- Data to fit mean and variance
        '''
        self.mean = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)



    def fitTransform(self, x):
        '''
            Fits normalisation parameters and transforms data


            Parameters
            ----------
            x : np.ndarray -- Data to normalise

            Returns
            -------
            np.ndarray -- Normalised data
        '''
        self.fit(x)
        return self.transform(x)



    def transform(self, x):
        '''
            Applies stored normalisation parameters

            Parameters
            ----------
            x : np.ndarray -- Data to normalise

            Returns
            -------
            np.ndarray -- Normalised data
        '''
        return (x-self.mean)/np.sqrt(self.var + self.e)
    
    
def labelSmoother(y, e = 0.1):
    '''
        Applies label smoothing to one-hot encoded labels

        Label smoothing replaces hard targets with softer probabilities

        Parameters
        ----------
        y : np.ndarray -- One-hot encoded labels
        e : float -- Smoothing factor

        Returns
        -------
        np.ndarray -- Smoothed labels
    '''
    return (1 - e) * y + (e/y.shape[1])