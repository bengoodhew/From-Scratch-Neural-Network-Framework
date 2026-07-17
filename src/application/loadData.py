import numpy as np
import pickle
from pathlib import Path

'''
    Dataset loading utilities.

    Loads CIFAR-10 pickle files, selects classes, and converts data into NumPy arrays
'''

def toNumpy(data):
    '''
        Combines a list of batches into a single NumPy array
        Handles different shapes
    '''
    samples = []

    for x in data:
        samples.append(x.shape[0])

    try:
        thisShape = (np.sum(samples), data[0][0].shape[0])
    except IndexError:
        thisShape = (np.sum(samples),)

    outData = np.zeros(shape=thisShape)
    sampTotal = 0
    for dat, samp in zip(data, samples):
        outData[sampTotal : sampTotal + samp] = dat
        sampTotal += samp

    return outData
    

def unpickle(file):
    '''Loads a pickle dataset file'''
    with open(file, 'rb') as fp:
        dict = pickle.load(fp, encoding='bytes')
    return dict


def loadBatch(file, labelIds, x, y):
    '''Loads one dataset batch and appends selected samples with one-hot encoded labels'''
    print("Loading:", file)

    rawData = unpickle(file)
    fullData = rawData[b'data']
    dataLabels = np.array(rawData[b'labels'])

    for idx, indexes in enumerate(labelIds):
        sampleIdx = np.where(dataLabels == indexes)
        
        x.append(fullData[sampleIdx])

        ### Creates one-hot encoded vector
        tempYs = np.zeros(len(labelIds))
        tempYs[idx] = 1

        y.append(np.full((sampleIdx[0].shape[0], 5), tempYs))




def loadInitial(folder, labelsWanted):
    '''
        Loads dataset and returns training/testing arrays

        Selects requested classes, loads batches, converts data format, and prepares labels
    '''
    ## Root will be the main project folder -- assumes loadData.py in src/application/ and root is two levels above
    root = Path(__file__).resolve().parents[2]
    folder = root / "dataset" / folder
    
    batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

    ### Extracts numerical value for labelsWanted
    labels = unpickle(folder / "batches.meta")[b'label_names']
    labelIds = []
    for i in range(len(labels)):
        for label in labelsWanted:
            if label.encode() == labels[i]:
                labelIds.append(i)
    labelIds = np.array(labelIds)

    ### Loads every training batch
    xTrain = []
    yTrain = []
    for batch in batches:
        loadBatch(folder / batch, labelIds, xTrain, yTrain)
        
    xTrain = toNumpy(xTrain)
    yTrain = toNumpy(yTrain)
    
    
    ### cifar-10 has sporate test batch
    xTest = []
    yTest = []
    loadBatch(folder / "test_batch", labelIds, xTest, yTest)
    xTest = toNumpy(xTest)
    yTest = toNumpy(yTest)


    return xTrain, yTrain, xTest, yTest
