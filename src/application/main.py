import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import framework.utility.data as data
import framework.utility.preprocess as prePro

import loadData
import utility
from trainer import train, test
from models.baseLineModel import baseLineModel
from models.deepModel import deepModel


'''
    Experiment entry point

    loads configuration, prepares the dataset, creates the model and runs training or testing
'''


def main():
    ### Read command line arguments ###
    try:
        fileName = sys.argv[1]
    except IndexError:
        print("Must select model either baseline, deep or custom\n")
        print("Must select train/test mode -- format:  main.py baseline test\n ")
        print("If select custom must give file -- format:  main.py custom config.yaml\n ")
        exit()
    

    try:
        secondOption = sys.argv[2]
    except IndexError:
        print("Must select model option train or test")
        exit()

    
    if fileName.upper() == "BASELINE":
        configOption = ("configs/baseline.yaml", "configs/baselineTest.yaml")

    elif fileName.upper() == "DEEP":
        configOption = ("configs/deep.yaml", "configs/deepTest.yaml")

    elif fileName.upper() == "CUSTOM":
        configOption = None

    else:
        print("Must select model either baseline, deep or custom")
        exit()


    
    if configOption == None:
        configFile = secondOption
    
    elif secondOption.upper() == "TRAIN":
        configFile = configOption[0]

    elif secondOption.upper() == "TEST":
        configFile = configOption[1]

    else:
        print("Must select model option train, test or config file")
        exit()


    ### Load experiment configuration ###
    try:
        with open(configFile, "r") as fp:
            params = yaml.load(fp, Loader = yaml.Loader)
    except FileNotFoundError:
        raise FileNotFoundError("Must give valid config file")
        
    try:
        seed = params["seed"]

        datasetFolder = params["datasetFolder"]
        datasetClasses = params["datasetClasses"]
        dataSplit = params["dataSplit"]

        batchSize = params["batchSize"]

        trainModel = params["trainModel"]
        numEpochs = params["numEpochs"]
        learningRate = params["learningRate"]

        testModel = params["testModel"]

        saveFolder = params["saveFolder"]
        loadModel = params["loadModel"]
        
        loss = params["loss"]
        ModelType = params["ModelType"]
        
    except KeyError:
        raise KeyError("Config format invalid")


    ### Sets up random seed and experiment folder ###
    np.random.seed(seed)
    saveFolder = utility.initExpFolder(saveFolder)

    ### Load and prepare data
    print("---Loading Data---")
    x_train, y_train, x_test, y_test = loadData.loadInitial(datasetFolder, datasetClasses)
    print("---Load Complete---\n")

    
    ### Splits training data ###
    xTrain, yTrain, xValidate, yValidate = data.dataSplit(x_train, y_train, dataSplit) 


    ###  Build Training Dataset  ###
    normaliser = prePro.Normaliser()
    normXtrain = normaliser.fitTransform(xTrain)
    yTrainSmooth = prePro.labelSmoother(yTrain)
    trainingSet = data.dataLoader(normXtrain, yTrainSmooth, batchSize)

    ###  Build Validation Dataset  ###
    valNorm = normaliser.transform(xValidate)
    yTrueClass = np.argmax(yValidate, axis=1)
    valSet = data.dataLoader(valNorm, yTrueClass, batchSize)

    ###  Normalise Test Input  ###
    norXtest = normaliser.transform(x_test)
    testLabel = np.argmax(y_test, axis=1)


    ### Creates Model ###
    if ModelType == 0:
        model = baseLineModel(loss)
        print("Running baseline Model\n")

    elif ModelType == 1:
        model = deepModel(loss)
        print("Running deep Model\n")

    else:
        raise ValueError("Invalid model selection must be 0/1")
    
    
    ### Loads Model ###
    if loadModel:
        try:
            print("Loading Model Weights")
            model.loadModel(saveFolder / "bestModel.npy")
        except FileNotFoundError:
            saveFolder.rmdir()
            raise FileNotFoundError("Must first train a model with same folder name before running test")


    ### Starts Training ###
    if trainModel:
        print("Trainig")
        train(model, trainingSet, valSet, saveFolder, numEpochs, lr=learningRate)


    ### Starts Testing ###
    if testModel:
        print("Testing")
        acc = test(model, norXtest, testLabel)
        print("Test Accuracy:", acc)




if __name__ == "__main__":
    main()

