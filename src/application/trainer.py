import numpy as np
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import framework.utility.evaluation as eval

import utility


def train(model, trainingSet, valSet, saveFolder, numEpochs = 100, lr = 0.1):
    '''
        Runs training loop
        saves best/latest model checkpoints
        saves loss and validation
    '''
    losses = []
    vals = []
    totalTime = 0
    bestVal = 0
    start = time.time()
    for i in range(numEpochs):
        stepLosses = []
        stepVals = []

        # Enable training behaviour
        model.trainMode()
        for x, y in trainingSet:
            yPred = model.forward(x)
            loss = model.loss(y, yPred)

            model.backward(y, yPred, lr)

            stepLosses.append(loss)

        # Enable inference behaviour
        model.inferenceMode()
        for x, y in valSet:
            yPred = model.forward(x)
            yPredLabel = np.argmax(yPred, axis=1)
            acc = eval.accuracy(y, yPredLabel)

            stepVals.append(acc)


        losses.append(np.mean(stepLosses))
        vals.append(np.mean(stepVals))

        interval = time.time() - start
        print(f"epoch: {i} loss: {losses[-1]:.6f} validation accuracy: {vals[-1]:.6f} took {interval:.2f}")
        totalTime += interval
        start = time.time()
        
        # Save checkpoint whenever validation accuracy improves
        if bestVal < vals[-1]:
            bestVal = vals[-1]
            model.saveModel(saveFolder / "bestModel.npy")
            utility.saveNewBestSummary(saveFolder / "bestInfo.txt", i, vals[-1])

        # Save latest weights and training history
        model.saveModel(saveFolder / "latest.npy")
        utility.saveArray(saveFolder / "trainLoss.npy", np.array(losses))
        utility.saveArray(saveFolder / "valAcc.npy", np.array(vals))

    print("total time:", totalTime)
    return losses, vals




def test(model, norXtest, testLabel):
    '''Evaluates model accuracy on full test data'''

    model.inferenceMode()

    y_ = model.forward(norXtest)

    yPredLabel = np.argmax(y_, axis=1)

    acc = eval.accuracy(testLabel, yPredLabel)

    return acc