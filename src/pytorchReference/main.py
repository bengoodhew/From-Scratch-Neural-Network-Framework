import sys
from pathlib import Path
import torch
import time
import numpy as np


from baseline import Baseline
from deep import Deep


sys.path.append(str(Path(__file__).resolve().parents[1]))

from application import loadData
from application import utility
from framework.utility import data
from framework.utility import preprocess as prePro
from framework.utility import evaluation



    
### config
lr = 0.1
loadWeights = False




try:
    modelType = sys.argv[1]
except IndexError:
    print("Must give model type -- baseline / deep")
    exit()

if modelType.upper() == "BASELINE":
    numEpochs = 8
    experimentFolder = utility.initExpFolder("pytorchExp/baseline")
    mlp = Baseline()

elif modelType.upper() == "DEEP":
    numEpochs = 116
    experimentFolder = utility.initExpFolder("pytorchExp/deep")
    mlp = Deep()
else:
    print("Must give model type -- baseline / deep")
    exit()


##Gpu accelorator
device = torch.cuda.current_device() if torch.accelerator.is_available() else "cpu"
print("Using", torch.cuda.get_device_name(device))

np.random.seed(0)
torch.random.manual_seed(0)

x_train, y_train, x_test, y_test = loadData.loadInitial("cifar-10", ['airplane', 'ship', 'automobile', 'frog', 'horse'])

### Splits training data ###
xTrain, yTrain, xValidate, yValidate = data.dataSplit(x_train, y_train, 0.8) 



###  Build Training Dataset  ###
normaliser = prePro.Normaliser()
normXtrain = normaliser.fitTransform(xTrain)
yTrainSmooth = prePro.labelSmoother(yTrain)
trainSet = torch.utils.data.TensorDataset(torch.tensor(normXtrain, dtype=torch.float32, device = device), torch.tensor(yTrainSmooth, dtype=torch.float32, device = device))
trainingSet = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)

###  Build Validation Dataset  ###
valNorm = normaliser.transform(xValidate)
yTrueClass = np.argmax(yValidate, axis=1)
valData = torch.utils.data.TensorDataset(torch.tensor(valNorm, dtype=torch.float32, device = device), torch.tensor(yTrueClass, dtype=torch.float32, device = "cpu"))
valSet = torch.utils.data.DataLoader(valData, batch_size=64, shuffle=True)

###  Normalise Test Input  ###
testXnorm = torch.tensor(normaliser.transform(x_test), dtype=torch.float32, device=device)
testLabel = np.argmax(y_test, axis=1)



if loadWeights:
    with open(experimentFolder / "best.pth", "rb") as fp:
        mlp.load_state_dict(torch.load(fp))


mlp.to(device)


optimiser = torch.optim.SGD(mlp.parameters(), lr)

criterion = torch.nn.CrossEntropyLoss()



losses = []
vals = []
bestVal = 0
bestEpoch = 0
totalTime = 0
start = time.time()

##Train loop
for i in range(numEpochs):
    stepLosses = []
    stepVals = []

    ## Training
    mlp.train()
    for x, y in trainingSet:
        optimiser.zero_grad()

        yPred = mlp(x)
        loss = criterion(yPred, y)

        loss.backward()
        optimiser.step()

        stepLosses.append(loss.item())


    ## Validation
    mlp.eval()
    for x, y in valSet:
        with torch.no_grad():
            yPred = mlp(x)
        yPredLabel = np.argmax(yPred.detach().cpu().numpy(), axis=1)
        acc = evaluation.accuracy(y.numpy(), yPredLabel)

        stepVals.append(acc)


    losses.append(np.mean(stepLosses))
    vals.append(np.mean(stepVals))
    if vals[-1] > bestVal:
        bestVal = vals[-1]
        bestEpoch = i

        with open(experimentFolder / "best.pth", "wb") as fp:
            torch.save(mlp.state_dict(), fp)

    with open(experimentFolder / "latest.pth", "wb") as fp:
        torch.save(mlp.state_dict(), fp)

    
    utility.saveArray(experimentFolder / "trainLoss.npy", np.array(losses))
    utility.saveArray(experimentFolder / "valAcc.npy", np.array(vals))

    interval = time.time() - start
    totalTime += interval
    start = time.time()
    print(f"epoch: {i} loss: {losses[-1]:.6f} validation accuracy: {vals[-1]:.6f} took {interval:.2f}")

print(f"Best val {bestVal:.6f} at epoch {bestEpoch} -- took {totalTime:.0f} to train {numEpochs} epochs")


## Testing
mlp.eval()
with torch.no_grad():
    yPred = mlp(testXnorm)
yPredLabel = np.argmax(yPred.detach().cpu().numpy(), axis=1)
acc = evaluation.accuracy(testLabel, yPredLabel)

print(f"Test accuracy: {acc}")
