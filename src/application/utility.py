import numpy as np
from pathlib import Path

'''
    Helper functions for IO management

    Includes saving and making experiment folder -- here to clean up training loop code
'''

def saveArray(filename, data):
    ''' Saves NumPy array'''

    with open(filename, 'wb') as f:
        np.save(f, data)

def saveNewBestSummary(filename, epoch, validation):
    ''' Saves which the peak epoch and validation score for the best model to text file
        Useful to know where the best weights came from'''
    
    with open(filename, "w") as f:
        f.write(f"New best at epoch: {epoch} with validation: {validation:.6f}")

def initExpFolder(saveFolder):
    '''Creates experiments folder if not their
        Returns the path to experiment folder'''
    ## Root will be the main project folder -- assumes utility.py in src/application/ root is two levels above
    root = Path(__file__).resolve().parents[2]

    saveFolder = root / "experiments" / saveFolder
    saveFolder.mkdir(parents=True, exist_ok=True)
    
    return saveFolder