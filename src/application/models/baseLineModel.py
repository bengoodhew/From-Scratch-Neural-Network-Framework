from framework.nn.Model import Model
import framework.nn.Layers as Layers

'''
Baseline network architecture

A compact model used as a reference for comparisons
'''

class baseLineModel(Model):
    def __init__ (self, loss):
        super().__init__(loss)
        self.model = [Layers.FullyConnected(3072, 512),
                      Layers.ActivationLayer("RELU"),
                      Layers.FullyConnected(512, 256),
                      Layers.ActivationLayer("RELU"),
                      Layers.FullyConnected(256, 128),
                      Layers.ActivationLayer("RELU"),
                      Layers.FullyConnected(128, 5, initMethod = "GLOROT"),
                      Layers.ActivationLayer("softmaxWithCrossEntropy")]

    
                

    

        
        
