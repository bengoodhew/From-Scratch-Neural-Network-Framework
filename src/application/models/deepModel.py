from framework.nn.Model import Model
import framework.nn.Layers as Layers

'''
Deep network architecture

A larger model used to evaluate how the framework handles increased network depth
'''

class deepModel(Model):
    def __init__ (self, loss):
        super().__init__(loss)
        self.model = [Layers.FullyConnected(3072, 1024),
                      Layers.BatchNormLayer(1024),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.4),


                      Layers.FullyConnected(1024, 1024),
                      Layers.BatchNormLayer(1024),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.4),


                      Layers.FullyConnected(1024, 512),
                      Layers.BatchNormLayer(512),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.4),


                      Layers.FullyConnected(512, 512),
                      Layers.BatchNormLayer(512),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.4),


                      Layers.FullyConnected(512, 512),
                      Layers.BatchNormLayer(512),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.4),


                      Layers.FullyConnected(512, 256),
                      Layers.BatchNormLayer(256),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.4),


                      Layers.FullyConnected(256, 256),
                      Layers.BatchNormLayer(256),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.4),


                      Layers.FullyConnected(256, 256),
                      Layers.BatchNormLayer(256),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.4),


                      Layers.FullyConnected(256, 128),
                      Layers.BatchNormLayer(128),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.2),


                      Layers.FullyConnected(128, 64),
                      Layers.BatchNormLayer(64),
                      Layers.ActivationLayer("RELU"),
                      Layers.DropoutLayer(0.2),


                      Layers.FullyConnected(64, 5, initMethod = "GLOROT"),
                      Layers.ActivationLayer("softmaxWithCrossEntropy")]

