import numpy as np
from neural_layer import NeuralLayer

class NeuralNetwork:
  def __init__(self, shape):
    self.num_layers = len(shape)
    #might need to rework this
    self.network = [0.] * self.num_layers
    self.network[0] = NeuralLayer(shape[0], shape[0])
    '''
    self.network[0].setAsInputLayer()

    for layer in range(1, self.num_layers):
      self.network[layer] = NeuralLayer(shape[layer-1], shape[layer])
    '''

  def feedForward(self, inputs):
    for layer in range(self.num_layers):
      self.network[layer].setPotentials(inputs)
      inputs = self.network[layer].getPotentials()
    return inputs

  def feedBack(self, delta):
    #may need to change to not affect 0-layer
    for layer in range(self.num_layers-1, -1, -1):
      delta = self.network[layer].backPropogate(delta)

  def computerDError(self, expected, returned):
    return (returned - expected)
