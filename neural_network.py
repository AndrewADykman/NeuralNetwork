import numpy as np
from neural_layer import NeuralLayer

class NeuralNetwork:
  def __init__(self, shape, num_inputs):
    self.num_inputs = num_inputs
    self.num_layers = len(shape)
    self.network = [NeuralLayer(num_inputs, shape[0])]

    for layer in range(1, self.num_layers):
      self.network.append(NeuralLayer(shape[layer-1], shape[layer]))

  def feedForward(self, inputs):
    for layer in range(self.num_layers):
      self.network[layer].setPotentials(inputs)
      inputs = self.network[layer].getActivations()
    return inputs

  def feedBack(self, delta):
    #may need to change to not affect 0-layer
    for layer in range(self.num_layers-1, -1, -1):
      delta = self.network[layer].backPropogate(delta)

  def computeDError(self, expected, returned):
    return (returned - expected)

if __name__ == '__main__':
  
  hiddenLayerDimensions = [2];
  inputSize = 2;

  TEST_INPUT = [0.9,0.91];
  TEST_OUTPUT = [.2,.3];


  a = NeuralNetwork(hiddenLayerDimensions, inputSize)
  a.network[0].weights = np.array([[.25,.25,-.5],[-.25,-.25,.5]])
  y_hat = a.feedForward(TEST_INPUT)
  print 'initial prediction:',y_hat

  # BACKPROP ROUND 1
  for i in range(1,20):
    delta = a.computeDError(np.array(TEST_OUTPUT), y_hat)
    a.feedBack(delta)
    y_hat = a.feedForward(TEST_INPUT)
    print 'initial prediction:',y_hat

