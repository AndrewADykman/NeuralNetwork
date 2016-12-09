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
  
  a = NeuralNetwork([2], 2)
  a.network[0].weights = np.array([[.25,.25,-.5],[-.25,-.25,.5]])
  
  y_hat = a.feedForward([1,1])
  #print out
  delta = a.computeDError(np.array([-1,1]), y_hat)
  a.feedBack(delta)
  # print 'weights (pre-BP):',a.network[0].weights
  # print 'delta:', a.network[0].Delta/2.
  a.network[0].weights = a.network[0].weights + a.network[0].Delta/2.
  # print 'weights (post-BP):',a.network[0].weights

  y_hat = a.feedForward([1,1])
  # print 'y_hat:',y_hat


  delta = a.computeDError(np.array([-1,1]), y_hat)
  a.feedBack(delta)
  a.network[0].weights = a.network[0].weights + a.network[0].Delta/2.
  y_hat = a.feedForward([1,1])
  # print 'y_hat:',y_hat

