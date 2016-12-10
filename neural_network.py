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
      # print 'delta',layer,':',delta
      
      delta = self.network[layer].backPropogate(delta)
      delta = delta[1:len(delta)];

  def computeDError(self, expected, returned):
    return (returned - expected)

def main():
  TEST_INPUT = [0.9,0.91,.1];
  TEST_OUTPUT = [.5,.3,.2, 1];

  inputSize = len(TEST_INPUT);
  hiddenLayerDimensions = [ 999, 8998, 6, 3, len(TEST_OUTPUT) ];

  a = NeuralNetwork(hiddenLayerDimensions, inputSize)
  y_hat = a.feedForward(TEST_INPUT)
  print 'initial prediction:',y_hat

  # BACKPROP ROUNDS
  for i in range(1,50):
    delta = a.computeDError(np.array(TEST_OUTPUT), y_hat)
    a.feedBack(delta)
    y_hat = a.feedForward(TEST_INPUT)
    print 'updated prediction:',y_hat

if __name__ == '__main__':
  main()
