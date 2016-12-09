import numpy as np

class NeuralLayer:
  def __init__(self, num_inputs, num_neurons):
    self.num_inputs = num_inputs
    self.num_neurons = num_neurons
    self.potentials = np.zeros([self.num_neurons, 1])
    #change to have negative values
    self.transform_matrix = np.random.rand(self.num_inputs, self.num_neurons)
    self.bias = np.random.rand(self.num_neurons, 1)

  def setPotentials(self, input_potentials):
    self.potentials = np.dot(input_potentials, self.transform_matrix) + self.bias

  def getPotentials(self):
    return self.potentials

  def backPropogate(self, guilt_vector):
    pass
    #alter transform matrix
    #create pass-back guilt vector

  def setAsInputLayer(self):
    self.transform_matrix = np.identity(self.num_inputs)
    self.bias = np.zeros([self.num_neurons, 1])
