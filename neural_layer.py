import numpy as np
from scipy.stats import logistic

class NeuralLayer:
  def __init__(self, num_inputs, num_neurons):
    self.num_inputs = num_inputs
    self.num_neurons = num_neurons
    self.potentials = np.zeros([self.num_neurons, 1])
    #change to have negative values
    self.weights = np.random.rand(self.num_inputs, self.num_neurons)
    self.bias = np.random.rand(self.num_neurons, 1)
    self.Delta = np.zeros_like(self.potentials)

  def setPotentials(self, input_potentials):
    self.potentials = np.dot(input_potentials, self.weights) + self.bias

  def getPotentials(self):
    return logistic(self.potentials)

  def backPropogate(self, delta_forward):
    g_prime = self.derivOfLogistic()
    delta = np.multiply(np.transpose(self.weights)*delta_forward, g_prime)
    return delta

  def setAsInputLayer(self):
    self.weights = np.identity(self.num_inputs)
    self.bias = np.zeros([self.num_neurons, 1])

  def derivOfLogistic(self):
    return self.potentials(np.ones_like(self.potentials) - self.potentials)
