import numpy as np
from scipy.stats import logistic

class NeuralLayer:
  def __init__(self, num_inputs, num_neurons):

    self.num_inputs = num_inputs + 1 #for bias term
    self.num_neurons = num_neurons
    self.potentials = np.zeros([self.num_neurons, 1])
    self.activations = self.potentials
    self.weights = (np.random.rand(self.num_neurons, self.num_inputs) -.5)/10
    self.Delta = np.zeros_like(self.weights)
    self.last_inputs = None

  def setPotentials(self, input_vec):
    input_vec = np.append(1, input_vec)
    self.last_inputs = input_vec
    self.potentials = np.dot(self.weights, input_vec)

  def getActivations(self):
    self.activations = np.array(logistic.cdf(self.potentials))
    return self.activations

  def backPropogate(self, delta_forward):
    g_prime = self.derivOfLogistic()
    delta = np.multiply(np.dot(self.weights.T, delta_forward), g_prime)

    print 'delta:',delta
    print 'li:', self.last_inputs
    

    lenLI = len(self.last_inputs);
    self.last_inputs = np.asarray(self.last_inputs);
    self.last_inputs.shape = (lenLI,1);

    self.Delta = self.last_inputs * delta
    print 'Delta:',self.Delta
    print 'weights:',self.weights
    self.weights -= self.Delta
    #a = self.weights + self.Delta
    return delta

  def derivOfLogistic(self):
    vs = np.vectorize(dol)
    return vs(self.last_inputs)

def dol(i):
  return i*(1-i)
