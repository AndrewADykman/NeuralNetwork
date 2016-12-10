from neural_network import *;
import mnist;
import numpy as np;
import itertools

def main():

  testImages = np.load('data/test_images.npy');
  testLabels = np.load('data/test_labels.npy');

  print np.shape(testImages)
  print np.shape(testLabels)

if __name__ == '__main__':
  main()

