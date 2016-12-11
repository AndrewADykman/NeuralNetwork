from batch_neural_network import *
# import mnist
import numpy as np
import itertools
import pickle
import argparse

def reshapeInstance(t_input, t_label, numElements):

  #reshape and scale // make it a list
  t_input = np.reshape(t_input,(numElements,1))
  t_input = (t_input + 1)/257.
  t_input = t_input.tolist()
  t_input = list(itertools.chain.from_iterable(t_input))

  #make it a list
  t_label = t_label.tolist()
  t_label = list(itertools.chain.from_iterable(t_label))

  return (t_input, t_label)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--error-file', type=str, default='error_batch.pickle')
  parser.add_argument('--num-epochs', type=int, default=3)
  parser.add_argument('--alpha', type=float, default=.03)
  parser.add_argument('--batch-size', type=int, default=2, help='number of examples per batch, -1 is full batch')
  args = parser.parse_args()
  error_file = args.error_file

  #config parameters
  numEpochs = args.num_epochs
  alpha = args.alpha
  batchSize = args.batch_size

  # loading in that good good data
  trainImages = np.load('data/train_images.npy')
  trainLabels = np.load('data/train_labels.npy')
  
  testImages = np.load('data/test_images.npy')
  testLabels = np.load('data/test_labels.npy')

  imSize = np.shape(trainImages[0])
  inputSize = imSize[0]*imSize[1]
  numLabels = len(trainLabels[0])

  #print '# features in:',inputSize
  #print '# possibe labels out:',numLabels

  #Neural Network Definition
  hiddenLayerDimensions = [ 300, 30, numLabels ]
  NN = BatchNeuralNetwork(hiddenLayerDimensions, inputSize,alpha)
  correctTest = 0.0
  error_over_iters = np.array([], dtype='float64')
  if batchSize == -1: batchSize = len(trainLabels)

  for j in range(0,numEpochs):
    i_last = 0
    for i in range(0,len(trainLabels),batchSize): 

      upperBound = i+batchSize
      if upperBound > len(trainLabels) - 1:
        upperBound = len(trainLabels) - 1

      inputs = trainImages[i:upperBound]
      labels = trainLabels[i:upperBound]

      train_inputs = []
      train_labels = []
      bsBatchSize = upperBound - i
      for k in range(0, bsBatchSize):
        (train_input, train_label) = reshapeInstance(inputs[k], labels[k], inputSize)
        train_inputs.append(train_input)
        train_labels.append(train_label)

      y_hats = NN.feedForward(train_inputs)
      outputDeltas = NN.computeDError(train_labels, y_hats)
      NN.feedBack(outputDeltas)

      # print y_hats
      for a in range(0,len(y_hats)):
        y_hat = y_hats[a]
        train_label = train_labels[a]
        same = (np.argmax(np.asarray(y_hat)) == np.argmax(np.asarray(train_label)))
        correctTest += float(same)

      if (i % 1000 == 0 and i != 0):
        denom = i - i_last
        i_last = i
        train_accuracy = 100 - (100 * (float(correctTest)/float(denom)))
        #print 'epoch//sample:',j,'//',i
        #print 'Train Accuracy:', int(train_accuracy),'%'
        correctTest = 0
        error_over_iters = np.append(error_over_iters, train_accuracy)
      #   correct = 0;
      #   for idx in range(0,len(testLabels)):
      #     (test_input, test_label) = reshapeInstance(testImages[idx], testLabels[idx], inputSize);
      #     y_hat = NN.feedForward(test_input)
          
      #     same = (np.argmax(np.asarray(y_hat)) == np.argmax(np.asarray(test_label)));
      #     correct += float(same);
          
      #   print 'Test Accuracy:', int(100 * (float(correct)/float(idx+1)) ),'%';
  with open(error_file, 'wb') as f:
    pickle.dump(error_over_iters, f)

if __name__ == '__main__':
  main()
