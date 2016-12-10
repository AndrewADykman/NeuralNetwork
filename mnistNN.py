from neural_network import *;
import mnist;
import numpy as np;
import itertools

def main():

  #config parameters
  numEpochs = 2;
  alpha = 0.1;

  # loading in that good good data
  trainImages = np.load('data/train_images.npy');
  trainLabels = np.load('data/train_labels.npy');

  imSize = np.shape(trainImages[0]);
  inputSize = imSize[0]*imSize[1];
  numLabels = len(trainLabels[0]);

  print '# features in:',inputSize
  print '# possibe labels out:',numLabels

  #Neural Network Definition
  hiddenLayerDimensions = [ 300, 100, numLabels ];
  NN = NeuralNetwork(hiddenLayerDimensions, inputSize,alpha)
  correct = 0.0;

  for j in range(0,numEpochs):
    for i in range(0,len(trainLabels)): 

      train_input = np.reshape(trainImages[i],(inputSize,1));
      train_input = train_input.tolist();
      
      train_label = trainLabels[i];
      train_label = train_label.tolist();

      train_input = list(itertools.chain.from_iterable(train_input))
      train_label = list(itertools.chain.from_iterable(train_label))

      train_input = [(x+1)/257. for x in train_input];

      # print train_input
      # print train_label

      y_hat = NN.feedForward(train_input)
      delta = NN.computeDError(np.array(train_label), y_hat)
      NN.feedBack(delta)

      same = (np.argmax(np.asarray(y_hat)) == np.argmax(np.asarray(train_label)));
      correct += float(same);

      if (i%1000 == 0 and i != 0):
        print 'epoch//sample:',j,'//',i
        print 'label:',np.argmax(np.asarray(y_hat))
        # print train_input
        print 'y_hat:',np.argmax(np.asarray(train_label))

        total  = j*len(trainLabels) + i;

        print 'Percentage correct this chunk:', int(100*float(correct)/float(1000)),'%';
        correct = 0;


if __name__ == '__main__':
  main()

