# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:06:37 2017

@author: ThinkStation
"""

import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_1 = unpickle('./cifar-10-batches-py/data_batch_1')
train_2 = unpickle('./cifar-10-batches-py/data_batch_2')
train_3 = unpickle('./cifar-10-batches-py/data_batch_3')
train_4 = unpickle('./cifar-10-batches-py/data_batch_4')
train_5 = unpickle('./cifar-10-batches-py/data_batch_5')

train_data_1 = train_1[b'data']
train_labels_1 = train_1[b'labels']
train_data_2 = train_2[b'data']
train_labels_2 = train_2[b'labels']
train_data_3 = train_3[b'data']
train_labels_3 = train_3[b'labels']
train_data_4 = train_4[b'data']
train_labels_4 = train_4[b'labels']
train_data_5 = train_5[b'data']
train_labels_5 = train_5[b'labels']

train_data =  np.append(train_data_1, train_data_2, axis = 0)
train_data = np.append(train_data, train_data_3, axis = 0)
train_data = np.append(train_data, train_data_4, axis = 0)
train_data = np.append(train_data, train_data_5, axis = 0)

#print(train_data.shape) (50000,3072)

train_labels = np.append(train_labels_1, train_labels_2, axis = 0)
train_labels = np.append(train_labels, train_labels_3, axis = 0)
train_labels = np.append(train_labels, train_labels_4, axis = 0)
train_labels = np.append(train_labels, train_labels_5, axis = 0)


#print(train_labels.shape) (50000,)

test = unpickle('./cifar-10-batches-py/test_batch')

test_data = test[b'data']
test_labels = test[b'labels']

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(train_data, train_labels) # train the classifier on the training images and labels
Yte_predict = nn.predict(test_data) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % ( np.mean(Yte_predict == test_labels) ))
