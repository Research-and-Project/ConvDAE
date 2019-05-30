## import lib

import tensorflow as tf
from scipy.io import loadmat
import cv2
import numpy as np
import matplotlib.pyplot as plt

## load data

train_data = loadmat('D:\\1-Research\\resource\\Dataset\\MNIST2evt\\MNIST_diff_train.mat')
test_data = loadmat('D:\\1-Research\\resource\\Dataset\\MNIST2evt\\MNIST_diff_test.mat')
train_x = train_data['train_diff']
train_y = train_data['train_diff_gt']
print('train_x shape:', train_x.shape, '\ntrain_y shape:', train_y.shape)

## show image

train_x_img1 = (train_x[2]).astype('float64')
train_y_img1 = (train_y[2]).astype('float64')


cv2.imshow('train_x_img1', train_x_img1)
cv2.imshow('train_y_img1', train_y_img1)

## batch iterator

# input: batch_size; features, labels
# output: batches from iterator
def batch_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices) #随机打乱样本顺序
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i:min(i+batch_size, num_examples)])
        features_batch = features[j,:,:]
        labels_batch = labels[j,:,:]
        yield features_batch, labels_batch

## batch_iter测试
batch_size = 10

for X,Y in batch_iter(batch_size, train_x, train_y):
    X = X.astype('float64')
    Y = Y.astype('float64')
    for k in range(5):
        plt.subplot(2,5,k+1)
        plt.imshow(X[k])
        plt.title('train_x_%d'%(k+1))
        plt.xticks([])
        plt.yticks([])        
        plt.subplot(2,5,k+6)
        plt.imshow(Y[k])
        plt.title('train_y_%d'%(k+1))
        plt.xticks([])
        plt.yticks([])
    break

plt.show()
