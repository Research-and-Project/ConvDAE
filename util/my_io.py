## ------------------------------

# my lib for tensorflow I/O operation

## ------------------------------


## import packages

import numpy as np
import scipy.io as sio

## load data from .mat file

# function:load and convert .mat data to numpy array type
# input:filepath str
# output:dict from .mat contains numpy ndarray data
# note: generally, the ndarray'datatype need to be further converted accordingly

def load_mat(filepath):
    data = sio.loadmat(filepath);
    header = data.pop('__header__');
    version = data.pop('__version__');
    # print('info: ', header, '\nversion: ', version, '\ndatatype: ',
    #       data[list(data.keys())[-1]].dtype )
    return data
    
def save_mat(filepath, data):
    sio.savemat(filepath, data) 

## batch iterator

# function: generate batches (from iterator) using given features and labels data
# input: batch_size; features; labels; shuffle = True是否打乱顺序（默认打乱，在test时可以选择不打乱）
# throw_insufficient = False(默认不丢弃不够batch_size长度的batch)
# output: batches from iterator

def batch_iter(batch_size, features, labels, shuffle = True, throw_insufficient = False):  
    num_examples = len(features)
    indices = list(range(num_examples))
    if shuffle:
        np.random.shuffle(indices) #随机打乱样本顺序
    if throw_insufficient:
        batch_num = num_examples//batch_size
    else:
        batch_num = np.int32(np.ceil(num_examples/batch_size))
        
    for i in range(batch_num):
        j = np.array(indices[i*batch_size:min((i+1)*batch_size, num_examples)])
        features_batch = features[j,...]
        labels_batch = labels[j,...]
        yield features_batch, labels_batch

## 
