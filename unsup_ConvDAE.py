# -*- coding: utf-8 -*-
'''
【basic unsupervised convolutional denoising autoencoder——bsc-ConvDAE】
--max_pool in decoder, non dropout layer
contains two modes:
unspervised: input data will be randomly masked(through dropout), inputs_ & targets_ are the same (noisy data)
supervised: inputs_ & targets_ are clean data and noisy data separately

Author: zhihong (z_zhi_hong@163.com)
Date: 20190505
Modified: zhihong_20190810

'''

# In[]:
# import modules
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import os
from util import my_io

# In[] 
# environment config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.8


# In[]:
# graph reset
tf.reset_default_graph() 


# In[]:
# parameters config
# flags
SUP_FLAG = 1 # supervised learning flag

# setting
epochs = 20
batch_size = 128
learning_rate = 0.001
pic_size = [28,28]
keep_prob_v = 0.5
if SUP_FLAG:
    mask_prob_v  = 0.0 # supervised mode: mask layer' drop probability. 
else:
    mask_prob_v = 0.5  # unsupervised mode: mask layer' drop probability. 

# path
path1 = "./dataset/N_MNIST_pic/N_MNIST_pic_train.mat"
path2 = "./dataset/N_MNIST_pic/N_MNIST_pic_test.mat"

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "./model_data/"

if SUP_FLAG:
    model_name =  "bsc-ConvDAE(sup)"
else:
    model_name =  "bsc-ConvDAE(unsup)"
    
model_dir = model_name + "--" +  timestamp
model_path = model_root_path + model_dir 

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = './logs/train/bsc-ConvDAE_'+timestamp
test_log_dir = './logs/test/bsc-ConvDAE_'+timestamp

# In[]: 
# functions
# tensorflow log summary
def summaryWriter(train_writer, test_writer, record_point, run_tensor, train_feed_dict, test_feed_dict, iter):
    tr, tr_cost = sess.run([record_point, run_tensor], feed_dict=train_feed_dict)
    te, te_cost = sess.run([record_point, run_tensor], feed_dict=test_feed_dict)        
    train_writer.add_summary(tr, iter)
    test_writer.add_summary(te, iter)         
    print(iter,"Train cost:",tr_cost,"Test cost",te_cost)  

    
# In[]:
# model
# activate function
act_fun = tf.nn.relu    # inner layer act_fun
act_fun_out = tf.nn.tanh     # output layer act_fun
#act_fun = tf.nn.tanh

with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.float32, (None, *pic_size, 1), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, *pic_size, 1), name='targets_')
    keep_prob = tf.placeholder(tf.float32)    #range 0.0-1.0
    mask_prob = tf.placeholder(tf.float32)    #range 0.0-1.0
    
# net structure
# Encoder
with tf.name_scope('encoder'):
    drop = tf.nn.dropout(inputs_, 1-mask_prob)  #unsupervised：randomly masked

    conv1 = tf.layers.conv2d(drop, 64, (3,3), padding='same', activation=act_fun)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    max_p1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

    conv2 = tf.layers.conv2d(max_p1, 32, (3,3), padding='same', activation=act_fun)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    max_p2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

    conv3 = tf.layers.conv2d(max_p2, 16, (3,3), padding='same', activation=act_fun)
    conv3 = tf.nn.dropout(conv3, keep_prob)
    max_p3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

# Decoder
with tf.name_scope('decoder'):
    res4 = tf.image.resize_nearest_neighbor(max_p3, (7,7))
    conv4 = tf.layers.conv2d(res4, 16, (3,3), padding='same', activation=act_fun)
    conv4 = tf.nn.dropout(conv4, keep_prob)

    res5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
    conv5 = tf.layers.conv2d(res5, 32, (3,3), padding='same', activation=act_fun)
    conv5 = tf.nn.dropout(conv5, keep_prob)

    res6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
    conv6 = tf.layers.conv2d(res6, 64, (3,3), padding='same', activation=act_fun)
    conv6 = tf.nn.dropout(conv6, keep_prob)

# logits and outputs
with tf.name_scope('outputs'):
    logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

    outputs_ = act_fun_out(logits_, name='outputs_')

# loss and Optimizer
with tf.name_scope('loss'):
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
#     loss = tf.reduce_sum(tf.square(targets_ -  outputs_))

    loss = tf.losses.mean_squared_error(targets_ , outputs_)

    cost = tf.reduce_mean(loss)
    tf.summary.scalar('cost', cost)
    
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[]:
# load data
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)

train_x = train_data['N_MNIST_pic_train'].astype('float32')
if SUP_FLAG==0:
    train_y = train_x
else:
    train_y = train_data['N_MNIST_pic_train_gt'].astype('float32')
test_x = test_data['N_MNIST_pic_test'].astype('float32')
test_y = test_data['N_MNIST_pic_test_gt'].astype('float32')

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape, 
     '\ntest_x: ', test_x.shape, '\ttest_y: ', test_y.shape)

# to avoid OOM, use part of test dataset for testing
test_idx = np.linspace(0,len(test_x)-1,1000).astype('int32')
test_x1 = test_x[test_idx].reshape((-1, *pic_size, 1))
test_y1 = test_y[test_idx].reshape((-1, *pic_size, 1))

# data disp
#for k in range(5):
#    plt.subplot(2,5,k+1)
#    plt.imshow(train_x[k])
#    plt.title('train_x_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])        
#    plt.subplot(2,5,k+6)
#    plt.imshow(train_y[k])
#    plt.title('train_y_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])


# In[]
# initialize
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 

# In[]:
# train
test_feed_dict={inputs_: test_x1, targets_: test_y1, keep_prob: 1.0, mask_prob:0.0}

summaryWriter(writer_tr, writer_te, merged, cost, test_feed_dict, test_feed_dict, 0)
for e in range(1, 1+epochs):
    for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y, throw_insufficient=True):
        
        x = batch_x.reshape((-1, *pic_size, 1))
        y = batch_y.reshape((-1, *pic_size, 1))
            
        train_feed_dict = {inputs_: x, targets_: y, keep_prob: keep_prob_v, mask_prob: mask_prob_v}
        sess.run(optimizer, feed_dict=train_feed_dict)
        
    if e%5 == 0: 
        summaryWriter(writer_tr, writer_te, merged, cost, train_feed_dict, test_feed_dict, e)        
    
    if e%20 == 0 and e!=0:
        saver.save(sess, model_path+'my_model',global_step=e, write_meta_graph=False)
        # saver.save(sess,model_path+'my_model') 
        print('epoch %d model saved to:'%e, model_path+'my_model')


summaryWriter(writer_tr, writer_te, merged, cost, train_feed_dict, test_feed_dict, e)
    
saver.save(sess,model_path+'my_model') 
print('epoch: %d model saved to:'%e, model_path+'my_model') 

# In[61]:
# test
start = 0
end = len(test_x)-1
idx = np.linspace(start, end, 10).astype('int32')  # show 10 results at equal intervals

in_imgs = test_x[idx]
gt_imgs = test_y[idx]  

reconstructed = sess.run(outputs_, feed_dict={inputs_: in_imgs.reshape((10, *pic_size, 1)), keep_prob: 1.0, mask_prob:0.0})
reconstructed = np.squeeze(reconstructed)

    
fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((*pic_size)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)


# In[ ]:
# release
sess.close()







