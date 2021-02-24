# -*- coding: utf-8 -*-
'''
【"U" shape convolutional denoising autoencoder——U-ConvDAE】
--skip connection and deconvolution
contains two modes:
unspervised: input data will be randomly masked(through dropout), inputs_ & targets_ are the same (noisy data)
supervised: inputs_ & targets_ are clean data and noisy data separately

Author: zhihong (z_zhi_hong@163.com)
Date: 20190505
Modified: zhihong_20210223

'''

# In[]:
# import modules
import numpy as np
import tensorflow as tf
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
from util import my_io
import os


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
SUP_FLAG = 0  # supervised learning flag

# setting
epochs = 50
batch_size = 32
learning_rate=0.001
keep_prob_v = 0.7 #dropout layers' keep probability
pic_size = [64,64] # picture size
if SUP_FLAG:
    mask_prob_v  = 0.0 # supervised mode: mask layer' drop probability. 
else:
    mask_prob_v = 0.3  # unsupervised mode: mask layer' drop probability. 
    
# dataset path
# train_path = "./dataset/N_MNIST_pic/N_MNIST_pic_train.mat"
# test_path = "./dataset/N_MNIST_pic/N_MNIST_pic_test.mat"
train_path = "./dataset/single_molecule_localization/sml_train.mat"
test_path = "./dataset/single_molecule_localization/sml_test.mat"

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "./model_data/"

if SUP_FLAG:
    model_name =  "U-ConvDAE(sup)"
else:
    model_name =  "U-ConvDAE(unsup)"
    
model_dir = model_name + "--" +  timestamp
model_path = model_root_path + model_dir 

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = './logs/train/'+model_name+'_'+timestamp
test_log_dir = './logs/test/'+model_name+'_'+timestamp


# In[]: 
# functions
# tensorflow log summary
def summaryWriter(train_writer, test_writer, record_point, run_tensor, train_feed_dict, test_feed_dict, iter):
    tr, tr_cost = sess.run([record_point, run_tensor], feed_dict=train_feed_dict)
    te, te_cost = sess.run([record_point, run_tensor], feed_dict=test_feed_dict)        
    train_writer.add_summary(tr, iter)
    test_writer.add_summary(te, iter)         
    print("Epoch:",iter,"Train cost:",tr_cost,"Test cost",te_cost)   

# deconv
def deconv(input, deconv_weight, output_shape, strides):
    dyn_input_shape = tf.shape(input)
    batch_size = dyn_input_shape[0]
    output_shape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])
    output = tf.nn.conv2d_transpose(input, deconv_weight, output_shape, strides, padding="SAME")
    return output  


# In[]:
# model
# activate function
act_fun = tf.nn.relu    # inner layer act_fun
act_fun_out = tf.nn.relu     # output layer act_fun, for slm
# act_fun_out = tf.nn.tanh     # output layer act_fun, for N-MNIST


# net structure
# input
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.float32, (None, *pic_size, 1), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, *pic_size, 1), name='targets_')
    keep_prob = tf.placeholder(tf.float32)    #range 0.0-1.0
    mask_prob = tf.placeholder(tf.float32)    #range 0.0-1.0
    
# Encoder
with tf.name_scope('encoder'):
    mask_inputs_ = tf.nn.dropout(inputs_, 1-mask_prob) #unsupervised：randomly masked
    conv1 = tf.layers.conv2d(mask_inputs_, 64, (3,3), padding='same', activation=act_fun)  # pic_size*32
    conv1 = tf.nn.dropout(conv1, keep_prob)   
    maxp1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same') # pic_size/2*32

    conv2 = tf.layers.conv2d(maxp1, 32, (3,3), padding='same', activation=act_fun) # pic_size/2*64
    conv2 = tf.nn.dropout(conv2, keep_prob)
    maxp2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same') # pic_size/4*64

    conv3 = tf.layers.conv2d(maxp2, 32, (3,3), padding='same', activation=act_fun) # pic_size/4*64
    conv3 = tf.nn.dropout(conv3, keep_prob)
    maxp3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same') # pic_size/8*64


# mid
    mid1 = tf.layers.conv2d(maxp3, 16, (3,3), padding='same', activation=act_fun) # pic_size/8*128
    mid2 = tf.layers.conv2d(mid1, 32, (3,3), padding='same', activation=act_fun) # pic_size/8*64
    
# decoder
with tf.name_scope('decoder'):
    deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=0.1), name='deconv_weight_1') 
    deconv1 = deconv(mid2, deconv_weight_1, [batch_size, np.ceil(pic_size[0]/4), np.ceil(pic_size[1]/4),32], [1, 2, 2, 1]) # pic_size/4*64
    concat1 = tf.concat([deconv1, conv3], 3) # pic_size/4*128
    deconv_1 = tf.layers.conv2d(concat1, 32, (3,3), padding='same', activation=act_fun) #pic_size/4*64
    deconv_1 = tf.nn.dropout(deconv_1, keep_prob) 

    deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=0.1), name='deconv_weight_2') 
    deconv2 = deconv(deconv_1, deconv_weight_2, [batch_size, np.ceil(pic_size[0]/2), np.ceil(pic_size[1]/2), 32], [1, 2, 2, 1]) # pic_size/2*64
    concat2 = tf.concat([deconv2, conv2], 3) # pic_size/2*128
    deconv_2 = tf.layers.conv2d(concat2, 64, (3,3), padding='same', activation=act_fun) # pic_size/2*32
    deconv_2 = tf.nn.dropout(deconv_2, keep_prob)

    deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), name='deconv_weight_3')
    deconv_3 = deconv(deconv_2, deconv_weight_3, [batch_size, pic_size[0], pic_size[1], 64], [1, 2, 2, 1]) # pic_size*32
    
    # delete the longest skip connection
    # deconv3 = deconv(deconv_2, deconv_weight_3, [batch_size, pic_size[0], pic_size[1], 32], [1, 2, 2, 1]) # pic_size*32
    # concat3 = tf.concat([deconv3, conv1], 3) # pic_size*64    
    # deconv_3 = tf.layers.conv2d(concat3, 1, (3,3), padding='same', activation=act_fun) # pic_size*1
    
    deconv_3 = tf.nn.dropout(deconv_3, keep_prob)

with tf.name_scope('outputs'):
    outputs_ = tf.layers.conv2d(deconv_3, 1, (3,3), padding='same', activation=act_fun_out)
    
with tf.name_scope('loss'):
    # cross entropy loss
#     xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
#     cost = tf.reduce_mean(xentropy)

    # mse loss
    mse = tf.losses.mean_squared_error(targets_ , outputs_)
    cost = tf.reduce_mean(mse)
    
    tf.summary.scalar('cost', cost)
    
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[]:
# load data
train_data = my_io.load_mat(train_path)
test_data = my_io.load_mat(test_path)

train_x = train_data['data'].astype('float32')
test_x = test_data['data'].astype('float32')
if SUP_FLAG==0:
    train_y = train_x
    test_y = test_x
else:
    train_y = train_data['data_gt'].astype('float32')
    test_y = test_data['data_gt'].astype('float32')

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
test_feed_dict={inputs_: test_x1, targets_: test_y1, keep_prob: 1.0, mask_prob: 0.0}

summaryWriter(writer_tr, writer_te, merged, cost, test_feed_dict, test_feed_dict, 0)
time_start = time()

for e in range(1, 1+epochs):
    for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y, throw_insufficient=True):
        
        x = batch_x.reshape((-1, *pic_size, 1))
        y = batch_y.reshape((-1, *pic_size, 1))
            
        train_feed_dict = {inputs_: x, targets_: y, keep_prob: keep_prob_v, mask_prob: mask_prob_v}
        sess.run(optimizer, feed_dict=train_feed_dict)
        
    if e%10 == 0: 
        time_cost = time()-time_start
        summaryWriter(writer_tr, writer_te, merged, cost, train_feed_dict, test_feed_dict, e)
        
        res_imgs = sess.run(outputs_, feed_dict={inputs_: test_x1, targets_: test_y1,keep_prob:1.0, mask_prob:0.0})
        res_imgs = np.squeeze(res_imgs)
        data_save = {'reconstructed': res_imgs}
        my_io.save_mat(test_log_dir+'/'+ test_path.split('/')[-1][0:-4]+'_epoch'+str(e)+'.mat', data_save)
        print('Time:', time_cost, '   Reconstruction test data saved to :',test_log_dir + '\n')     
    
    if e%20 == 0 and e!=0:
        saver.save(sess, model_path+'my_model',global_step=e, write_meta_graph=False)
        # saver.save(sess,model_path+'my_model') 
        print('epoch %d model saved to:'%e, model_path+'my_model\n')


summaryWriter(writer_tr, writer_te, merged, cost, train_feed_dict, test_feed_dict, e)
saver.save(sess,model_path+'my_model') 
print('epoch: %d, model saved to:'%e, model_path+'my_model') 

 
# In[]:
# test
start = 0
end = len(test_x)-1
idx = np.linspace(start, end, 10).astype('int32')  # show 10 results at equal intervals

in_imgs = test_x[idx]
gt_imgs = test_y[idx]    

reconstructed = sess.run(outputs_, feed_dict={inputs_: in_imgs.reshape((10, *pic_size, 1)), keep_prob: 1.0, mask_prob: 0.0})
reconstructed = np.squeeze(reconstructed)

    
fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((*pic_size)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)
plt.show()

# In[ ]:
# release
sess.close()

