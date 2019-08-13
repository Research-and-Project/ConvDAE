# -*- coding: utf-8 -*-
"""

【finetune for ConvDAE】

Author: zhihong (z_zhi_hong@163.com)
Date: 20190505
Modified: zhihong_20190810

"""


# In[]:
# import modules
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from time import time
import numpy as np
from util import my_io
import os


# In[] 
# environment config
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


# In[]:
# graph reset
tf.reset_default_graph() 


# In[]:
# parameters config
# flag
SUP_FLAG = 0 # supervised learning flag
TEST_FLAG = 1

# setting
epochs = 2
batch_size = 128
learning_rate=0.001
model_ind = -1  # model index
pic_size = [28,28]
keep_prob_v = 0.7
if SUP_FLAG:
    mask_prob_v  = 0.0 # supervised mode: mask layer' drop probability. 
else:
    mask_prob_v = 0.4  # unsupervised mode: mask layer' drop probability. 


t1 = time()

# data path
path1 = "./dataset/N_MNIST_pic/N_MNIST_pic_train.mat"
path2 = "./dataset/N_MNIST_pic/N_MNIST_pic_test.mat"

# old model path
root_old_model_path = "./model_data/"
old_model_dir = "L-ConvDAE(unsup)--08-10_20-32/"
old_model_path = root_old_model_path + old_model_dir
old_model_name = 'my_model'

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "./model_data/"
model_dir = old_model_dir[:-1]+'_finetune_'+timestamp
model_path = model_root_path + model_dir

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = './logs/train/' + old_model_dir[:-1] + '_finetune_'+timestamp
test_log_dir = './logs/test/' + old_model_dir[:-1]  + '_finetune_' + timestamp


# In[]:
# load model
# initialize
sess = tf.Session(config=config)

# fine Op names
#layers = [op.name for op in graph.get_operations() if op.type == 'train' in op.name]   找op的方法  
#tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称

restorer = tf.train.import_meta_graph(old_model_path+old_model_name+'.meta')

ckpt = tf.train.get_checkpoint_state(old_model_path)
if ckpt:
    ckpt_states = ckpt.all_model_checkpoint_paths
    restorer.restore(sess, ckpt_states[model_ind])

# load Ops and variables according to old model and your need
graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name("inputs/inputs_:0")
mask_prob = graph.get_tensor_by_name("inputs/Placeholder_1:0")
targets_ = graph.get_tensor_by_name("inputs/targets_:0")
keep_prob = graph.get_tensor_by_name("inputs/Placeholder:0")
merged = graph.get_tensor_by_name("Merge/MergeSummary:0")       
outputs_ = graph.get_tensor_by_name("outputs/conv2d/Tanh:0")
cost = graph.get_tensor_by_name("loss/Mean:0")  
training_op = graph.get_operation_by_name('train/Adam')


# In[]:
# load data
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)

train_x = train_data['N_MNIST_pic_train'].astype('float32')
test_x = test_data['N_MNIST_pic_test'].astype('float32')
if SUP_FLAG==0:
    train_y = train_x
#    test_y = test_x
else:
    train_y = train_data['N_MNIST_pic_train_gt'].astype('float32')
#    test_y = test_data['N_MNIST_pic_test_gt'].astype('float32')
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


# In[22]:
# train
# initialize
sess.run(tf.global_variables_initializer())    

# tensorboard
writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 

# saver
saver = tf.train.Saver()

# original performance
cost_v = sess.run(cost, feed_dict={inputs_: test_x1, targets_:test_y1, keep_prob:1.0, mask_prob:0.0})
print('\nfinetune mode:\n\nimport model: ', ckpt_states[model_ind], 
    '\ncurrent test cost: ', cost_v, '\n\nfinetuning start!\n')

for e in range(1, 1+epochs):
    for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y):
        x = batch_x.reshape((-1, *pic_size, 1))
        y = batch_y.reshape((-1, *pic_size, 1))
        sess.run(training_op, feed_dict={inputs_: x, targets_: y, keep_prob:keep_prob_v, mask_prob:mask_prob_v})

    if e%1 == 0: 
        tr, tr_cost = sess.run([merged, cost], feed_dict={inputs_: x, targets_: y, keep_prob:keep_prob_v,mask_prob:mask_prob_v})
        te, te_cost = sess.run([merged, cost], feed_dict={inputs_: test_x1, targets_: test_y1, keep_prob:1.0, mask_prob:0.0})
    
        writer_tr.add_summary(tr, e)
        writer_te.add_summary(te, e)     

        print(e,"Train cost:",tr_cost,"Test cost",te_cost)
    
    if e%20 == 0 and e!=0:
        saver.save(sess, model_path+'my_model',global_step=e, write_meta_graph=False)
        # saver.save(sess,model_path+'my_model') 
        print('epoch %d model saved to:'%e, model_path+'my_model')
    
saver.save(sess,model_path+'my_model') 
print('epoch: %d, model saved to:'%e, model_path+'my_model') 


# In[61]:
# test
if TEST_FLAG==1:
    start = 0
    end = len(test_x)-1
    idx = np.linspace(start, end, 10).astype('int32')  # show 10 results at equal intervals
    
    in_imgs = test_x[idx]
    gt_imgs = test_y[idx]   


reconstructed = sess.run(outputs_, 
                         feed_dict={inputs_: in_imgs.reshape((10, *pic_size, 1)), keep_prob:1.0, mask_prob:mask_prob_v})


fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((*pic_size)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)


# In[ ]:
# release
#sess.close()

