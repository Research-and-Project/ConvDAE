# -*- coding: utf-8 -*-
'''
简单无监督卷积去噪自编码器ConvDAE_unsup
可调参数：超参数及激活函数等
'''

# In[45]:
#导入基本模块
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import os
from my_tf_lib import my_io

# In[] 
# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.8

# In[9]:
#重置tensorboard graph
tf.reset_default_graph() 

# In[]
# 路径参数设置
# 参数
epochs = 5
batch_size = 128
learning_rate=0.001
keep_prob_v = 0.7

# 标志
predflag = 1

# 路径
path1 = "N_MNIST_pic/N_MNIST_pic_train.mat"
path2 = "N_MNIST_pic/N_MNIST_pic_test.mat"

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "model_data/"
model_dir = "ConvDAE_unsup--" +  timestamp
model_path = model_root_path + model_dir

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = 'logs/train/ConvDAE_unsup_'+timestamp
test_log_dir = 'logs/test/ConvDAE_unsup_'+timestamp

# In[]:
# 函数定义
def summaryWriter(train_writer, test_writer, record_point, run_tensor, feed_dict, iter):
    tr, tr_cost = sess.run([record_point, run_tensor], feed_dict=feed_dict)
    te, te_cost = sess.run([record_point, run_tensor], feed_dict=feed_dict)        
    train_writer.add_summary(tr, iter)
    test_writer.add_summary(te, iter)         
    print(iter,"Train cost:",tr_cost,"Test cost",te_cost)  
    
# In[]:
# 加载数据
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)

# pic数据集
train_x = train_data['N_MNIST_pic_train'].astype('float32')
train_y = train_data['N_MNIST_pic_train'].astype('float32')  #无监督，输入输出均为相同样本
test_x = test_data['N_MNIST_pic_test'].astype('float32')
test_y = test_data['N_MNIST_pic_test_gt'].astype('float32')

# diff数据集
#train_x = train_data['train_diff'].astype('float32')
#train_y = train_data['train_diff'].astype('float32') #gt使用diff_gt
#test_x = test_data['test_diff'].astype('float32')
#test_y = test_data['test_diff_gt'].astype('float32')

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape, 
     '\ntest_x: ', test_x.shape, '\ttest_y: ', test_y.shape)

# 数据打印测试
# for k in range(5):
#     plt.subplot(2,5,k+1)
#     plt.imshow(train_x[k])
#     plt.title('train_x_%d'%(k+1))
#     plt.xticks([])
#     plt.yticks([])        
#     plt.subplot(2,5,k+6)
#     plt.imshow(train_y[k])
#     plt.title('train_y_%d'%(k+1))
#     plt.xticks([])
#     plt.yticks([])


# In[]:
# 构造模型
# 输入
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')
    keep_prob = tf.placeholder(tf.float32)    #弃权概率0.0-1.0  1.0表示不使用弃权

#选择激活函数
# act_fun = tf.nn.relu 
act_fun = tf.nn.tanh
act_fun_out = tf.nn.tanh

# Encoder
with tf.name_scope('encoder'):
    drop = tf.nn.dropout(inputs_, keep_prob) #去噪自编码随机mask输入

    conv1 = tf.layers.conv2d(drop, 64, (3,3), padding='same', activation=act_fun)
    max_p1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

    conv2 = tf.layers.conv2d(max_p1, 32, (3,3), padding='same', activation=act_fun)
    max_p2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

    conv3 = tf.layers.conv2d(max_p2, 16, (3,3), padding='same', activation=act_fun)
    max_p3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

# Decoder
with tf.name_scope('decoder'):
    res4 = tf.image.resize_nearest_neighbor(max_p3, (7,7))
    conv4 = tf.layers.conv2d(res4, 16, (3,3), padding='same', activation=act_fun)

    res5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
    conv5 = tf.layers.conv2d(res5, 32, (3,3), padding='same', activation=act_fun)

    res6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
    conv6 = tf.layers.conv2d(res6, 64, (3,3), padding='same', activation=act_fun)

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


# In[]
# 初始化会话和模型保存器及tensorboard
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 

# In[]:
# 训练

test_feed_dict={inputs_: test_x1, targets_: test_y1, keep_prob: 1.0}

summaryWriter(writer_tr, writer_te, merged, cost, test_feed_dict, -1)
for e in range(epochs):
    for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y, throw_insufficient=True):
        
        x = batch_x.reshape((-1, 28, 28, 1))
        y = batch_y.reshape((-1, 28, 28, 1))
            
        train_feed_dict = {inputs_: x, targets_: y, keep_prob: keep_prob_v}
        sess.run(optimizer, feed_dict=train_feed_dict)
        
    if e%5 == 0: 
        summaryWriter(writer_tr, writer_te, merged, cost, test_feed_dict, e)        
    
    if e%20 == 0 and e!=0:
        saver.save(sess, model_path+'my_model',global_step=e, write_meta_graph=False)
        # saver.save(sess,model_path+'my_model') 
        print('epoch %d model saved to:'%e, model_path+'my_model')


summaryWriter(writer_tr, writer_te, merged, cost, test_feed_dict, e)
    
saver.save(sess,model_path+'my_model') 
print('epoch: %d model saved to:'%e, model_path+'my_model') 

# In[61]:
# 预测
if predflag==1:
  
    # ConvDAE3
    in_imgs = test_x[600:10000:1000]
    gt_imgs = test_y[600:10000:1000]
    
    reconstructed = sess.run(outputs_, feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1)), keep_prob: 1.0})
    reconstructed = np.squeeze(reconstructed)
        
    # zzh:极性化，阈值-0.5，0.5        
    thresh = 0.25
    polarization = 0
    if polarization:     
        reconstructed[reconstructed<=-1*thresh] = -1.
        reconstructed[reconstructed>=thresh] = 1.
        reconstructed[(reconstructed>-1*thresh) & (reconstructed<thresh)] = 0.
        
    fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
    for images, row in zip([in_imgs, reconstructed, gt_imgs], axes):
        for img, ax in zip(images, row):
            ax.imshow(img.reshape((28, 28)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)


# In[ ]:
# 释放
# release
sess.close()







