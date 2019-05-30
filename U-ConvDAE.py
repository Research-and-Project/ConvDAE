# -*- coding: utf-8 -*-
'''
【U型卷积去噪自编码器：U-ConvDAE】
无监督：对网络的输入层进行随机mask（dropout），网络的inputs_和targets_均输入待去噪数据
有监督：不用对网络的输入层进行随机mask（dropout），网络的inputs_和targets_分别为含噪数据和无噪数据
可调参数：超参数及激活函数等
'''

# In[45]:
#导入基本模块
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from my_tf_lib import my_io
import os


# In[] 
# 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.8


# In[9]:
#重置tensorboard graph
tf.reset_default_graph() 


# In[9]:
# 参数标志路径
# 参数
epochs = 2
batch_size = 128
learning_rate=0.001
keep_prob_v = 0.9 #dropout layers' keep probability
mask_prob_v  = 0.5 #unsupervised mode: mask layer' drop probability. 

# 标志
predflag = 1
in_data_flag = 1
unsup =  0 # unsupervised learning flag

# 路径
timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "D:/1-Document/data/model_data/ConvDAE/"
model_dir = "U-ConvDAE--" +  timestamp
model_path = model_root_path + model_dir

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = 'logs/train/U-ConvDAE-diff_'+timestamp
test_log_dir = 'logs/test/U-ConvDAE-diff_'+timestamp

if in_data_flag==1:
    path1 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_train.mat"
    path2 = "D:/1-Codes/matlab/resource/dataset/N_MNIST_pic/N_MNIST_pic_test.mat"
if in_data_flag==2:    
    path1 = "D:/1-Codes/matlab/resource/dataset/MNIST_diff/MNIST_diff_train.mat"
    path2 = "D:/1-Codes/matlab/resource/dataset/MNIST_diff/MNIST_diff_test.mat"

# In[]:
# 函数定义
def summaryWriter(train_writer, test_writer, record_point, run_tensor, feed_dict, iter):
    tr, tr_cost = sess.run([record_point, run_tensor], feed_dict=feed_dict)
    te, te_cost = sess.run([record_point, run_tensor], feed_dict=feed_dict)        
    train_writer.add_summary(tr, iter)
    test_writer.add_summary(te, iter)         
    print(iter,"Train cost:",tr_cost,"Test cost",te_cost)    
    
# In[]:
# 加载数据及预处理
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)

if in_data_flag==1:
    train_x = train_data['N_MNIST_pic_train'].astype('float32')
    if unsup==1:
        train_y = train_x
    else:
        train_y = train_data['N_MNIST_pic_train_gt'].astype('float32')
    test_x = test_data['N_MNIST_pic_test'].astype('float32')
    test_y = test_data['N_MNIST_pic_test_gt'].astype('float32')

if in_data_flag==2:
    train_x = train_data['train_diff'].astype('float32')
    if unsup==1:
        train_y = train_x
    else:
        train_y = train_data['train_diff_gt'].astype('float32')
    test_x = test_data['test_diff'].astype('float32')
    test_y = test_data['test_diff_gt'].astype('float32')

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape, 
     '\ntest_x: ', test_x.shape, '\ttest_y: ', test_y.shape)

# 取部分测试集，防止OOM
test_x1 = test_x[0:10000:10].reshape((-1, 28, 28, 1))
test_y1 = test_y[0:10000:10].reshape((-1, 28, 28, 1))  
## 数据打印测试
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

# In[]:
# 构造模型

# 选择激活函数
# act_fun = tf.nn.relu 
act_fun = tf.nn.tanh
act_fun_out = tf.nn.tanh

# 定义deconv函数
def deconv(input, deconv_weight, output_shape, strides):
    dyn_input_shape = tf.shape(input)
    batch_size = dyn_input_shape[0]
    output_shape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])
    output = tf.nn.conv2d_transpose(input, deconv_weight, output_shape, strides, padding="SAME")
    return output

# 定义网络结构
# 输入
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
    targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')
    keep_prob = tf.placeholder(tf.float32)    #弃权概率0.0-1.0  1.0表示不使用弃权
    mask_prob = tf.placeholder(tf.float32)    #弃权概率0.0-1.0  1.0表示不使用弃权
    
# Encoder
with tf.name_scope('encoder'):
    if unsup==1:
        inputs_ = tf.nn.dropout(inputs_, 1-mask_prob) #无监督：去噪自编码随机mask输入层
    conv1 = tf.layers.conv2d(inputs_, 32, (3,3), padding='same', activation=act_fun)  # 28*28*32
    conv1 = tf.nn.dropout(conv1, keep_prob) #去噪自编码随机mask输入    
    maxp1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same') # 14*14*32

    conv2 = tf.layers.conv2d(maxp1, 64, (3,3), padding='same', activation=act_fun) # 14*14*64
    conv2 = tf.nn.dropout(conv2, keep_prob) #去噪自编码随机mask输入    
    maxp2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same') # 7*7*64

    conv3 = tf.layers.conv2d(maxp2, 64, (3,3), padding='same', activation=act_fun) # 7*7*64
    conv3 = tf.nn.dropout(conv3, keep_prob) #去噪自编码随机mask输入     
    maxp3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same') # 4*4*64


# mid
    mid1 = tf.layers.conv2d(maxp3, 128, (3,3), padding='same', activation=act_fun) # 4*4*128
    mid2 = tf.layers.conv2d(mid1, 64, (3,3), padding='same', activation=act_fun) # 4*4*64
    
# decoder
with tf.name_scope('decoder'):
    deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), name='deconv_weight_1') 
    deconv1 = deconv(mid2, deconv_weight_1, [batch_size, 7, 7, 64], [1, 2, 2, 1]) # 7*7*64
    concat1 = tf.concat([deconv1, conv3], 3) # 7*7*128
    deconv_1 = tf.layers.conv2d(concat1, 64, (3,3), padding='same', activation=act_fun) # 7*7*64
    deconv_1 = tf.nn.dropout(deconv_1, keep_prob) #去噪自编码随机mask输入 

    deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), name='deconv_weight_2') 
    deconv2 = deconv(deconv_1, deconv_weight_2, [batch_size, 14, 14, 64], [1, 2, 2, 1]) # 14*14*64
    concat2 = tf.concat([deconv2, conv2], 3) # 14*14*128
    deconv_2 = tf.layers.conv2d(concat2, 32, (3,3), padding='same', activation=act_fun) # 14*14*32
    deconv_2 = tf.nn.dropout(deconv_2, keep_prob) #去噪自编码随机mask输入

    deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], stddev=0.1), name='deconv_weight_3')
    deconv3 = deconv(deconv_2, deconv_weight_3, [batch_size, 28, 28, 32], [1, 2, 2, 1]) # 28*28*32
    concat3 = tf.concat([deconv3, conv1], 3) # 28*28*64    
    deconv_3 = tf.layers.conv2d(concat3, 1, (3,3), padding='same', activation=act_fun) # 28*28*1
    deconv_3 = tf.nn.dropout(deconv_3, keep_prob) #去噪自编码随机mask输入 

with tf.name_scope('outputs'):
    outputs_ = tf.layers.conv2d(deconv_3, 1, (3,3), padding='same', activation=act_fun_out) # 28*28*1
    
with tf.name_scope('loss'):
    # cross entropy loss，由于包含负值像素，可能有问题
#     xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
#     cost = tf.reduce_mean(xentropy)

    # mse loss
    mse = tf.losses.mean_squared_error(targets_ , outputs_)
    cost = tf.reduce_mean(mse)
    
    tf.summary.scalar('cost', cost)
    
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[]
# 初始化会话和模型保存器及tensorboard
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 


# In[22]:
# 训练
test_feed_dict={inputs_: test_x1, targets_: test_y1, keep_prob: 1.0, mask_prob: 0.0}

summaryWriter(writer_tr, writer_te, merged, cost, test_feed_dict, -1)
for e in range(epochs):
    for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y, throw_insufficient=True):
        
        x = batch_x.reshape((-1, 28, 28, 1))
        y = batch_y.reshape((-1, 28, 28, 1))
            
        train_feed_dict = {inputs_: x, targets_: y, keep_prob: keep_prob_v, mask_prob: mask_prob_v}
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
    thresh = 0.5
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
# sess.close()

