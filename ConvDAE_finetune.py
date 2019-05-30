# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:10:49 2019
【累积重构-卷积去噪自编码】
finetune

@author: dawnlh
"""


# In[45]:
#导入基本模块
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from time import time
from my_tf_lib import my_io
import os


# In[]
#运行环境配置
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


# In[9]:
#重置tensorboard graph
tf.reset_default_graph() 


# In[9]:
# 参数标志路径
# 参数
epochs = 10
batch_size = 128
learning_rate=0.001
model_ind = -1

# 标志
finetuneState = 1 #微调模型标志,0-pred， 1-finetune
t1 = time()


# 路径
path1 = r"D:\1-Codes\matlab\resource\dataset\N_MNIST_pic\N_MNIST_pic_train.mat"
path2 = r"D:\1-Codes\matlab\resource\dataset\N_MNIST_pic\N_MNIST_pic_test.mat"

timestamp = '{:%m-%d_%H-%M/}'.format(datetime.now())
model_root_path = "D:/1-Document/data/model_data/ConvDAE/"
model_dir = "ConvDAE_3_2_xentrp--" +  timestamp
model_path = model_root_path + model_dir

if not os.path.isdir(model_path):
   os.makedirs(model_path)

train_log_dir = 'logs/train/ConvDAE_3_2_xentrp_'+timestamp
test_log_dir = 'logs/test/ConvDAE_3_2_xentrp_'+timestamp

# old model to fnetune
if finetuneState==1: 
#    root_old_model_path = "D:/1-Document/data/model_data/ConDAE/good_bak/"
    root_old_model_path = "D:/1-Document/data/model_data/ConvDAE/"
    old_model_dir = "ConvDAE_2--03-29_10-36(cost-0.027)/"
    old_model_path = root_old_model_path + old_model_dir
    old_model_name = 'my_model'
    
# In[]:
# 数据读取
train_data = my_io.load_mat(path1)
test_data = my_io.load_mat(path2)
train_x = train_data['N_MNIST_pic_train'].astype('float32')
train_y = train_data['N_MNIST_pic_train_gt'].astype('float32')
test_x = test_data['N_MNIST_pic_test'].astype('float32')
test_y = test_data['N_MNIST_pic_test_gt'].astype('float32')

print('train_x: ', train_x.shape, '\ttrain_y: ', train_y.shape, 
     '\ntest_x: ', test_x.shape, '\ttest_y: ', test_y.shape)

# 全部测试会OOM
test_x1 = test_x[500:10000:1000].reshape((-1, 28, 28, 1))
test_y1 = test_y[500:10000:1000].reshape((-1, 28, 28, 1)) 
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


# In[] 模型加载
# 初始化
sess = tf.Session(config=config)

# 加载
#layers = [op.name for op in graph.get_operations() if op.type == 'train' in op.name]   找op的方法  
#tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称

restorer = tf.train.import_meta_graph(old_model_path+old_model_name+'.meta')

ckpt = tf.train.get_checkpoint_state(old_model_path)
if ckpt:
    ckpt_states = ckpt.all_model_checkpoint_paths
    restorer.restore(sess, ckpt_states[model_ind])

graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name("inputs/inputs_:0")
targets_ = graph.get_tensor_by_name("inputs/targets_:0")
merged = graph.get_tensor_by_name("Merge/MergeSummary:0")       
outputs_ = graph.get_tensor_by_name("outputs/outputs_:0")
cost = graph.get_tensor_by_name("loss/Mean:0")  
training_op = graph.get_operation_by_name('train/Adam')

cost_v = sess.run(cost, feed_dict={inputs_: test_x1, targets_:test_y1})  #for SeqRNN2--LSTM
print('\nfinetune mode:\n\nimport model: ', ckpt_states[model_ind], 
    '\ncurrent test cost: ', cost_v, '\n\nfinetuning start!\n')



# In[22]:
# 训练
# 初始化
sess.run(tf.global_variables_initializer())    

# tensorboard
writer_tr = tf.summary.FileWriter(train_log_dir, sess.graph)
writer_te = tf.summary.FileWriter(test_log_dir)
merged = tf.summary.merge_all() 

# saver
saver = tf.train.Saver()

for e in range(epochs):
    for batch_x, batch_y in my_io.batch_iter(batch_size, train_x, train_y):
        x = batch_x.reshape((-1, 28, 28, 1))
        y = batch_y.reshape((-1, 28, 28, 1))
        sess.run(training_op, feed_dict={inputs_: x, targets_: y})

    if e%1 == 0: 
        tr, tr_cost = sess.run([merged, cost], feed_dict={inputs_: x, targets_: y})
        te, te_cost = sess.run([merged, cost], feed_dict={inputs_: test_x1, targets_: test_y1})
    
        writer_tr.add_summary(tr, e)
        writer_te.add_summary(te, e)     

        print(e,"Train cost:",tr_cost,"Test cost",te_cost)

    
    if e%20 == 0 and e!=0:
        saver.save(sess, model_path+'my_model',global_step=e, write_meta_graph=False)
        # saver.save(sess,model_path+'my_model') 
        print('epoch %d model saved to:'%e, model_path+'my_model')
    
saver.save(sess,model_path+'my_model') 
print('epoch: %d model saved to:'%e, model_path+'my_model') 


# In[61]:
# 预测
if finetuneState==0:
    
    # k=10
    # in_imgs = abs(test_x[k:k+10])
    # gt_imgs = abs(test_y[k:k+10])
    # in_imgs = test_x[k:k+10]
    # gt_imgs = test_y[k:k+10]
    
    # ConvDAE3
    in_imgs = test_x[600:10000:1000]
    gt_imgs = test_y[600:10000:1000]
    
    reconstructed = sess.run(outputs_, 
                             feed_dict={inputs_: in_imgs.reshape((10, 28, 28, 1))})
    
    # zzh:极性化，阈值-0.5，0.5        
    thresh = 0.3
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
#sess.close()

