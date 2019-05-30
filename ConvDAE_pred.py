'''
【累积重构-卷积去噪自编码 预测器】
使用说明：
修改内容：模型路径，模型目录； in_data_flag; save_flag; in_imgs gt_imgs, keep_prob定义和feed_dict, outputs (根据act_out_fun)

注意事项：
不同模型的outputs_定义可能不同，但是有时候没有报错，只是结果异常，可以创建模型后确认下outputs_定义
'''
# In[45]:


#导入基本模块
import numpy as np
import tensorflow as tf
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
from my_tf_lib import my_io
from my_imgproc_lib import my_img_evaluation as my_evl
import os

# In[]
#图设置
tf.reset_default_graph()  


# In[]:
# 路径参数

#参数  
model_ind = -1 #导入模型的序号
save_flag = 0 #保存预测结果的标志
positive_polarity = False
test_batch_size = 100

#路径
path1 = "N_MNIST_pic/N_MNIST_pic_train.mat"
path2 = "N_MNIST_pic/N_MNIST_pic_test.mat"

root_model_path = "model_data/"  #模型根目录
model_dir = "ConvDAE_unsup--05-15_16-00/" #模型保存目录
model_path = root_model_path + model_dir
model_name = 'my_model' #模型名称
model_ind = -1 #模型序号

pred_res_path = 'predict_res/'+model_dir #测试结果保存路径
if not os.path.isdir(pred_res_path) and save_flag:
   os.makedirs(pred_res_path)

# In[]:
# 加载数据
pic_test_data = my_io.load_mat(path2)
pic_test_x = pic_test_data['N_MNIST_pic_test'].astype('float32')
pic_test_y = pic_test_data['N_MNIST_pic_test_gt'].astype('float32')
print('pic_test_x: ', pic_test_x.shape, '\tpic_test_y: ', pic_test_y.shape)
in_imgs = pic_test_x
gt_imgs = pic_test_y


# In[]:
# 数据打印测试
#for k in range(5):
#    plt.subplot(2,5,k+1)
#    plt.imshow(test_x[k])
#    plt.title('test_x_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])        
#    plt.subplot(2,5,k+6)
#    plt.imshow(test_y[k])
#    plt.title('test_y_%d'%(k+1))
#    plt.xticks([])
#    plt.yticks([])


# In[]
#模型加载   
sess = tf.Session()
sess.run(tf.global_variables_initializer())

restorer = tf.train.import_meta_graph(model_path+model_name+'.meta')

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt:
    ckpt_states = ckpt.all_model_checkpoint_paths
    restorer.restore(sess, ckpt_states[model_ind])

graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name("inputs/inputs_:0")
targets_ = graph.get_tensor_by_name("inputs/targets_:0")
keep_prob = graph.get_tensor_by_name("inputs/Placeholder:0")  #for dropout
#outputs_ = graph.get_tensor_by_name("outputs/outputs_:0")
outputs_ = graph.get_tensor_by_name("outputs/outputs_:0") #for tanh act_out_fun
#outputs_ = graph.get_tensor_by_name("outputs/conv2d/Tanh:0") #for relu  act_out_fun
cost = graph.get_tensor_by_name("loss/Mean:0")


# In[]:
# 预测
# 计算测试集上全部样本平均预测误差，平均耗时和重构结果
ind = 0
mean_cost = 0
time_cost = 0
reconstructed = np.zeros(in_imgs.shape, dtype='float32')
for batch_x, batch_y in my_io.batch_iter(test_batch_size,in_imgs, gt_imgs, shuffle=False):
    x = batch_x.reshape((-1, 28, 28, 1))
    y = batch_y.reshape((-1, 28, 28, 1))
    feed_dict = {inputs_: x, targets_: y, keep_prob:1.0} #for dropout
#    feed_dict = {inputs_: x, targets_: y}  #for non dropout
    
    time1 = time()
    res_imgs = sess.run(outputs_, feed_dict=feed_dict)
    time2 = time()
    time_cost += (time2 - time1)
    res_imgs = np.squeeze(res_imgs)
    reconstructed[ind*test_batch_size:(ind+1)*test_batch_size] = res_imgs
    
    cost_v = sess.run(cost, feed_dict=feed_dict)
    mean_cost += cost_v*len(x)
    ind += 1
mean_cost = mean_cost/len(in_imgs)
time_cost = time_cost/len(in_imgs)
print('\ncurrent mean cost (mse):%f, mean time cost(ms):%f'%(mean_cost, time_cost*1e3))



# 保存预测结果数据  
if save_flag:
    np.save(pred_res_path+'pred_res',reconstructed) 
    print('\nreconstruction data saved to : \n',pred_res_path+'pred_res.npy' )

# IAQ: image quality assessment
#noisy_MSE = my_evl.myMSE(in_imgs, gt_imgs)
#noisy_PSNR = my_evl.myPSNR(in_imgs, gt_imgs,2)
#noisy_SSIM = my_evl.mySSIM(in_imgs, gt_imgs)
#
#denoise_MSE = my_evl.myMSE(reconstructed, gt_imgs) #这个mse和上面的网络计算的mse很不一样的话，是因为上面进行了极性化
#denoise_PSNR = my_evl.myPSNR(reconstructed, gt_imgs,2)
#denoise_SSIM = my_evl.mySSIM(reconstructed, gt_imgs)
#
#print('\nIQA before denoising:\nMSE:%f \nPSNR:%fdB \nSSIM:%f'%(noisy_MSE, noisy_PSNR, noisy_SSIM))
#print('\nIQA after denoising:\nMSE:%f \nPSNR:%fdB \nSSIM:%f'%(denoise_MSE, denoise_PSNR, denoise_SSIM))


start = 250
in_images = in_imgs[start:10000:1000]
recon_images = reconstructed[start:10000:1000]
gt_images = gt_imgs[start:10000:1000]


fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([in_images, recon_images, gt_images], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)


# In[24]:
#sess.close()


# In[ ]:




