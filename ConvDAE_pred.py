# -*- coding: utf-8 -*-
'''
【predictor for ConvDAE】
usage：
contents to modify：model path, model dir; SAVE_FLAG; 
in_imgs,  gt_imgs; keep_prob, feed_dict; outputs (according to act_out_fun)

notes：
Different models' "outputs_" could be different. Sometimes it won't report errors, 
but the result is not as expected.In this case, return to your model definition and confirm your "outputs_"

Author: zhihong (z_zhi_hong@163.com)
Date: 20190505
Modified: zhihong_20190809

'''
# In[]:
# import modules
import os
import numpy as np
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as plt_img
from util import my_io
#from util import my_img_evaluation as my_evl



# In[]:
# graph reset
tf.reset_default_graph()  


# In[]:
# parameters config
# setting 
SAVE_FLAG = 1   # flag of saving the outputs of the network's prediction
test_batch_size = 100
pic_size = [28,28]

# path
path1 = "./dataset/N_MNIST_pic/N_MNIST_pic_train.mat"
path2 = "./dataset/N_MNIST_pic/N_MNIST_pic_test.mat"
root_model_path = "model_data/"  # model's root dir
model_dir = "L-ConvDAE(unsup)--08-10_20-32/" # model'saving dir
model_path = root_model_path + model_dir

# model
model_name = 'my_model' #model's name
model_ind = -1  #model's index

pred_res_path = 'predict_res/'+model_dir  # dir of the prediction results
if not os.path.isdir(pred_res_path) and SAVE_FLAG:
   os.makedirs(pred_res_path)


# In[]
# load model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

restorer = tf.train.import_meta_graph(model_path+model_name+'.meta')

ckpt = tf.train.get_checkpoint_state(model_path)
if ckpt:
    ckpt_states = ckpt.all_model_checkpoint_paths
    restorer.restore(sess, ckpt_states[model_ind])
    
# load Ops and variables according to old model and your need
graph = tf.get_default_graph()
inputs_ = graph.get_tensor_by_name("inputs/inputs_:0")
mask_prob = graph.get_tensor_by_name("inputs/Placeholder_1:0")
targets_ = graph.get_tensor_by_name("inputs/targets_:0")
keep_prob = graph.get_tensor_by_name("inputs/Placeholder:0")  #for dropout
#outputs_ = graph.get_tensor_by_name("outputs/outputs_:0")
outputs_ = graph.get_tensor_by_name("outputs/conv2d/Tanh:0")
cost = graph.get_tensor_by_name("loss/Mean:0")


# In[]:
# load data
pic_test_data = my_io.load_mat(path2)
pic_test_x = pic_test_data['N_MNIST_pic_test'].astype('float32')
pic_test_y = pic_test_data['N_MNIST_pic_test_gt'].astype('float32')
print('pic_test_x: ', pic_test_x.shape, '\tpic_test_y: ', pic_test_y.shape)
#in_imgs = pic_test_x
#gt_imgs = pic_test_y
test_idx = np.linspace(0,len(pic_test_x)-1,200).astype('int32')
in_imgs = pic_test_x[test_idx]
gt_imgs = pic_test_y[test_idx]


# In[]:
# data disp
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


# In[]:
# prediction
ind = 0
mean_cost = 0
time_cost = 0
reconstructed = np.zeros(in_imgs.shape, dtype='float32')
for batch_x, batch_y in my_io.batch_iter(test_batch_size,in_imgs, gt_imgs, shuffle=False):
    x = batch_x.reshape((-1, 28, 28, 1))
    y = batch_y.reshape((-1, 28, 28, 1))
    feed_dict = {inputs_: x, targets_: y, keep_prob:1.0, mask_prob:0.0} #for dropout
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


# In[]:
# save the prediction results
if SAVE_FLAG:
#    np.save(pred_res_path+'pred_res',reconstructed)   # save pics in the format of .npy
#    print('\nreconstruction data saved to : \n',pred_res_path+'pred_res.npy' )    
    for i in range(len(reconstructed)):
        plt_img.imsave(pred_res_path+str(i)+'.png', reconstructed[i])
    print('\nreconstruction data saved to : \n',pred_res_path)
    

# IAQ: image quality assessment
#noisy_MSE = my_evl.myMSE(in_imgs, gt_imgs)
#noisy_PSNR = my_evl.myPSNR(in_imgs, gt_imgs,2)
#noisy_SSIM = my_evl.mySSIM(in_imgs, gt_imgs)
#
#denoise_MSE = my_evl.myMSE(reconstructed, gt_imgs)
#denoise_PSNR = my_evl.myPSNR(reconstructed, gt_imgs,2)
#denoise_SSIM = my_evl.mySSIM(reconstructed, gt_imgs)
#
#print('\nIQA before denoising:\nMSE:%f \nPSNR:%fdB \nSSIM:%f'%(noisy_MSE, noisy_PSNR, noisy_SSIM))
#print('\nIQA after denoising:\nMSE:%f \nPSNR:%fdB \nSSIM:%f'%(denoise_MSE, denoise_PSNR, denoise_SSIM))


# In[]:
# illustrate the results
start = 0
end = len(reconstructed)-1
idx = np.linspace(start, end, 10).astype('int32')  # show 10 results at equal intervals

in_images = in_imgs[idx]
recon_images = reconstructed[idx]
gt_images = gt_imgs[idx]


fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([in_images, recon_images, gt_images], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)


# In[24]:
#sess.close()
sess.close()





