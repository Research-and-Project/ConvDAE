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
# setting 
SAVE_FLAG = 1   # flag of saving the outputs of the network's prediction
test_batch_size = 100
pic_size = [28,28]

# path
data_path = "./dataset/N_MNIST_pic/N_MNIST_pic_test.mat"
root_model_path = "model_data/"  # model's root dir
model_dir = "L-ConvDAE(sup)--08-11_14-45/" # model'saving dir
model_path = root_model_path + model_dir

# model
model_name = 'my_model' #model's name
model_ind = -1  #model's index

pred_res_path = './predict_res/'+model_dir  # dir of the prediction results
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
keep_prob = graph.get_tensor_by_name("inputs/Placeholder:0")  # for dropout
#outputs_ = graph.get_tensor_by_name("outputs/outputs_:0")
#outputs_ = graph.get_tensor_by_name("outputs/conv2d/Relu:0")  # act_fun = relu
outputs_ = graph.get_tensor_by_name("outputs/conv2d/Tanh:0") # act_fun = relu
cost = graph.get_tensor_by_name("loss/Mean:0")


# In[]:
# load data
pic_test_data = my_io.load_mat(data_path)
pic_test_x = pic_test_data['N_MNIST_pic_test'].astype('float32')
print('pic_test_x: ', pic_test_x.shape)
in_imgs = pic_test_x
#num_selected = 200
#test_idx = np.linspace(0,len(pic_test_x)-1,num_selected).astype('int32')
#in_imgs = pic_test_x[test_idx]
#gt_imgs = pic_test_y[test_idx]



# In[]:
# prediction
ind = 0
mean_cost = 0
time_cost = 0
reconstructed = np.zeros(in_imgs.shape, dtype='float32')
for batch_x, _ in my_io.batch_iter(test_batch_size,in_imgs, in_imgs, shuffle=False):
    x = batch_x.reshape((-1, *pic_size, 1))
    feed_dict = {inputs_: x, keep_prob:1.0, mask_prob:0.0} #for dropout
#    feed_dict = {inputs_: x, targets_: y}  #for non dropout
    
    time1 = time()
    res_imgs = sess.run(outputs_, feed_dict=feed_dict)
    time2 = time()
    time_cost += (time2 - time1)
    res_imgs = np.squeeze(res_imgs)
    reconstructed[ind*test_batch_size:(ind+1)*test_batch_size] = res_imgs
    ind += 1
time_cost = time_cost/len(in_imgs)
print('\nmean time cost(ms):%f\n'%(time_cost*1e3))


# In[]:
# save the prediction results
if SAVE_FLAG:
#    np.save(pred_res_path+'pred_res',reconstructed)   # save pics in the format of .npy
#    print('\nreconstruction data saved to : \n',pred_res_path+'pred_res.npy' )    
    for i in range(len(reconstructed)):
        plt_img.imsave(pred_res_path+str(i)+'.png', reconstructed[i], cmap=plt.cm.gray)
    print('\nreconstruction data saved to : \n',pred_res_path)
    


# In[]:
# illustrate the results
start = 0
end = len(reconstructed)-1
idx = np.linspace(start, end, 10).astype('int32')  # show 10 results at equal intervals

in_images = in_imgs[idx]
recon_images = reconstructed[idx]

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([in_images, recon_images], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((*pic_size)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)


# In[24]:
# release
sess.close()





