# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:07:35 2019

image processing functions

@author: dawnlh
"""
import numpy as np
from skimage.measure import compare_ssim


# MSE
# input:去噪图像、参考图像
# output：a float value
def myMSE(y_true,y_pred):
    return np.mean(np.square(y_pred - y_true))

 # MSE
# input:去噪图像、参考图像
# output：a float value   
def myPSNR(y_true,y_pred,max_v):
    return 10* np.log10(max_v*2 / (np.mean(np.square(y_pred - y_true))))
 

# 直接调用skimage的ssim
def mySSIM(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False, dynamic_range=None, **kwargs):
    return compare_ssim(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False, dynamic_range=None, **kwargs)



''' 未验证代码

'''

''' 废弃代码
# ssim
def mySSIM(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom
    
# non-reference SSIM
# input:含噪图像、去噪图像, 均为单通道二维图像
# output：a float value
def myNRSS(noisy_img, denoising_img):
#        noisy_img = noisy_img.reshape(noisy_img.size, order='C')
#        denoising_img = denoising_img.reshape(denoising_img.size, order='C')
    MNI = noisy_img-denoising_img
    _, N = compare_ssim(noisy_img, MNI,full=True)
    _, P = compare_ssim(noisy_img, denoising_img,full=True)
    N = N.reshape(N.size, order='C')
    P= P.reshape(P.size, order='C')
    nrss = np.corrcoef(P,N)[0,1]       
    return nrss
'''