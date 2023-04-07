# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:50:24 2022

@author: Yixiao
"""

import numpy as np
from scipy import signal as sp
from sklearn.preprocessing import normalize
import cv2
import matplotlib.pyplot as plt 


def findSurfaceIdx(img, perc_thresh, zmin=120):
    Nz = np.size(img,0)
    Nx = np.size(img,1)
    img[0:zmin,:] = 0
    img_blur = sp.medfilt2d(img)
    img_blur2 = cv2.bilateralFilter(img_blur.astype(np.float32), 15, 80, 80)
    k  = np.array([ 0.004711,  0.069321,  0.245410,  0.361117,  0.245410,  0.069321,  0.004711]);
    d  = np.array([ 0.018708,  0.125376,  0.193091,  0.000000, -0.193091, -0.125376, -0.018708]);
    kern_yderiv = np.transpose(np.expand_dims(k,axis=1)*np.expand_dims(d,axis=0))
    im_yderiv = np.absolute(sp.convolve2d(img_blur2, kern_yderiv, 'same'))
    ub_yderiv = 0.125*np.max(im_yderiv)
    im_yderiv[im_yderiv>ub_yderiv] = ub_yderiv
    im_yderiv = normalize(im_yderiv,axis=0,norm='l1')
    im_yderiv = (im_yderiv-np.min(im_yderiv))/np.max(im_yderiv)
    im_yderiv[im_yderiv < perc_thresh] = 0
    
    surface_idx = np.zeros((Nx,))
    for xi in range(Nx):
        zi = np.squeeze(np.array(np.nonzero(im_yderiv[:,xi])))
        zi_list = np.argwhere(zi>zmin)
        if np.any(zi_list):
            surface_idx[xi] = zi[zi_list[0]]
        else:
            if xi == 0:
                surface_idx[xi] = 50
            else:
                surface_idx[xi] = surface_idx[xi-1]
                
    b,a = sp.butter(5, 4.667, fs=Nx, btype='low', analog=False)
    surface_idx_smooth = sp.savgol_filter(surface_idx,99,2,mode='mirror')
    surface_temp = sp.filtfilt(b,a,surface_idx_smooth)
    surface_idx_fitted = np.concatenate((surface_idx_smooth[0:50],surface_temp[50:Nx-50],surface_idx_smooth[Nx-50:Nx]))
    surface_idx_fitted = sp.savgol_filter(surface_idx_fitted,99,2,mode='interp')
    surface_idx_fitted[surface_idx_fitted > Nz-1] = Nz-1
    return surface_idx_fitted

def selectThresh(img):
    Nx = np.size(img,1)
    ub_img = 0.35*np.max(img)
    
    fig1 = plt.figure(figsize = (10,14))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12, 'font.weight':'bold'})
    plt.imshow(img, cmap = 'gray', vmin = 0, vmax = ub_img)
    
    thresh_choice = [0.05,0.1,0.15,0.2]
    for perc_thresh in thresh_choice:
        surface_idx = findSurfaceIdx(img, perc_thresh)
        plt.plot(np.arange(Nx),surface_idx,label=str(perc_thresh),lw=3,linestyle='dotted')
    
    plt.xlim([0,Nx+375])
    plt.legend(loc='right')
    plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.show()
    thresh_f = float(input('Choose gradient threshold: '))
    surface_idx_f = findSurfaceIdx(img, thresh_f)
    return surface_idx_f