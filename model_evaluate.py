# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:48:39 2023

@author: Yixiao
"""
# -*- coding: utf-8 -*-
import os
from scipy.io import loadmat
from argparse import Namespace
import sys
from findSurfaceIdx import selectThresh
from genROI import genROI, plotROI_v0, genModelInput3_wref

sys.path.append('C:/Users/Yixiao/Desktop/AR-PAM_YL/CNN classification/tools')
from guidedBackPropPAM import computeGBP_Bscan3
from plotUSPAscan import plotActivation3v2, plotUSPAscan_S


import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
from PIL import Image as PImage

#%% LOAD SCAN AND NORMAL REFERENCE
root_path = os.getcwd()
im_dir = os.path.join(root_path,'example_scan.mat')

usparef_path = os.path.join(root_path,'example_normal_reference.mat')
USPAref = PImage.open(usparef_path).convert('L')

im_width = 1000
params = Namespace(dr_US = 3, dr_PA = 3.5, aspectRatio = 5/9, perc_overlap = 0.75,
                 cancer_threshold = 0.2, scar_threshold = 0.25, artifact_threshold = 0.1)

#%% LOAD MODEL
USPAnet_wts_path = os.path.join(root_path,'uspam_densenet')
USPAnet_wts = torch.load(USPAnet_wts_path,map_location = torch.device('cpu'))
USPAnet = models.DenseNet(12,[4,8,6],64)
USPAnet.features.conv0 = torch.nn.Conv2d(5,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
num_features = USPAnet.classifier.in_features
USPAnet.classifier = torch.nn.Linear(num_features,3)
USPAnet.load_state_dict(USPAnet_wts)
USPAnet.eval()

#%% PREVIEW IMAGE AND MAKE CLASSIFICATIONs
D = loadmat(im_dir)
im_label = {'cancer':np.array([]), 'artifact':np.array([]), 'scar':np.array([])}
plotUSPAscan_S(D,0.75)

img = D['US_polar1']
surface_idx = selectThresh(img)
    
PAim = D['PA_polar1']
for i in range(1000):
    surface_idx_temp = int(surface_idx[i])
    PAim[0:np.maximum(1,surface_idx_temp-20),i] = 0
    PAim[surface_idx_temp+150:640,i]=0
    
D['PA_polar1'] = PAim
    
ROI_info = genROI(D, surface_idx, im_label, params)
ROI_info['baseline image'] = np.float32(USPAref)
plotROI_v0(D, surface_idx, ROI_info)
US_ds,label_ds,USPA_ds = genModelInput3_wref('11',ROI_info)

print('>>>>>>>> USPA model predicting ... ')
inputsUSPA, labels_true = Variable(USPA_ds),Variable(label_ds)
outputsUSPA = USPAnet(inputsUSPA)
_, predsUSPA = torch.max(outputsUSPA.data, 1)


#%% GUIDED BACKPROP AND MODEL INTERPRETATION
r_normal_seq, r_cancer_seq, r_artifact_seq, normal_activation_final, cancer_activation_final, artifact_activation_final = computeGBP_Bscan3(USPA_ds, USPAnet, ROI_info, params)

cancer_activation_plot = cv2.bilateralFilter(cancer_activation_final.astype('float32'), 15, 80, 80)
normal_activation_plot = cv2.bilateralFilter(normal_activation_final.astype('float32'), 15, 80, 80)
artifact_activation_plot = cv2.bilateralFilter(artifact_activation_final.astype('float32'), 15, 80, 80)
plotActivation3v2(D, cancer_activation_plot, normal_activation_plot, artifact_activation_plot)

