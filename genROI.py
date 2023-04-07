# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:30:27 2022

@author: Yixiao
"""

import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import torch
from torchvision import datasets, models, transforms

#%%
def flattenImage(img_in,surface_fitted):
    zmin = int(np.min(surface_fitted))
    zmax = int(np.max(surface_fitted))
    DeltaZ =zmax - zmin
    Nz = np.size(img_in,0)
    Nx = np.size(img_in,1)
    img_flat = np.zeros((Nz+DeltaZ,Nx))
    deltaz = np.zeros((Nx,))
    for xi in range(Nx):
        dz = surface_fitted[xi] - zmin
        padTop = int(DeltaZ - dz)
        padBottom = int(DeltaZ - padTop)
        aline_expand = np.concatenate((np.zeros((padTop,)),img_in[:,xi],np.zeros((padBottom,))))
        img_flat[:,xi] = aline_expand
        deltaz[xi] = padTop
    return img_flat, deltaz, DeltaZ

#%%
def wrapImage(img,Noverlap):
    Nx = np.size(img,1)
    wrap_right = img[:,Nx-Noverlap:Nx]
    wrap_left = img[:,0:Noverlap]
    img2 = np.concatenate((wrap_right,img,wrap_left),axis=1)
    return img2


#%% function that generates ROIs from original image
def genROI(D,surface_idx,im_label, params):
    img_USpolar = D['US_polar1']
    img_PApolar = D['PA_polar1']
    #Nz = np.size(img_USpolar,0)
    Nx = np.size(img_USpolar,1)
    for i in range(Nx):
        img_PApolar[0:int(surface_idx[i])-5,i] = 0
        
    # Flatten image
    US_flat , deltaz, DeltaZ = flattenImage(img_USpolar,surface_idx)
    PA_flat , _ , _ = flattenImage(img_PApolar,surface_idx)
    w_wrap = int(np.floor(Nx/8))
    US_expand = wrapImage(img_USpolar,w_wrap)
    PA_expand = wrapImage(img_PApolar,w_wrap)
    US_flat_expand = wrapImage(US_flat,w_wrap)
    PA_flat_expand = wrapImage(PA_flat,w_wrap)
    
    # Find bounding boxes and their labels
    w_roi = np.round(Nx/6)
    h_roi = np.round(w_roi/params.aspectRatio)+1
    perc_lumen_space = 0.08
    z_avg = int(np.mean(surface_idx + deltaz))
    deltaz = np.concatenate((deltaz[Nx-w_wrap:Nx],deltaz,deltaz[0:w_wrap]))
    
    im_label_expand = copy.deepcopy(im_label)
    if np.any(im_label['cancer']):
        cancer_label = im_label['cancer']
        im_label_expand['cancer'] = cancer_label + w_wrap
        for i in range(int(cancer_label.size / 2)):
            xleft = cancer_label[i,0]; xright = cancer_label[i,1]
            if xleft <= w_wrap:
                xright = min(w_wrap, xleft + w_roi)
                im_label_expand['cancer'] = np.concatenate((im_label_expand['cancer'],np.expand_dims(np.array([xleft+Nx+w_wrap,xright+Nx+w_wrap]),axis=0)))
            elif xright > Nx - w_wrap:
                xleft = max(Nx - w_wrap,xleft)
                im_label_expand['cancer'] = np.concatenate((im_label_expand['cancer'],np.expand_dims(np.array([xleft-Nx+w_wrap,xright-Nx+w_wrap]),axis=0)))
                
    if np.any(im_label['artifact']):
        artifact_label = im_label['artifact']
        im_label_expand['artifact'] = artifact_label + w_wrap
        for i in range(int(artifact_label.size / 2)):
            xleft = artifact_label[i,0]; xright = artifact_label[i,1]
            if xleft <= w_wrap:
                xright = min(w_wrap, xleft + w_roi)
                im_label_expand['artifact'] = np.concatenate((im_label_expand['artifact'],np.expand_dims(np.array([xleft+Nx+w_wrap,xright+Nx+w_wrap]),axis=0)))
            elif xright > Nx - w_wrap:
                xleft = max(Nx - w_wrap,xleft)
                im_label_expand['artifact'] = np.concatenate((im_label_expand['artifact'],np.expand_dims(np.array([xleft-Nx+w_wrap,xright-Nx+w_wrap]),axis=0)))
    
    if np.any(im_label['scar']):
        scar_label = im_label['scar']
        im_label_expand['scar'] = scar_label + w_wrap
        for i in range(int(scar_label.size / 2)):
            xleft = scar_label[i,0]; xright = scar_label[i,1]
            if xleft <= w_wrap:
                xright = min(w_wrap, xleft + w_roi)
                im_label_expand['scar'] = np.concatenate((im_label_expand['scar'],np.expand_dims(np.array([xleft+Nx+w_wrap,xright+Nx+w_wrap]),axis=0)))
            elif xright > Nx - w_wrap:
                xleft = max(Nx - w_wrap,xleft)
                im_label_expand['scar'] = np.concatenate((im_label_expand['scar'],np.expand_dims(np.array([xleft-Nx+w_wrap,xright-Nx+w_wrap]),axis=0)))
    
    # starting y coordinate of ROIs in flattened image
    z0_roi = z_avg - int(perc_lumen_space*h_roi)
    if z0_roi + h_roi >= np.size(US_flat,0):
        z0_roi = np.size(US_flat,0) - h_roi - 1
    
    # starting x coordinates of ROIs in flattened image
    x_roi_left = []
    x_roi_left_temp = 4
    x_roi_left.append(x_roi_left_temp)
    while x_roi_left_temp + w_roi < np.size(US_expand,1):
        x_roi_left_temp += int((1-params.perc_overlap)*w_roi)
        x_roi_left.append(x_roi_left_temp)
        
    x_roi_left = np.array(x_roi_left)
    N_roi = len(x_roi_left)
    
    # Determine ROI labels: normal 0, cancer 1, artifact 2, scar 3
    roi_labels = np.zeros((N_roi,))
    for roi_i in range(N_roi):
        roi_interval = np.linspace(x_roi_left[roi_i],x_roi_left[roi_i]+w_roi,num = int(w_roi) + 1)
        roi_label_i = 0
        if np.any(im_label_expand['cancer']):
            cancer_label = im_label_expand['cancer']
            cancer_interval = np.array([])
            for i in range(int(cancer_label.size / 2)):
                xleft = cancer_label[i,0]; xright = cancer_label[i,1]
                cancer_i_interval = np.linspace(xleft,xright,num = int(xright-xleft)+1)
                cancer_interval = np.concatenate((cancer_interval,cancer_i_interval))
            cancer_intersection = np.intersect1d(roi_interval,cancer_interval)
            if np.any(cancer_intersection):
                L_cancer = len(cancer_intersection)
                ratio = L_cancer / w_roi
                if ratio > params.cancer_threshold:
                    roi_label_i = 1
                    
        if np.any(im_label_expand['artifact']):
           artifact_label = im_label_expand['artifact']
           artifact_interval = np.array([])
           for i in range(int(artifact_label.size / 2)):
               xleft = artifact_label[i,0]; xright = artifact_label[i,1]
               artifact_i_interval = np.linspace(xleft,xright,num = int(xright-xleft)+1)
               artifact_interval = np.concatenate((artifact_interval,artifact_i_interval))
           artifact_intersection = np.intersect1d(roi_interval,artifact_interval)
           if np.any(artifact_intersection):
               L_artifact = len(artifact_intersection)
               ratio = L_artifact / w_roi
               if ratio > params.artifact_threshold:
                   roi_label_i = 2
                   
        if np.any(im_label_expand['scar']):
           scar_label = im_label_expand['scar']
           scar_interval = np.array([])
           for i in range(int(scar_label.size / 2)):
               xleft = scar_label[i,0]; xright = scar_label[i,1]
               scar_i_interval = np.linspace(xleft,xright,num = int(xright-xleft)+1)
               scar_interval = np.concatenate((scar_interval,scar_i_interval))
           scar_intersection = np.intersect1d(roi_interval,scar_interval)
           if np.any(scar_intersection):
               L_scar = len(scar_intersection)
               ratio = L_scar / w_roi
               if ratio > params.scar_threshold:
                   roi_label_i = 3
                   
        roi_labels[roi_i] = roi_label_i
        ROI_info = {'z0':z0_roi, 'DZ': DeltaZ, 'x0':x_roi_left, 'labels':roi_labels, 'US_expand':US_expand, 'US_flat':US_flat_expand,
               'PA_expand':PA_expand, 'PA_flat':PA_flat_expand, 'deltaz':deltaz, 'ROI_dims':np.array([w_roi,h_roi])}
    return ROI_info

#%%
def plotROI_v0(D,surface_idx,ROI_info,Nplot=8):
    N_roi = len(ROI_info['labels'])
    z0 = ROI_info['z0']
    
    USscan = ROI_info['US_expand']
    ub_US = 0.35*np.max(USscan)
    
    PAscan = ROI_info['PA_expand']
    ub_PA = 0.35*np.max(PAscan)
            
    fig1 = plt.figure(figsize = (10,15))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/15, vmax = ub_US, alpha = 0.975)
    plt.imshow(PAscan,cmap = 'hot',  vmin = 0, vmax = ub_PA, alpha = np.minimum(np.maximum(PAscan,0)*500,1))
    plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    plt.title('US+PA')
    
    roiidx2plot = (np.round(np.linspace(3,N_roi-5,num=Nplot-1))).astype(int)
    
    # for roi_i in roiidx2plot:
    #     x0 = ROI_info['x0'][roi_i]
    #     xc = int(x0 + ROI_info['ROI_dims'][0]/2)
    #     dz_i = ROI_info['deltaz'][xc]
    #     z0i = z0 - dz_i
        
    #     roi_i_label = ROI_info['labels'][roi_i]
    #     roi_colorcode = 'green'
    #     if roi_i_label == 1:
    #         roi_colorcode = 'crimson'
    #     elif roi_i_label == 2:
    #         roi_colorcode = 'cyan'
    #     elif roi_i_label == 3:
    #         roi_colorcode = 'yellow'
            
    #     ax.add_patch(Rectangle((x0,z0i), ROI_info['ROI_dims'][0], ROI_info['ROI_dims'][1], fc = 'none', color = roi_colorcode, linewidth = 3, linestyle = 'dotted'))
    fig.set_dpi(400)
    plt.show()
    return


#%% Plot ROIs on flattened scan
def plotROI_vf(D,surface_idx,ROI_info,Nplot=8):
    N_roi = len(ROI_info['labels'])
    z0 = ROI_info['z0']
    z0_plot = int(z0/2)
    
    USscan = ROI_info['US_flat']
    Nz = np.size(USscan,0)
    zf_plot = min(Nz,int(z0+ROI_info['ROI_dims'][1]+100))
    
    USscan = USscan[z0_plot:zf_plot,:]
    ub_US = 0.5*np.max(USscan)
    
    PAscan = ROI_info['PA_flat']
    PAscan = PAscan[z0_plot:zf_plot,:]
    ub_PA = 0.35*np.max(PAscan)
    
    z0 -= z0_plot
        
    fig1 = plt.figure(figsize = (10,15))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/15, vmax = ub_US, alpha = 0.975)
    plt.imshow(PAscan,cmap = 'hot',  vmin = ub_PA/50, vmax = ub_PA, alpha = np.minimum(np.maximum(PAscan,0)*500,1))
    plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    plt.title('US+PA')
    
    roiidx2plot = (np.round(np.linspace(3,N_roi-5,num=Nplot-1))).astype(int)
    
    
    # for roi_i in roiidx2plot:
    #     x0 = ROI_info['x0'][roi_i]
    #     roi_i_label = ROI_info['labels'][roi_i]
    #     roi_colorcode = 'green'
    #     if roi_i_label == 1:
    #         roi_colorcode = 'crimson'
    #     elif roi_i_label == 2:
    #         roi_colorcode = 'cyan'
    #     elif roi_i_label == 3:
    #         roi_colorcode = 'yellow'
        
    #     ax.add_patch(Rectangle((x0,z0), ROI_info['ROI_dims'][0], ROI_info['ROI_dims'][1], fc = 'none', color = roi_colorcode, linewidth = 3, linestyle = 'dotted'))
    
    fig.set_dpi(400)
    plt.show()
    return

#%%
def genModelInput(imqual_in,ROI_info):
    US_ds = []
    USPA_ds = []
    label_ds = []
    N_roi = len(ROI_info['labels'])-1
    z0 = int(ROI_info['z0'])
    USscan = copy.deepcopy(ROI_info['US_flat'])
    ub_US = 0.4*np.max(USscan)
    USscan[USscan>ub_US]=ub_US
    USscan[USscan<ub_US/10] = 0
    
    w_roi = int(ROI_info['ROI_dims'][0])
    h_roi = int(ROI_info['ROI_dims'][1])+1
    transformUS = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(0.5,0.5),
                                      transforms.Resize([128,64])])
    for roi_idx in range(N_roi):
        x0_i = int(ROI_info['x0'][roi_idx])
        roi_i = USscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
        roi_i = np.array(roi_i / ub_US * 255, dtype = np.uint8)
        roi_i = transformUS(roi_i)
        US_ds.append(roi_i)
        label_ds.append(ROI_info['labels'][roi_idx])
      
    US_ds = torch.stack(US_ds)
    label_ds = np.array(label_ds)
    label_ds = torch.tensor(label_ds)
    label_ds = label_ds.type(torch.LongTensor)
        
    if imqual_in == '11':
        PAscan = copy.deepcopy(ROI_info['PA_flat'])
        ub_PA = 0.35*np.max(PAscan)
        PAscan[PAscan>ub_PA]=ub_PA
        PAscan[PAscan<ub_PA/15] = 0
        
        transformUSPA = transforms.Compose([transforms.Normalize([0.115,0.5,0.35],[0.275,0.35,0.43]),
                                            transforms.Resize([128,64])])
        
        for roi_idx in range(N_roi):
            x0_i = int(ROI_info['x0'][roi_idx])
            USroi_i = USscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
            USroi_i = np.float32(2.5*np.array(USroi_i / ub_US))
            PAroi_i = PAscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
            PAroi_i[91:h_roi,:]=0
            PAroi_i = np.float32(5*np.array(PAroi_i / ub_PA))
            deriv_i = np.absolute(cv2.Sobel(USroi_i,cv2.CV_64F,0,1,ksize=5))
            deriv_i = np.float32(deriv_i / np.max(deriv_i))
            derivroi_i = np.float32(2.5*deriv_i + PAroi_i)
            roi_i = torch.tensor(np.stack([PAroi_i,USroi_i,derivroi_i],axis=0))
            roi_i = transformUSPA(roi_i)
            USPA_ds.append(roi_i)
            
        USPA_ds = torch.stack(USPA_ds)
    return US_ds, label_ds, USPA_ds

#%% get model inputs for 3 inputs
def genModelInput3(imqual_in,ROI_info):
    US_ds = []
    USPA_ds = []
    label_ds = []
    N_roi = len(ROI_info['labels'])-1
    z0 = int(ROI_info['z0'])
    USscan = copy.deepcopy(ROI_info['US_flat'])
    
    USscan = cv2.medianBlur(USscan.astype('float32'), 3)
    ub_US = 0.45*np.max(USscan)
    USscan[USscan>ub_US]=ub_US
    USscan[USscan<ub_US/10] = 0
    
    w_roi = int(ROI_info['ROI_dims'][0])
    h_roi = int(ROI_info['ROI_dims'][1])
    transformUS = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(0.5,0.5),
                                      transforms.Resize([224,224])])
    for roi_idx in range(N_roi):
        x0_i = int(ROI_info['x0'][roi_idx])
        roi_i = USscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
        roi_i = np.array(roi_i / ub_US * 255, dtype = np.uint8)
        roi_i = transformUS(roi_i)
        US_ds.append(roi_i)
        label_ds.append(ROI_info['labels'][roi_idx])
      
    US_ds = torch.stack(US_ds)
    label_ds = np.array(label_ds)
    label_ds = torch.tensor(label_ds)
    label_ds = label_ds.type(torch.LongTensor)
        
    if imqual_in == '11':
        PAscan = copy.deepcopy(ROI_info['PA_flat'])
        PAscan = cv2.medianBlur(PAscan.astype('float32'), 3)
        ub_PA = 0.35*np.max(PAscan)
        PAscan[PAscan>ub_PA]=ub_PA
        PAscan[PAscan<ub_PA/15] = 0
        
        transformUSPA = transforms.Compose([transforms.Normalize([0.125,0.5,0.35],[0.275,0.35,0.43]),
                                            transforms.Resize([128,64])])
        
        for roi_idx in range(N_roi):
            x0_i = int(ROI_info['x0'][roi_idx])
            USroi_i = USscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
            USroi_i = np.float32(2.5*np.array(USroi_i / ub_US))
            PAroi_i = PAscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
            PAroi_i[51:h_roi,:]=0

            PAroi_i = np.float32(4.5*np.array(PAroi_i / ub_PA))
            deriv_i = np.absolute(cv2.Sobel(USroi_i,cv2.CV_64F,0,1,ksize=5))
            deriv_i = np.float32(deriv_i / np.max(deriv_i))
            derivroi_i = np.float32(2.5*deriv_i + PAroi_i)
            roi_i = torch.tensor(np.stack([PAroi_i,USroi_i,derivroi_i],axis=0))
            roi_i = transformUSPA(roi_i)
            USPA_ds.append(roi_i)
            
        USPA_ds = torch.stack(USPA_ds)
    return US_ds, label_ds, USPA_ds

#%%
def genModelInput3_wref(imqual_in,ROI_info):
    US_ds = []
    USPA_ds = []
    label_ds = []
    N_roi = len(ROI_info['labels'])-1
    z0 = int(ROI_info['z0'])
    imbaseline = (ROI_info['baseline image']) / 255
    usbaseline = (ROI_info['baseline US']) / 255
    
    USfactor = 2.5
    PAfactor = 1.1
    USbaseline_uspa = USfactor*imbaseline[:,0:167]
    PAbaseline_uspa = PAfactor*imbaseline[:,167:334]
    
    #plt.imshow(PAbaseline_uspa)
    #plt.colorbar()
    
    USscan = copy.deepcopy(ROI_info['US_flat'])
    USscan = cv2.medianBlur(USscan.astype('float32'), 3)
    ub_US = 0.45*np.max(USscan)
    USscan[USscan > ub_US]=ub_US
    USscan[USscan < ub_US/12] = 0
    
    w_roi = int(ROI_info['ROI_dims'][0])
    h_roi = int(ROI_info['ROI_dims'][1])
    transformUS = transforms.Compose([transforms.Normalize((0.525,0.5),(0.5,0.5)),
                                      transforms.Resize([128,64])])
    
    for roi_idx in range(N_roi):
        x0_i = int(ROI_info['x0'][roi_idx])
        roi_i = USscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
        roi_i = np.array(roi_i / ub_US, dtype = np.float32)
        roi_i = np.stack([roi_i,usbaseline],axis=0)
        roi_i = transformUS(torch.Tensor(roi_i))
        US_ds.append(roi_i)
        label_ds.append(ROI_info['labels'][roi_idx])
      
    US_ds = torch.stack(US_ds)
    label_ds = np.array(label_ds)
    label_ds = torch.tensor(label_ds)
    label_ds = label_ds.type(torch.LongTensor)
        
    if imqual_in == '11':
        PAscan = copy.deepcopy(ROI_info['PA_flat'])
        PAscan = cv2.medianBlur(PAscan.astype('float32'), 3)
        ub_PA = 0.375*np.max(PAscan)
        PAscan[PAscan>ub_PA]=ub_PA
        PAscan[PAscan<ub_PA/15] = 0
        
        transformUSPA = transforms.Compose([transforms.Normalize([0.25,0.5,0.35,0.25,0.5],[0.25,0.5,0.4,0.25,0.5]),
                                            transforms.Resize([128,64])])
        
        for roi_idx in range(N_roi):
            x0_i = int(ROI_info['x0'][roi_idx])
            USroi_i = USscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
            USroi_i = np.float32(USfactor*np.array(USroi_i / ub_US))
            PAroi_i = PAscan[z0:z0+h_roi, x0_i:x0_i+w_roi]
            #PAroi_i[55:h_roi,:]=0
            PAroi_i = np.float32(PAfactor*np.array(PAroi_i / ub_PA))
            deriv_i = np.absolute(cv2.Sobel(USroi_i,cv2.CV_64F,0,1,ksize=5))
            deriv_i = np.float32(deriv_i / np.max(deriv_i))
            derivroi_i = np.float32(0.5*deriv_i + 1*PAroi_i)
          
            
            roi_i = torch.tensor(np.stack([PAroi_i,USroi_i,derivroi_i,PAbaseline_uspa,USbaseline_uspa],axis=0))
            roi_i = transformUSPA(roi_i)
            USPA_ds.append(roi_i)
            
        USPA_ds = torch.stack(USPA_ds)
    return US_ds, label_ds, USPA_ds