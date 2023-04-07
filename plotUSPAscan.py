# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:04:50 2022

@author: Yixiao
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

#%% Plot B scans in Cartesian coordinates
def plotUSPAscan(D):
    USscanp = D['US_polar1']
    Nrows = np.size(USscanp,0)
    Ncols = np.size(USscanp,1)
    Nrows_corr = np.zeros([int(np.round(0.3*Nrows)),Ncols])
    USscanp = np.concatenate((Nrows_corr, USscanp))
    Nr = int(1.25*Nrows)
    USscan = cv2.warpPolar(cv2.rotate(USscanp,cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    ub_US = 0.5*np.max(USscan)
    
    
    PAscanp = D['PA_polar1']
    PAscanp = np.concatenate((Nrows_corr, PAscanp))
    PAscanp[int(np.round(0.3*Nrows))+550:Nrows+int(np.round(0.3*Nrows)),:] = 0
    PAscan = cv2.warpPolar(cv2.rotate(PAscanp,cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    ub_PA = 0.4*np.max(PAscan)
    
    USscan = cv2.rotate(USscan,cv2.ROTATE_90_COUNTERCLOCKWISE)
    PAscan = cv2.rotate(PAscan,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    fig1 = plt.figure(figsize=(24,10))
    plt.rcParams.update({'font.size': 24})
    ax1 = plt.subplot(1,3,1)
    ax1.imshow(USscan,cmap = 'gray', vmin = ub_PA/35, vmax = ub_US)
    ax1.tick_params(bottom=False)
    plt.title('US')
    ax1.set_yticks([])
    ax1.set_xticks([])
    
    ax2 = plt.subplot(1,3,2)
    ax2.imshow(PAscan,cmap = 'hot', vmin = ub_PA/50, vmax = 0.5*ub_PA)
    ax2.tick_params(bottom=False,left=False)
    plt.title('PA')
    ax2.set_yticks([])
    ax2.set_xticks([])
    
    ax3 = plt.subplot(1,3,3)
    ax3.imshow(USscan,cmap = 'gray', vmin = ub_US/50, vmax = ub_US,     alpha = 0.975)
    ax3.imshow(PAscan,cmap = 'hot',  vmin = ub_PA/50, vmax = 0.5*ub_PA, alpha = np.minimum(np.maximum(PAscan,0)*500,1))
    ax3.tick_params(bottom=False,left=False)
    plt.title('Overlay')
    ax3.set_yticks([])
    ax3.set_xticks([])
    
    plt.subplots_adjust(wspace=0.02,hspace=0.01)
    fig1.set_dpi(400)
    plt.show()
    return

#%%
def plotUSPAscan_S(D, PA_factor):
    USscanp = D['US_polar1']
    Nrows = np.size(USscanp,0)
    Ncols = np.size(USscanp,1)
    Nrows_corr = np.zeros([int(np.round(0.3*Nrows)),Ncols])
    USscanp = np.concatenate((Nrows_corr, USscanp))
    Nr = int(1.25*Nrows)
    USscan = cv2.warpPolar(cv2.rotate(USscanp,cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    ub_US = 0.4*np.max(USscan)
    
    
    PAscanp = D['PA_polar1']
    PAscanp = np.concatenate((Nrows_corr, PAscanp))
    PAscanp[int(np.round(0.3*Nrows))+550:Nrows+int(np.round(0.3*Nrows)),:] = 0
    PAscan = cv2.warpPolar(cv2.rotate(PAscanp,cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    ub_PA = 0.4*np.max(PAscan)
    
    USscan = cv2.rotate(USscan,cv2.ROTATE_90_COUNTERCLOCKWISE)
    PAscan = cv2.rotate(PAscan,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    fig1 = plt.figure(figsize=(24,10))
    plt.rcParams.update({'font.size': 24})
    ax1 = plt.gca()
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/10, vmax = 0.95*ub_US,     alpha = 0.975)
    plt.imshow(PAscan,cmap = 'hot',  vmin = ub_PA/50, vmax = PA_factor*ub_PA, alpha = np.minimum(np.maximum(PAscan,0)*500,1))
    plt.tick_params(bottom=False,left=False)
    ax1.set_yticks([])
    ax1.set_xticks([])
    
    fig1.set_dpi(400)
    plt.show()
    return

#%% Plot B scans in polar coordinates with labels
def plotUSPAScanLabel(D,im_label):
    USscan = D['US_polar1']
    Nz = np.size(USscan,0)
    Nx = np.size(USscan,1)
    USscan[0:35,:] = 0
    ub_US = 0.35*np.max(USscan)
    
    PAscan = D['PA_polar1']
    PAscan[0:35,:] = 0
    ub_PA = 0.35*np.max(PAscan)
    
    
    fig1 = plt.figure(figsize = (5,8))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/8, vmax = ub_US, alpha = 0.975)
    plt.imshow(PAscan,cmap = 'hot',  vmin = ub_PA/50, vmax = ub_PA, alpha = np.minimum(np.maximum(PAscan,0)*5000,1))
    plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    plt.title('US+PA')
    if np.any(im_label['cancer']):
        cancer_label = im_label['cancer']
        for i in range(int(cancer_label.size / 2)):
            xcoords = cancer_label[i,:]
            ax.add_patch( Rectangle((xcoords[0], 5), xcoords[1]-xcoords[0], Nz-10, fc='none', color ='crimson', linewidth = 3, linestyle="dotted") )
    if np.any(im_label['artifact']):
        artifact_label = im_label['artifact']
        for i in range(int(artifact_label.size / 2)):
            xcoords = artifact_label[i,:]
            ax.add_patch( Rectangle((xcoords[0], 5), xcoords[1]-xcoords[0], Nz-10, fc='none', color ='cyan', linewidth = 3, linestyle="dotted") )
    if np.any(im_label['scar']):
        scar_label = im_label['scar']
        for i in range(int(scar_label.size / 2)):
            xcoords = scar_label[i,:]
            ax.add_patch( Rectangle((xcoords[0], 5), xcoords[1]-xcoords[0], Nz-10, fc='none', color ='yellow', linewidth = 3, linestyle="dotted") )
    fig.set_dpi(400)
    plt.show()
    return

#%% Plot activation maps
def plotActivation(D,cancer_activation_final,normal_activation_final): 
    Normalalpha_factor = 1.05
    Canceralpha_factor = 2
    act_all = np.concatenate((cancer_activation_final,normal_activation_final),0)
    USscanp = D['US_polar1']
    Nrows = np.size(USscanp,0)
    Ncols = np.size(USscanp,1)
    Nrows_corr = np.zeros([int(np.round(0.3*Nrows)),Ncols])
    USscanp = np.concatenate((Nrows_corr, USscanp))
    Nr = int(1.25*Nrows)
    USscan = cv2.warpPolar(cv2.rotate(USscanp,cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    ub_US = 0.5*np.max(USscan)
    
    cancer_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, cancer_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    normal_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, normal_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    

    USscan = cv2.rotate(USscan,cv2.ROTATE_90_COUNTERCLOCKWISE)
    cancer_cart = cv2.rotate(cancer_cart,cv2.ROTATE_90_COUNTERCLOCKWISE)
    normal_cart = cv2.rotate(normal_cart,cv2.ROTATE_90_COUNTERCLOCKWISE)

    fig1 = plt.figure(figsize = (12,10))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/9, vmax = ub_US, alpha = 0.975)
    plt.imshow(normal_cart,vmin=np.percentile(act_all,50),vmax=np.percentile(act_all,99.95),cmap='hot',alpha = np.minimum(normal_cart*Normalalpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.title('Model response to normal region')
    plt.show()
    
    fig2 = plt.figure(figsize = (12,10))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = 0, vmax = ub_US, alpha = 0.95)
    plt.imshow(cancer_cart,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot', alpha = np.minimum(cancer_cart*Canceralpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.title('Model response to cancer region')
    plt.show()
    
    fig3 = plt.figure(figsize = (5,8))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(D['US_polar1'],cmap = 'gray', vmin = 0, vmax = ub_US, alpha = 0.95)
    plt.imshow(normal_activation_final,vmin=np.percentile(act_all,50),vmax=np.percentile(act_all,99.95),cmap='hot',alpha = np.minimum(normal_activation_final*Normalalpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.title('Model response to normal region')
    plt.show()
    
    fig4 = plt.figure(figsize = (5,8))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(D['US_polar1'],cmap = 'gray', vmin = 0, vmax = ub_US, alpha = 0.95)
    plt.imshow(cancer_activation_final,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot', alpha = np.minimum(cancer_activation_final*Canceralpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.title('Model response to cancer region')
    plt.show()
    return

#%% Plot activation maps version 2
def plotActivationv2(D,cancer_activation_final,normal_activation_final):
    hotcm = cm.get_cmap('hot',256)
    hotcm_data = hotcm(np.linspace(0, 1, 256))
    greencm_data = hotcm_data[:, [2, 1, 0, 3]]
    greencm = ListedColormap(greencm_data, name='GreenHot')
    
    Normalalpha_factor = 1.25
    Canceralpha_factor = 2.5
    act_all = np.concatenate((cancer_activation_final,normal_activation_final),0)
    USscanp = D['US_polar1']
    Nrows = np.size(USscanp,0)
    Ncols = np.size(USscanp,1)
    Nrows_corr = np.zeros([int(np.round(0.3*Nrows)),Ncols])
    USscanp = np.concatenate((Nrows_corr, USscanp))
    Nr = int(1.25*Nrows)
    USscan = cv2.warpPolar(cv2.rotate(USscanp,cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    ub_US = 0.35*np.max(USscan)
    
    cancer_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, cancer_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    normal_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, normal_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    

    fig1 = plt.figure(figsize = (12,10))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = 0, vmax = ub_US, alpha = 0.95)
    plt.imshow(normal_cart,vmin=np.percentile(act_all,50),vmax=np.percentile(act_all,99.95),cmap=greencm,alpha = np.minimum(normal_cart*Normalalpha_factor,1))
    plt.imshow(cancer_cart,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot', alpha = np.minimum(cancer_cart*Canceralpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.show()
    
    fig3 = plt.figure(figsize = (5,8))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(D['US_polar1'],cmap = 'gray', vmin = 0, vmax = ub_US, alpha = 0.95)
    plt.imshow(normal_activation_final,vmin=np.percentile(act_all,50),vmax=np.percentile(act_all,99.95),cmap=greencm,alpha = np.minimum(normal_activation_final*Normalalpha_factor,1))
    plt.imshow(cancer_activation_final,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot', alpha = np.minimum(cancer_activation_final*Canceralpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.show()
    
    return

#%% Plot 3 activation maps
def plotActivation3(D,cancer_activation_final,normal_activation_final, artifact_activation_final): 
    Normalalpha_factor = 1.25
    Canceralpha_factor = 2
    Artifactalpha_factor = 2.5
    act_all = np.concatenate((cancer_activation_final,normal_activation_final,artifact_activation_final),0)
    USscanp = D['US_polar1']
    Nrows = np.size(USscanp,0)
    Ncols = np.size(USscanp,1)
    Nrows_corr = np.zeros([int(np.round(0.3*Nrows)),Ncols])
    USscanp = np.concatenate((Nrows_corr, USscanp))
    Nr = int(1.25*Nrows)
    USscan = cv2.warpPolar(cv2.rotate(USscanp,cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    ub_US = 0.35*np.max(USscan)
    
    cancer_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, cancer_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    normal_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, normal_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    artifact_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, artifact_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    

    fig1 = plt.figure(figsize = (12,10))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/12.5, vmax = ub_US, alpha = 0.95)
    plt.imshow(normal_cart,vmin=np.percentile(act_all,50),vmax=np.percentile(act_all,99.95),cmap='hot',alpha = np.minimum(normal_cart*Normalalpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.title('Model response to normal region')
    plt.show()
    
    fig2 = plt.figure(figsize = (12,10))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/12.5, vmax = ub_US, alpha = 0.95)
    plt.imshow(cancer_cart,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot', alpha = np.minimum(cancer_cart*Canceralpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.title('Model response to cancer region')
    plt.show()
    
    fig3 = plt.figure(figsize = (5,8))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/12.5, vmax = ub_US, alpha = 0.95)
    plt.imshow(artifact_cart,vmin=0,vmax=np.percentile(act_all,99.999),cmap='hot',alpha = np.minimum(artifact_cart*Artifactalpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.title('Model response to artifact region')
    plt.show()
    return

#%%
def plotActivation3v2(D,cancer_activation_final,normal_activation_final, artifact_activation_final): 
    hotcm = cm.get_cmap('hot',256)
    hotcm_data = hotcm(np.linspace(0, 1, 256))
    greencm_data = hotcm_data[:, [2, 1, 0, 3]]
    greencm = ListedColormap(greencm_data, name='GreenHot')
    yellowcm_data = hotcm_data[:, [1, 0, 2, 3]]
    yellowcm_data[:,0] = yellowcm_data[:,0]
    yellowcm = ListedColormap(yellowcm_data, name='YellowHot')
    
    Normalalpha_factor = 1.0
    Canceralpha_factor = 1.5
    Artifactalpha_factor = 2
    act_all = np.concatenate((cancer_activation_final,normal_activation_final,artifact_activation_final),0)
    USscanp = D['US_polar1']
    Nrows = np.size(USscanp,0)
    Ncols = np.size(USscanp,1)
    Nrows_corr = np.zeros([int(np.round(0.3*Nrows)),Ncols])
    USscanp = np.concatenate((Nrows_corr, USscanp))
    Nr = int(1.25*Nrows)
    USscan = cv2.warpPolar(cv2.rotate(USscanp,cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    ub_US = 0.35*np.max(USscan)
    
    cancer_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, cancer_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    normal_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, normal_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    
    artifact_cart = cv2.warpPolar(cv2.rotate(np.concatenate((Nrows_corr, artifact_activation_final)),cv2.ROTATE_90_COUNTERCLOCKWISE), dsize=[2*Nr+1,2*Nr+1],center=[Nr,Nr],maxRadius=Nr,flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR)    

    USscan = cv2.rotate(USscan,cv2.ROTATE_90_COUNTERCLOCKWISE)
    cancer_cart = cv2.rotate(cancer_cart,cv2.ROTATE_90_COUNTERCLOCKWISE)
    normal_cart = cv2.rotate(normal_cart,cv2.ROTATE_90_COUNTERCLOCKWISE)
    artifact_cart = cv2.rotate(artifact_cart,cv2.ROTATE_90_COUNTERCLOCKWISE)

    fig1 = plt.figure(figsize = (12,10))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/12.5, vmax = ub_US, alpha = 0.95)
    plt.imshow(normal_cart,vmin=np.percentile(act_all,50),vmax=np.percentile(act_all,99.95),cmap=yellowcm,alpha = np.minimum(normal_cart*Normalalpha_factor,1))
    #plt.imshow(cancer_cart,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot', alpha = np.minimum(cancer_cart*Canceralpha_factor,1))
    #plt.imshow(artifact_cart,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot',alpha = np.minimum(artifact_cart*Artifactalpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.show()
    
    fig1 = plt.figure(figsize = (12,10))
    fig,ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    plt.imshow(USscan,cmap = 'gray', vmin = ub_US/12.5, vmax = ub_US, alpha = 0.95)
    #plt.imshow(normal_cart,vmin=np.percentile(act_all,50),vmax=np.percentile(act_all,99.95),cmap=yellowcm,alpha = np.minimum(normal_cart*Normalalpha_factor,1))
    plt.imshow(cancer_cart,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot', alpha = np.minimum(cancer_cart*Canceralpha_factor,1))
    #plt.imshow(artifact_cart,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot',alpha = np.minimum(artifact_cart*Artifactalpha_factor,1))
    ax.axis('off')
    fig.set_dpi(400)
    plt.show()
    
    #fig3 = plt.figure(figsize = (5,8))
    #ig,ax = plt.subplots()
    #plt.rcParams.update({'font.size': 12})
    #plt.imshow(D['US_polar1'],cmap = 'gray', vmin = ub_US/12.5, vmax = ub_US, alpha = 0.95)
    #plt.imshow(normal_activation_final,vmin=np.percentile(act_all,50),vmax=np.percentile(act_all,99.95),cmap=greencm,alpha = np.minimum(normal_activation_final*Normalalpha_factor,1))
    #plt.imshow(cancer_activation_final,vmin=0,vmax=np.percentile(act_all,99.95),cmap='hot', alpha = np.minimum(cancer_activation_final*Canceralpha_factor,1))
    #plt.imshow(artifact_activation_final,vmin=0,vmax=np.percentile(act_all,99.95),cmap=yellowcm, alpha = np.minimum(artifact_activation_final*Artifactalpha_factor,1))
    #ax.axis('off')
    #fig.set_dpi(400)
    #plt.show()
    
    return


