# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:22:03 2022

@author: lowes
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np  
import pydicom as dcm
import os 
from pydicom.errors import InvalidDicomError
from vtk.util.numpy_support import numpy_to_vtk
import vtk 
import torchio as tio
from functions import (
    num_of_params,
    sample_data,
    loss_func,
    requires_grad,
    get_transform_tr,
    load_state_dict_loose,
    crop_CT
    )
from config import (
    get_args,
    update_args,
    )
import argparse
from model_unet_3d import UNet3D
import cv2
from dataset_loader_v2 import CT_Dataset
import torch
import torch.nn as nn 
import glob



if __name__ ==  '__main__':
    

    
    # image = tio.ScalarImage(os.path.join(root, "imagesTr/spleen_32.nii.gz"))
    # resize = tio.Resize([128,128,100])
    # resize_img = resize(image)
    
    pre_vol = np.load( '../data/preprocessed_Decathlon/imagesTr/spleen_2.npy') / 255.0*2.0 - 1.0
    gt_vol = np.load( '../data/preprocessed_Decathlon/labelsTr/spleen_2.npy')
    
    net_name = 'just_learn'
    
    path = os.path.join("../runs",net_name,"checkpoint","*.pt")
    nets = glob.glob(path)
    nets.sort()
    net_path = nets[-1]
    
    device = "cuda"
    
    device = "cuda"
    arg_name = ''.join(filter(lambda x: not x.isdigit(), net_name))
    #args = get_args(name=arg_name[:-1])
    args = get_args(name=arg_name)
    transform_tr = False
    ds_tr = CT_Dataset(mode="train",
                         data_path='../data',
                         transform=transform_tr,
                         reshape = args.training.reshape,
                         reshape_mode = "padding",#args.training.reshape_mode,
                         datasets = "preprocessed_Decathlon",#args.training.datasets,
                         interp_mode = args.training.interp_mode)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.training.batch, drop_last=True,num_workers=0)
    dl_tr_sampled = sample_data(dl_tr)
    img, label_gt = next(dl_tr_sampled)
    img = img.to(device, dtype=torch.float)
    label = label_gt.to(device, dtype=torch.float)
    
    """ for debugging dataloader
    for batch in dl_tr:
        img, label_gt = batch
        print(label_gt)
    """
    from functions import loss_func
    import time
    t = time.time()
    net = UNet3D(**vars(args.unet)).to(device)
    ckpt = torch.load(net_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(ckpt["net"])
    net.eval()
    
    with torch.no_grad():
        m = nn.Sigmoid()
        outpred = m(net(img))
    print("forward pass took", time.time()-t)
    
    pre_vol = pre_vol.transpose(2,0,1)
    vol_t = torch.from_numpy(pre_vol).to(device, dtype=torch.float)
    # vol_t = torch.flip(vol_t,[0])
    ss = nn.Sigmoid()
    print("Predicting")
    with torch.no_grad():
        seg = net(vol_t.unsqueeze(0).unsqueeze(0))
        print("did seg")
        sseg = ss(seg)
    
    
    slice_idx = label_gt[0,0,:,:,:].nonzero()[:,0].median()
    plt.subplot(2,3,1)
    plt.title("Input")
    plt.imshow((np.squeeze(img[1,:,slice_idx,:,:].cpu().detach())+1)/2)
    plt.subplot(2,3,2)
    plt.title("Prediction")
    plt.imshow((np.squeeze(outpred[1,:,slice_idx,:,:].cpu().detach())))
    plt.subplot(2,3,3)
    plt.title("Ground truth")
    plt.imshow((np.squeeze(label_gt[1,:,slice_idx,:,:].cpu().detach())))

    _,_,idx = gt_vol.nonzero()
    slice_idx = np.median(idx).astype(int)
    plt.subplot(2,3,4)
    plt.title("Input")
    plt.imshow((np.squeeze(vol_t[slice_idx,:,:].cpu().detach())+1)/2)
    plt.subplot(2,3,5)
    plt.title("Prediction")
    plt.imshow((np.squeeze(sseg[0,:,slice_idx,:,:].cpu().detach())))
    plt.subplot(2,3,6)
    plt.title("Ground truth")
    plt.imshow(gt_vol[:,:,slice_idx])
