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



if __name__ ==  '__main__':
    

    
    # image = tio.ScalarImage(os.path.join(root, "imagesTr/spleen_32.nii.gz"))
    # resize = tio.Resize([128,128,100])
    # resize_img = resize(image)
    
    
    
    device = "cuda"
    name = os.path.join("../runs/default/checkpoint","006000.pt")
    net_name = 'default'
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
    with torch.no_grad():
        net = UNet3D(**vars(args.unet)).to(device)
        ckpt = torch.load(name, map_location=lambda storage, loc: storage)
        net.load_state_dict(ckpt["net"])
        net.eval()
        m = nn.Sigmoid()
        outpred = m(net(img))
    print("forward pass took", time.time()-t)
    
    slice_idx = 30
    plt.subplot(1,3,1)
    plt.title("Input")
    plt.imshow((np.squeeze(img[0,:,slice_idx,:,:].cpu().detach())+1)/2)
    plt.subplot(1,3,2)
    plt.title("Prediction")
    plt.imshow((np.squeeze(outpred[0,:,slice_idx,:,:].cpu().detach())))
    plt.subplot(1,3,3)
    plt.title("Ground truth")
    plt.imshow((np.squeeze(label_gt[0,:,slice_idx,:,:].cpu().detach())))
