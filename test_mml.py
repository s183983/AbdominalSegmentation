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
    
    root = "../Decathlon"
    
    img = nib.load(os.path.join(root, "imagesTr/spleen_32.nii.gz"))
    data = img.get_fdata()
    
    # image = tio.ScalarImage(os.path.join(root, "imagesTr/spleen_32.nii.gz"))
    # resize = tio.Resize([128,128,100])
    # resize_img = resize(image)
    
    
    
    device = "cuda"
    name = os.path.join("001500.pt")
    net_name = 'default'
    device = "cuda"
    arg_name = ''.join(filter(lambda x: not x.isdigit(), net_name))
    #args = get_args(name=arg_name[:-1])
    args = get_args(name=arg_name)
    transform_tr = False
    ds_tr = CT_Dataset(mode="train",
                         data_path='../',
                         transform=transform_tr,
                         reshape = args.training.reshape,
                         reshape_mode = args.training.reshape_mode,
                         datasets = args.training.datasets,
                         interp_mode = args.training.interp_mode)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.training.batch, drop_last=True,num_workers=0)
    dl_tr_sampled = sample_data(dl_tr)
    img, label_gt = next(dl_tr_sampled)
    img = img.to(device, dtype=torch.float)
    
    """ for debugging dataloader
    for batch in dl_tr:
        img, label_gt = batch
        print(label_gt)
    """
    
    with torch.no_grad():
        net = UNet3D(**vars(args.unet)).to(device)
        ckpt = torch.load(name, map_location=lambda storage, loc: storage)
        net.load_state_dict(ckpt["net"])
        net.eval()
        m = nn.Sigmoid()
        outpred = m(net(img[0,:,:,:,:].unsqueeze(0)))
        
    plt.subplot(1,2,1)
    plt.title("Prediction")
    plt.imshow((np.squeeze(outpred[0,:,32,:,:].cpu().detach())))
    plt.subplot(1,2,2)
    plt.title("Ground truth")
    plt.imshow((np.squeeze(label_gt[0,:,32,:,:].cpu().detach())))
