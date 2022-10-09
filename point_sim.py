# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:15:12 2022

@author: lowes
"""
import numpy as np
import torch
import nibabel as nib
import cv2

class pointSimulator():
    def __init__(self,
                 shape = [96,128,128],
                 radius = [1,1,1],
                 max_num_points = 12
                 ):
        self.shape = shape
        self.radius = radius
        self.max_num_points = max_num_points
        
    def __call__(self, label):
        """
        1. find border of the label
        2. sample some random points, which are on the border
        3. add random noise in x and y direction, maybe between 5-50px (not in which slice it is)
        4. categorize points as in or outside the label
        5. render points in volume with a fixed radius
        """
        print(label.shape)
        print(label.nonzero(as_tuple=True)[0])
        
        n_points = torch.randint(2, self.max_num_points, (1,))
        
        nnz_slices = label.nonzero(as_tuple=True)[0]
        slice_idx = nnz_slices[torch.randint(len(nnz_slices), (n_points,))]
        
        print(slice_idx)
        
        print(label[slice_idx].shape)
        
        torch_result_erosion = 1 - torch.clamp(torch.nn.functional.conv2d(1 - label[slice_idx[0]].unsqueeze(0), torch.tensor([1,1,1]), padding=(1, 1)), 0, 1)
        
        print((label[slice_idx[0]]-torch_result_erosion.squeeze()).sum())
        
        points_vol = 0
        return points_vol
    
    
if __name__=="__main__":
    pointMaker = pointSimulator()
    lab_name = "../data/Decathlon/labelsTr/spleen_2.nii.gz"
    label = nib.load(lab_name).get_fdata()
    label = cv2.resize(label,dsize=(128,128))
    label = torch.from_numpy(label).permute(2,0,1)
    
    point_vol = pointMaker(label)

    