# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:15:12 2022

@author: lowes
"""
import numpy as np
import torch
import nibabel as nib
import cv2
from scipy.ndimage import binary_erosion
import raster_geometry as rg

class pointSimulator():
    def __init__(self,
                 shape = [256,256,128],
                 radius = 1,
                 range_sampled_points = [2, 10],
                 border_mean = 10,
                 border_p = 0.5,
                 sphere_size = (5,2)
                 ):
        # [H,W,D]
        self.shape = shape 
        # for use in rg.sphere
        self.radius = radius
        self.sphere_size = sphere_size
        self.sphere = rg.sphere(sphere_size[0],sphere_size[1]).astype(int)
        self.sphere_nnz = np.array(self.sphere.nonzero())-sphere_size[0]//2
        self.range_sampled_points = range_sampled_points
        self.border_mean = border_mean
        self.border_p = border_p
        
        
    def __call__(self, label):
        """
        1. find border of the label
        2. sample some random points, which are on the border
        3. add random noise in x and y direction, maybe between 5-50px (not in which slice it is)
        4. categorize points as in or outside the label
        5. render points in volume with a fixed radius
        """
        
        n_points = np.random.randint(low=self.range_sampled_points[0],high=self.range_sampled_points[1]+1)
        nnz_slices = label.nonzero()[-1]
        slices = np.random.choice(nnz_slices,n_points)
        
        # print("Simulating",n_points)
        
        centers = []
        values = []
        for slice_idx in slices:
            label_slice = label[:,:,slice_idx].astype(int)
            label_border = label_slice - binary_erosion(label_slice).astype(label_slice.dtype)
            nnz_border = label_border.nonzero()
            border_idx = np.random.randint(low=0,high=len(nnz_border[0]))
            center = np.array([nnz_border[0][border_idx],nnz_border[1][border_idx]])
            center_rc = center + self._randomSign()*(np.random.binomial(2*self.border_mean,self.border_p,size=2)+1)
            centers.append(np.append(center_rc,slice_idx))
            values.append(label_slice[center_rc[0],center_rc[1]])
            
        # print(centers)
        
        points_vol = np.zeros(self.shape).astype(np.float32)
        
        for c,v in zip(centers,values):
            idx = c.reshape(3,1)+self.sphere_nnz
            points_vol[idx[0],idx[1],idx[2]] = 2*v-1
            
        self.centers = centers
        return points_vol
    
    
    def _randomSign(self):
        return (2*np.random.randint(0,2,size=2)-1)
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    import time
    
   
    lab_name = "../data/Synapse/labelsTr/label0003.nii.gz"
    label = nib.load(lab_name).get_fdata()
    label[~((label==2) | (label==3))] = 0
    label[((label==2) | (label==3))] = 1
    label = cv2.resize(label,dsize=(256,256))
    # label = torch.from_numpy(label).permute(2,0,1)
    
    
    pointMaker = pointSimulator(shape=label.shape,
                                range_sampled_points = [20,20])
    
    # n = 1000
    # t = time.time()
    # for _ in range(n):
    #     _ = pointMaker(label)
    # t1 = time.time()
    # print("Elapsed time", t1-t)
    # print("Average time", (t1-t)/n)
    
    point_vol = pointMaker(label)
    indices = pointMaker.centers
    

    fig = plt.figure(figsize=(10,10))
    for i in range(4):
        plt.subplot(2,2,i+1)
        point_g = point_vol[:,:,indices[i][-1]].copy()
        point_g[point_g<0]=0
        point_b = -point_vol[:,:,indices[i][-1]].copy()
        point_b[point_b<0] = 0
        im = np.zeros((256,256,3))
        im[:,:,0] = label[:,:,indices[i][-1]]
        im[:,:,1] = point_g
        im[:,:,2] = point_b
        plt.imshow(im)
    plt.show()
    