# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:15:12 2022

@author: lowes
"""
import numpy as np
import torch
import nibabel as nib

class pointSimulator():
    def __init__(self,
                 shape=[96,128,128],
                 radius=[1,1,1]
                 ):
        self.shape = shape
        self.radius = radius
        
        
    def __call__(self, label):
        """
        1. find border of the label
        2. sample some random points, which are on the border
        3. add random noise in x and y direction, maybe between 5-50px (not in which slice it is)
        4. categorize points as in or outside the label
        5. render points in volume with a fixed radius
        """
        
        return points_vol
    
    
if __name__=="__main__":
    pointMaker = pointSimulator()
    lab_name = "../Decathlon/labelsTr/spleen_2.nii.gz"
    label = nib.load(lab_name).get_fdata()

    