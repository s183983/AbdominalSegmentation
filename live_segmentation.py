# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:33:51 2022

@author: lowes
"""
import os
import numpy as np
from PIL import Image
# import glob
import annotator_3d as ann

scale_to_screen = True

folder_name = "data/scans"

net_list = ["mrbean_190000", "mrbean_ite_200000", "mrbean_unmask_120000",
            "tree_330000", "tree_ite_240000","mix256_40000","amix128_080000",
            "all_110000", "med_060000"]

net_name = "amix_150000" #net_list[-2] #"graph_200000" net_list[-3]


resize_size = 256 if net_name.split('_')[0].find('256') != -1 else 128


ann.annotate(folder_name,
             net_name,
             resize_size=resize_size,
             scale_to_screen = scale_to_screen
             )

