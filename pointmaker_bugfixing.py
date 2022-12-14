from functions import pointSimulator2
import numpy as np 
from scipy.ndimage.morphology import distance_transform_edt
from random import randrange
import matplotlib.pyplot as plt
import raster_geometry as rg
x = np.load("../data/crash.npz")

diff_gt = x["diff_gt"]
diff_pred = x["diff_pred"]
#%%
diff_pred = np.squeeze(diff_pred, 1)
diff_gt = np.squeeze(diff_gt, 1)
im_diff = diff_gt + diff_pred
shape = [128,128,128] #(D,W,H) AKA (2,0,1)
radius = 3
sphere_size = (5,2)
range_sampled_points = [2, 10]
points_vol = np.zeros_like(diff_pred).astype(np.float32)
batch_ims_diff = np.reshape(diff_gt+diff_pred, [diff_pred.shape[0], shape[0]*shape[1]*shape[2]]).sum(axis=1)
radius = sphere_size[1]
sphere_size = sphere_size
range_sampled_points = range_sampled_points
sphere = rg.sphere(sphere_size[0],sphere_size[1]).astype(int)
sphere_nnz = np.array(sphere.nonzero())-sphere_size[0]//2
for i in range(batch_ims_diff.shape[0]):
    if batch_ims_diff[i]>((shape[0]*shape[1]*shape[2])*1e-4):
        n_points = np.random.randint(low=range_sampled_points[0],high=range_sampled_points[1]+1)
        nnz_slices = im_diff[i].nonzero()[0]
        nnz_slices = nnz_slices[(nnz_slices>radius) & (nnz_slices<(shape[0]-radius))]
        slices = np.random.choice(nnz_slices, n_points)
        centers = []
        values = []
        for slice_idx in slices:
            if (diff_gt[i,slice_idx,:,:].sum()+diff_pred[i,slice_idx,:,:].sum()) < 1:
                np.savez("crash_case.npz",diff_gt=diff_gt,diff_pred=diff_pred)
                
            if np.random.randint(diff_gt[i,slice_idx,:,:].sum()+diff_pred[i,slice_idx,:,:].sum())<diff_gt[i,slice_idx,:,:].sum():
                dist_im = distance_transform_edt(np.pad(np.squeeze(diff_gt[i,slice_idx,:,:]), [(1, 1), (1, 1)], mode='constant'))[1:-1,1:-1]
            else:
                dist_im = distance_transform_edt(np.pad(np.squeeze(diff_pred[i,slice_idx,:,:]), [(1, 1), (1, 1)], mode='constant'))[1:-1,1:-1]
            s1, s2 = dist_im.shape
            dist_im[:radius,:] = 0
            dist_im[:,:radius] = 0
            dist_im[(s1-radius):,:] = 0
            dist_im[:,(s2-radius):] = 0
            
            if dist_im.sum() < 1:
                continue

            tmp = dist_im.flatten()
            idx = np.random.choice(np.arange(tmp.size), size = 1, p=tmp/tmp.sum())
            point = np.asarray(np.unravel_index(idx, dist_im.shape)).T
    
            center = np.array([point[0][0], point[0][1]])
            centers.append(np.append(slice_idx, center))
            values.append(im_diff[i,slice_idx, point[0][0], point[0][1]])
        
    # print(centers)
    
    
        for c,v in zip(centers,values):
            idx = c.reshape(3,1)+sphere_nnz
            points_vol[i, idx[0], idx[1], idx[2]] = v
            
        centers = centers
