import torch
import cv2
import os
from pydicom.errors import InvalidDicomError
import pydicom as dcm
import warnings
import sys
import volumentations as V
import raster_geometry as rg
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
from random import randrange

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with HiddenPrints():
    import numpy as np
    #from scipy.ndimage import gaussian_filter

def load_state_dict_loose(model_arch,state_dict,
                          allow_diff_size=True,
                          verbose=False,
                          return_load_info=False):
    arch_state_dict = model_arch.state_dict()
    load_info = {"arch_not_sd": [],"sd_not_arch": [],"match_same_size": [], "match_diff_size": []}
    sd_keys = list(state_dict.keys())
    for name, W in arch_state_dict.items():
        if name in sd_keys:
            sd_keys.remove(name)
            s1 = np.array(state_dict[name].shape)
            s2 = np.array(W.shape)
            l1 = len(s1)
            l2 = len(s2)
            l_max = max(l1,l2)
            if l1<l_max:
                s1 = np.concatenate((s1,np.ones(l_max-l1,dtype=int)))
            if l2<l_max:
                s2 = np.concatenate((s2,np.ones(l_max-l2,dtype=int)))
                
            if all(s1==s2):
                load_info["match_same_size"].append(name)
                arch_state_dict[name] = state_dict[name]
            else:
                if verbose:
                    m = ". Matching." if allow_diff_size else ". Ignoring."
                    print("Param. "+name+" found with sizes: "+str(list(s1[0:l1]))
                                                      +" and "+str(list(s2[0:l2]))+m)
                if allow_diff_size:
                    s = [min(i_s1,i_s2) for i_s1,i_s2 in zip(list(s1),list(s2))]
                    idx1 = [slice(None,s[i],None) for i in range(l2)]
                    idx2 = tuple([slice(None,s[i],None) for i in range(l2)])
                    
                    if l1>l2:
                        idx1 += [0 for _ in range(l1-l2)]
                    idx1 = tuple(idx1)
                    tmp = state_dict[name][idx1]
                    arch_state_dict[name][idx2] = tmp
                load_info["match_diff_size"].append(name)
        else:
            load_info["arch_not_sd"].append(name)
    for name in sd_keys:
        load_info["sd_not_arch"].append(name)
    model_arch.load_state_dict(arch_state_dict)
    if return_load_info:
        return model_arch, load_info
    else:
        return model_arch

def cat(arrays,axis=0):
    n_dims = max(np.array([len(np.array(array).shape) for array in arrays]).max(),axis)
    cat_arrays = []
    for array in arrays:
        if np.size(array)>1:
            tmp = np.array(array).copy()
            tmp = np.expand_dims(tmp,axis=tuple(range(len(tmp.shape),n_dims)))
            cat_arrays.append(tmp)
    SHAPE = np.array([list(array.shape) for array in cat_arrays]).max(0)
    for i in range(len(cat_arrays)):
        reps = SHAPE//(cat_arrays[i].shape)
        reps[axis] = 1
        cat_arrays[i] = np.tile(cat_arrays[i],reps)
    cat_arrays = np.concatenate(cat_arrays,axis=axis)
    return cat_arrays


def num_of_params(net,full_print=False,no_print=False):
    n_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            n_param += param.data.numel()
            if full_print:
                print(name+", shape="+str(param.data.shape))
    if not no_print: print("Net has " + str(n_param) + " params.")
    return n_param


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def loss_func(gt,
              pred,
              recon_mode="BCE",
              weight_mode=None,
              point = None):
    if weight_mode is None:
        weight = 1
    else:
        pos_w_bool = False
        if isinstance(weight_mode, str):
            if len(weight_mode)>=3:
                if weight_mode[:3]=="pos":
                    pos_w_bool = True
                    if len(weight_mode)>3:
                        pos_w = float(weight_mode[3:])
                    else:
                        pos_w = 0.5
        equal_pos_w_bool = False
        if isinstance(weight_mode,str):
            if len(weight_mode)>=9:
                if weight_mode[:9]=="equal_pos":
                    equal_pos_w_bool = True
                    if len(weight_mode)>9:
                        pos_w = float(weight_mode[9:])
                    else:
                        pos_w = 0.5
        if pos_w_bool:
            weight = 2*(pos_w*gt+(1-pos_w)*(1-gt))
        elif equal_pos_w_bool:
            n_pixels = gt.shape[2]*gt.shape[3]
            n_pos_pixels = gt.sum((1,2,3),keepdim=True)
            weight = n_pixels*(pos_w *    gt  *(1/(n_pos_pixels+1e-14))+
                              (1-pos_w)* (1-gt) *(1/(n_pixels-n_pos_pixels+1e-14))
                        )
        else:
            raise ValueError("Did not recognize weight_mode: ",weight_mode)
            # ((H+1)/2-p)*abs(H) != 0
        
        
    if recon_mode=="L1":
        recon_loss = (torch.abs(pred-gt)*weight).mean()
    elif recon_mode=="L2":
        recon_loss = (((pred-gt)**2)*weight).mean()
    elif recon_mode=="BCE":
        bce_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        recon_loss = (bce_func(pred,gt.bernoulli())*weight).mean()
    elif recon_mode=="BCE_Point":
        bce_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        bce_loss = (bce_func(pred,gt.bernoulli())*weight).mean()
        point_loss = (( (point*2-1-pred)*point.abs())**2).mean()
        recon_loss = bce_loss + point_loss
    else:
        raise ValueError("Recon_mode must be one of ['L1', 'L2', 'BCE'] not ", recon_mode)
        
    return recon_loss


def get_synth_data(n = 64,bs=4,device="cuda",size_std=0.1,noise_std=0.01,threshold=0.01):
        
    im = np.zeros((bs,1,n,n))
    im[:,:,n//4:n*3//4,n//4:n*3//4] = np.random.randn(bs,1,n//2,n//2)
    im = gaussian_filter(im,sigma=(0,0,n*size_std,n*size_std))
    threshold_vec = np.minimum(threshold,np.quantile(im,0.99,axis=(1,2,3))).reshape((-1,1,1,1))
    label = im>threshold_vec
    
    im += np.random.randn(*im.shape)*noise_std
    im *= 20
    im[im>1] = 1
    im[im<-1] = -1
    
    return (torch.tensor(im,device=device,dtype=torch.float), 
            torch.tensor(label,device=device,dtype=torch.float))


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
        
def get_transform_tr(ite=None,max_p=0.6,rampup_ite=1500):
    if ite is None:
        ite = rampup_ite
    
    if rampup_ite is None:
        p = 1
    else:
        p = max_p*min(1,ite/rampup_ite)
    transform_tr = transform_tr = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=p),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.0, 0.2), rotate_limit=15, p=p*0.8, 
                               border_mode=cv2.BORDER_CONSTANT),
            A.PiecewiseAffine(scale=(0.01, 0.03),p=p*0.5),
            A.ColorJitter(brightness=0.15,p=p),
        ])
    return transform_tr

def get_augmentation(patch_size):
    return V.Compose([
        #V.ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # V.Flip(0, p=0.3),
        # V.Flip(1, p=0.3),
        V.RandomCropFromBorders(crop_2_min=0, crop_2_max=0, p=0.5),
        V.Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
        #V.Flip(2, p=0.5),
        #V.Rotate((0, 0), (0, 0), (-90, 90), p=1),
        # V.RandomRotate90((0, 1), p=0.3),
        V.RandomGamma(gamma_limit=(80, 120), p=0.2),
        V.GaussianNoise(var_limit=(0, 0.2), p=0.2),
    ], p=0.8)

def find_all_files(file_path):
    print(f'reading file list in {file_path}')
    unsorted_list = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            # the next line can be used if you need to filter files
            # if ".dcm" in file:  # exclude non-dicoms, good for messy folders
            unsorted_list.append(os.path.join(root, file))
    print(f'{len(unsorted_list)} files found.')
    return unsorted_list

def read_dicom_files_carefully(file_path):
    slices = []
    unsorted_list = find_all_files(file_path)
    if len(unsorted_list) < 1:
        print("Did not find any files")
        return None

    for dicom_loc in unsorted_list:
        valid_file = True
        ds = None
        try:
            ds = dcm.read_file(dicom_loc, force=False)
        except InvalidDicomError:
            print('Exception when trying to read and parse', dicom_loc)
            valid_file = False
        except Exception as ex:
            print('Exception when trying to read and parse', dicom_loc)
            template = "An exception of type {0} occurred. Arguments: {1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            valid_file = False
        if valid_file:
            # uncompress files (using the gdcm package)
            try:
                ds.decompress()
            except:
                print('an instance in file %s" could not be decompressed.' % dicom_loc)
                valid_file = False
            if valid_file:
                slices.append(ds)

    if len(slices) < 2:
        print("Error: Not enough slices in scan")
        return None

    slices.sort(key=lambda x: int(x.InstanceNumber))
    full_scan = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    full_scan = full_scan.astype(np.int16)
    image_data = np.moveaxis(full_scan,0,-1)

    return image_data    
    
def crop_CT(image,max_z,z_size):
    #if type(image) != numpy.ndarray
    assert z_size > 20
    
    if z_size > max_z: 
        z_diff = z_size - max_z
        k1 = int(np.floor(z_diff/2))
        k2 = int(z_diff - k1) 
        cropped_image = image[:,:,k1:(z_size-k2)]
    elif z_size < max_z:
        z_diff = int(max_z - z_size)
        pad_mat = np.zeros([image.shape[0],image.shape[1],z_diff])
        cropped_image = np.concatenate((image,pad_mat), axis=2)
    elif z_size == max_z:
        cropped_image = image
        
    return cropped_image

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
    

def reshapeCT(image, size=96):
    image = image.transpose(1,2,0)
    im_min, im_max = np.quantile(image,[0.001,0.999])
    # image = (np.clip((image-im_min)/(im_max-im_min),0,1)*255).astype(np.float32)
    
    image = crop_CT(image, size, image.shape[2])

    if image.shape != (128,128,size):
        image = cv2.resize(image,dsize=(128, 128), interpolation=cv2.INTER_AREA)
    
    return image.transpose(2,0,1)


def saveTestSample(img, pred, label, save_name):
    img = img.cpu().numpy()
    pred = pred.cpu().detach().numpy()
    label = label.cpu().numpy()
    
    point_vol = img[0,1]
    image = img[0,0]/2+0.5
    slices_nnz = point_vol.nonzero()[0]
    if not slices_nnz.size:
        slices_nnz = label[0,0].nonzero()[0]
    indices = np.random.choice(slices_nnz,2)
    
    fig = plt.figure(figsize=(12,6))
    for i in range(2):
        plt.subplot(2,3,i*3+1)
        plt.title("Input, slice: "+str(indices[i]))
        point_g = point_vol[indices[i]].copy()
        point_g[point_g<0]=0
        point_b = -point_vol[indices[i]].copy()
        point_b[point_b<0] = 0
        im = np.dstack((image[indices[i]],image[indices[i]],image[indices[i]]))
        im[point_g.astype(bool),1] = 1
        im[point_b.astype(bool),2] = 1
        plt.imshow(im)
        
        plt.subplot(2,3,i*3+2)
        plt.title("Label, slice: "+str(indices[i]))
        plt.imshow(label[0,0,indices[i]])
        
        plt.subplot(2,3,i*3+3)
        plt.title("Prediction, slice: "+str(indices[i]))
        plt.imshow(pred[0,0,indices[i]])

    plt.savefig(save_name)
    plt.close(fig)
    
class pointSimulator2():
    def __init__(self,
                 shape = [256,256,128],
                 radius = 1,
                 sphere_size = (5,2),
                 range_sampled_points = [2, 10],
                 ):
        # [H,W,D]
        self.shape = shape 
        # for use in rg.sphere
        self.radius = radius
        self.sphere_size = sphere_size
        self.range_sampled_points = range_sampled_points
        self.sphere = rg.sphere(sphere_size[0],sphere_size[1]).astype(int)
        self.sphere_nnz = np.array(self.sphere.nonzero())-sphere_size[0]//2
        
        
    def __call__(self, label_pred, label_gt):
        """
        1. find border of the label
        2. sample some random points, which are on the border
        3. add random noise in x and y direction, maybe between 5-50px (not in which slice it is)
        4. categorize points as in or outside the label
        5. render points in volume with a fixed radius
        """
        if torch.is_tensor(label_pred):
            label_pred = label_pred.cpu().detach().numpy()
        if torch.is_tensor(label_gt):
            label_gt = label_gt.cpu().detach().numpy()
            

        
       
        TH = 0.5
        label_pred = (label_pred>TH).astype(int)
        label_gt = label_gt.astype(int)
        
        im_diff = label_gt - label_pred
        diff_gt = im_diff>0
        diff_pred = im_diff<0
        points_vol = np.zeros(diff_pred.shape).astype(np.float32)
        
        for i in range(diff_pred.shape[0]):
            n_points = np.random.randint(low=self.range_sampled_points[0],high=self.range_sampled_points[1]+1)
            nnz_slices = im_diff.nonzero()[-1]
            
            
            slices = np.random.choice(nnz_slices, n_points) 
            
            # print("Simulating",n_points)
            
            centers = []

            
            values = []
            for slice_idx in slices:
                if np.random.randint(diff_gt[i,:,:,:,slice_idx].sum()+diff_pred[i,:,:,:,slice_idx].sum())<diff_gt[i,:,:,:,slice_idx].sum():
                    dist_im = distance_transform_edt(np.pad(np.squeeze(diff_gt[i,:,:,:,slice_idx]), [(1, 1), (1, 1)], mode='constant'))[1:-1,1:-1]
                else:
                    dist_im = distance_transform_edt(np.pad(np.squeeze(diff_pred[i,:,:,:,slice_idx]), [(1, 1), (1, 1)], mode='constant'))[1:-1,1:-1]
                
                tmp = dist_im.flatten()
                idx = np.random.choice(np.arange(tmp.size), size = 1, p=tmp/tmp.sum())
                point = np.asarray(np.unravel_index(idx, dist_im.shape)).T
        
                center = np.array([point[0][0], point[0][1]])
                centers.append(np.append(center,slice_idx))
                values.append(im_diff[i,:, point[0][0],point[0][1], slice_idx])
                
            # print(centers)
            
            
            for c,v in zip(centers,values):
                idx = c.reshape(3,1)+self.sphere_nnz
                points_vol[i, :, idx[0], idx[1], idx[2]] = v
                
            self.centers = centers
        return points_vol
    
    
    def _randomSign(self):
        return (2*np.random.randint(0,2,size=2)-1)