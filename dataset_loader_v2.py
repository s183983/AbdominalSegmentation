from __future__ import print_function, division
import os
import torch
#from skimage import io, transform
#import numpy as np
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import nibabel as nib
import torchio as tio
from functions import crop_CT
import cv2
import numpy as np
import glob
from functions import get_augmentation
from point_sim import pointSimulator

class CT_Dataset(torch.utils.data.Dataset):
    

    def __init__(self, mode="train",
                     data_path="../data",
                     transform=False,
                     reshape=[128,128,128],
                     reshape_mode = None, # ['padding', 'fixed_size' or None]
                     datasets = "preprocessed_Synapse",
                     interp_mode=["area","nearest"],
                     tissue_range = [-100,600],
                     args = None):
    
        if mode=="train" and (datasets=="Decathlon" or datasets=="Decathlon1" or datasets=="Synapse"):
            self.data_path = os.path.join(data_path, datasets, "imagesTr")
            self.data_list = glob.glob(os.path.join(self.data_path,"*.nii*"))
            self.label_path = os.path.join(data_path, datasets, "labelsTr")
            self.label_list = glob.glob(os.path.join(self.label_path,"*.nii*"))
            
        elif mode=="test" and (datasets=="Decathlon"or datasets=="Synapse"):
            # self.data_path = os.path.join(data_path+datasets, "imagesTs")
            self.data_path = os.path.join(data_path, datasets, "imagesTr")
            self.data_list = glob.glob(os.path.join(self.data_path,"*.nii*"))
            self.label_path = os.path.join(data_path, datasets, "labelsTr")
            self.label_list = glob.glob(os.path.join(self.label_path,"*.nii*"))
            
        elif mode=="train" and datasets.find("preprocessed")!=-1:
            self.data_path = os.path.join(data_path, datasets, "imagesTr")
            self.data_list = glob.glob(os.path.join(self.data_path,"*.npy*"))
            self.label_path = os.path.join(data_path, datasets, "labelsTr")
            self.label_list = glob.glob(os.path.join(self.label_path,"*.npy*"))
        
        elif mode=="test" and datasets=="preprocessed_Decathlon":
            self.data_path = os.path.join(data_path+datasets, "imagesTs")
            
        else:
            raise ValueError("did not recognize mode: "+mode+" or datasets: "+datasets)
            
            
        # self.transforms_dict = {tio.RandomAffine(p=0.25),
        #                    tio.RandomElasticDeformation(p=0.25),
        #                    tio.RandomFlip((0, 1, 2), p=1)
        #                    }
        self.mode = mode
        self.tissue_range = tissue_range
        self.reshape = reshape
        self.reshape_mode = reshape_mode
        if reshape_mode == "fixed_size":
            self.imageResizer = tio.Resize(target_shape=self.reshape,
                                           image_interpolation="linear",
                                           label_interpolation="linear")
            self.labelResizer = tio.Resize(target_shape=self.reshape,
                                           image_interpolation="nearest",
                                           label_interpolation="nearest")
        self.datasets = datasets
        if transform and mode=="train":
            self.transform = get_augmentation((reshape[0], reshape[1], reshape[2]))
        else: 
            self.transform = None
            
        interp_dict = {"NEAREST": cv2.INTER_NEAREST,
                       "NEAREST_EXACT": cv2.INTER_NEAREST_EXACT,
                       "LINEAR": cv2.INTER_LINEAR,
                       "BILINEAR": cv2.INTER_LINEAR,
                       "CUBIC": cv2.INTER_CUBIC,
                       "BICUBIC": cv2.INTER_CUBIC,
                       "AREA": cv2.INTER_AREA,
                       "LANCZOS4": cv2.INTER_LANCZOS4,}
        self.interp_modes = [interp_dict[i_m.upper()] for i_m in interp_mode]
        
        if args.training.do_pointSimulation:
            self.pointSimulator = pointSimulator(**vars(args.pointSim))
            self.pointSimultionProb = args.training.do_pointSimulation
        else:
            self.pointSimulator = None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        # imgs = os.listdir(self.data_path)[idx]
        # labs = os.listdir(self.label_path)[idx]
        
        img_name = self.data_list[idx]
        lab_name = self.label_list[idx]
        #lab_name = os.path.join(self.label_path, os.path.basename(img_name)).replace('img','label')
        # self.img_name = img_name
        # img_name = os.path.join(self.data_path, imgs)
        # lab_name = os.path.join(self.label_path, labs)
        

        # image = tio.ScalarImage(img_name)
        # label = tio.ScalarImage(lab_name)

        if (self.datasets).find("preprocessed")==-1:
            img = nib.load(img_name)
            image = img.get_fdata()
            lab = nib.load(lab_name)
            label = lab.get_fdata()
            image = image.astype(float)
            if self.datasets=="Synapse":
                label[~((label==2) | (label==3))] = 0
                label[((label==2) | (label==3))] = 1
            # im_min, im_max = np.quantile(image,[0.001,0.999])
            # image = (np.clip((image-im_min)/(im_max-im_min),0,1)*255).astype(np.float32)
            #im_min, im_max = self.tissue_range
            #image = np.clip((image-im_min)/(im_max-im_min),0,1).astype(np.float32)
        elif self.datasets == "preprocessed_Decathlon" or "preprocessed_Synapse":
            image = np.load(img_name)#/255
            label = np.load(lab_name)#/255
        
        if self.reshape_mode == "padding":
            if self.reshape is not None:
                image = crop_CT(image,self.reshape[2],image.shape[2])
                label = crop_CT(label,self.reshape[2],label.shape[2])
            else:
                image = crop_CT(image,100,image.shape[2])
        
        elif self.reshape_mode == 'fixed_size':
            
            if image.shape != self.reshape:
                image = self.imageResizer(image[np.newaxis,...])[0]
                label = self.labelResizer(label[np.newaxis,...])[0]
        
        if image.shape != (self.reshape[0],self.reshape[1],self.reshape[2]):
            image = cv2.resize(image,dsize=(self.reshape[1],
                      self.reshape[0]),interpolation=self.interp_modes[0])
            label = cv2.resize(label,dsize=(self.reshape[1],
                      self.reshape[0]),interpolation=self.interp_modes[1])
            
        
        if self.transform is not None:
            data = {"image": image, "label": label}
            aug_batch = self.transform(**data)
        else:
            aug_batch = {"image": image, "label": label}
     
        
        #im_min, im_max = self.tissue_range
        #image = np.clip((image-im_min)/(im_max-im_min),0,1).astype(np.float32)
        # if np.amin(image)!= -1 or np.amax(image)!=1:
        #     image = np.clip(image, -1000, 1000)   
        #     image = image/1000
        #image = image.astype('float64')
        #labels = nib.load(lab_name)
        #data = img.get_fdata()


   
        # im_min, im_max = np.quantile(image,[0.001,0.999])
        # image = (np.clip((aug_batch["image"]-im_min)/(im_max-im_min),0,1)*2-1).astype(np.float32)
        im_min, im_max = self.tissue_range
        image = (np.clip((aug_batch["image"]-im_min)/(im_max-im_min),0,1)*2-1).astype(np.float32)
        image = torch.from_numpy(image)
        # image = torch.from_numpy(aug_batch["image"]).permute(2,0,1)*2-1 #/255*2-1
        
        
        
        if self.pointSimulator is not None:
            if np.random.random()<self.pointSimultionProb and self.mode=="train":
                point_vol = torch.from_numpy(self.pointSimulator(label))
            else:
                point_vol = torch.zeros_like(image) #(C,D,W,H)
            image = torch.stack((image,point_vol)).permute(0,3,1,2) #(C,D,W,H)
        else:
            image = image.permute(2,0,1).unsqueeze(0) #(C,D,W,H)
        label = torch.from_numpy(aug_batch["label"]).permute(2,0,1).unsqueeze(0) #(C,D,W,H)
        
        # if not isinstance(image.dtype, torch.float):
        #     image = image.type(torch.float)
        # if not isinstance(label.dtype, torch.float):
        #     label = label.type(torch.float)

        

        # if self.transform:
        #     sample = self.transform(sample)

        return image, label