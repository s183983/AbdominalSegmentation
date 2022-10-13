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

class CT_Dataset(torch.utils.data.Dataset):
    

    def __init__(self, mode="train",
                     data_path="../data",
                     transform=True,
                     reshape=[128,128,96],
                     reshape_mode = None, # ['padding', 'fixed_size' or None]
                     datasets = "preprocessed_Decathlon",
                     interp_mode=["area","nearest"]):
    
        if mode=="train" and (datasets=="Decathlon" or datasets=="Decathlon1"):
            self.data_path = os.path.join(data_path, datasets, "imagesTr")
            self.label_path = os.path.join(data_path, datasets, "labelsTr")
            self.data_list = glob.glob(os.path.join(self.data_path,"*.nii*"))
            
        elif mode=="test" and datasets=="Decathlon":
            self.data_path = os.path.join(data_path+datasets, "imagesTs")
            
        elif mode=="train" and datasets=="preprocessed_Decathlon":
            self.data_path = os.path.join(data_path, datasets, "imagesTr")
            self.label_path = os.path.join(data_path, datasets, "labelsTr")
            self.data_list = glob.glob(os.path.join(self.data_path,"*.npy*"))
        
        elif mode=="test" and datasets=="preprocessed_Decathlon":
            self.data_path = os.path.join(data_path+datasets, "imagesTs")
            
        else:
            raise ValueError("did not recognize mode: "+mode+" or datasets: "+datasets)
            
            
        # self.transforms_dict = {tio.RandomAffine(p=0.25),
        #                    tio.RandomElasticDeformation(p=0.25),
        #                    tio.RandomFlip((0, 1, 2), p=1)
        #                    }

            
        self.reshape = reshape
        self.reshape_mode = reshape_mode
        self.datasets = datasets
        self.transform = transform
        interp_dict = {"NEAREST": cv2.INTER_NEAREST,
                       "NEAREST_EXACT": cv2.INTER_NEAREST_EXACT,
                       "LINEAR": cv2.INTER_LINEAR,
                       "BILINEAR": cv2.INTER_LINEAR,
                       "CUBIC": cv2.INTER_CUBIC,
                       "BICUBIC": cv2.INTER_CUBIC,
                       "AREA": cv2.INTER_AREA,
                       "LANCZOS4": cv2.INTER_LANCZOS4,}
        self.interp_modes = [interp_dict[i_m.upper()] for i_m in interp_mode]
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        # imgs = os.listdir(self.data_path)[idx]
        # labs = os.listdir(self.label_path)[idx]
        
        img_name = self.data_list[idx]
        lab_name = os.path.join(self.label_path, os.path.basename(img_name))
        
        # img_name = os.path.join(self.data_path, imgs)
        # lab_name = os.path.join(self.label_path, labs)
        

        # image = tio.ScalarImage(img_name)
        # label = tio.ScalarImage(lab_name)
        if self.datasets != "preprocessed_Decathlon":
            img = nib.load(img_name)
            image = img.get_fdata()
            lab = nib.load(lab_name)
            label = lab.get_fdata()
            image = image.astype(float)
            im_min, im_max = np.quantile(image,[0.001,0.999])
            image = (np.clip((image-im_min)/(im_max-im_min),0,1)*255).astype(np.float32)
        elif self.datasets == "preprocessed_Decathlon":
            image = np.load(img_name)
            label = np.load(lab_name)
        
        if self.reshape_mode == "padding":
            if self.reshape is not None:
                image = crop_CT(image,self.reshape[2],image.shape[2])
                label = crop_CT(label,self.reshape[2],label.shape[2])
            else:
                image = crop_CT(image,100,image.shape[2])
        
        elif self.reshape_mode == 'fixed_size':
            if self.reshape is not None:
                resize = tio.Resize(self.reshape)
            else:
                resize = tio.Resize([128,128,96])
            if image.shape != (1,128,128,96):
                image = resize(image)
                label = resize(label)
        
        if image.shape != (128,128,96):
            image = cv2.resize(image,dsize=(self.reshape[1],
                      self.reshape[0]),interpolation=self.interp_modes[0])
            label = cv2.resize(label,dsize=(self.reshape[1],
                      self.reshape[0]),interpolation=self.interp_modes[1])
            
        
        
        # if np.amin(image)!= -1 or np.amax(image)!=1:
        #     image = np.clip(image, -1000, 1000)   
        #     image = image/1000
        #image = image.astype('float64')
        #labels = nib.load(lab_name)
        #data = img.get_fdata()


        if self.transform is True:
            aug = get_augmentation((128, 128, 96))
            data = {"image": image, "label": label}
            aug_batch = aug(**data)
        else:
            aug_batch = {"image": image, "label": label}
        
        
        image = torch.tensor(aug_batch["image"]).permute(2,0,1)/255*2-1#/255*2-1
        image = image.unsqueeze(0)
        label = torch.tensor(aug_batch["label"]).permute(2,0,1).unsqueeze(0)
        
        # if not isinstance(image.dtype, torch.float):
        #     image = image.type(torch.float)
        # if not isinstance(label.dtype, torch.float):
        #     label = label.type(torch.float)

        

        # if self.transform:
        #     sample = self.transform(sample)

        return image, label