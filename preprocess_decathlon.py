import os 
import glob 
import numpy as np 
import nibabel as nib
from tqdm import tqdm
from functions import crop_CT
import cv2
datasets = 'Synapse'
data_path='../data/'
im_path = os.path.join(data_path, datasets, "imagesTr")
label_path = os.path.join(data_path, datasets, "labelsTr")
im_list = glob.glob(os.path.join(im_path,"*.nii*"))
path = "../data/preprocessed_"+datasets
save_dir_ims = path+"/imagesTr"
save_dir_labs =path+"/labelsTr"

if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(save_dir_ims):
    os.makedirs(save_dir_ims)
if not os.path.exists(save_dir_labs):
    os.makedirs(save_dir_labs)
    
crop_size = [128,128,128]

diffs = []

for img_name in tqdm(im_list):
    
    lab_name = os.path.join(label_path, os.path.basename(img_name))
    lab_name = lab_name.replace('img','label')

    img = nib.load(img_name)
    
    image = img.get_fdata()
    lab = nib.load(lab_name)
    label = lab.get_fdata()
    label_shape = label.shape
    if datasets=="Synapse":
        label[~((label==2) | (label==3))] = 0
        label[((label==2) | (label==3))] = 1
    
    
    # nnz=np.array(label.nonzero())
    # spleen_mean = np.array(label.nonzero()).mean(1).round().astype(int)
    # diff = nnz.max(1)-nnz.min(1)
    # diffs.append(diff)
    
    image = image.astype(float)
    im_min, im_max = np.quantile(image,[0.001,0.999])
    image = (np.clip((image-im_min)/(im_max-im_min),0,1)*255).astype(np.float32)
    
    image = crop_CT(image, 96, image.shape[2])
    label = crop_CT(label, 96, label.shape[2])
    
    if datasets=="Synapse":
        label[~((label==2) | (label==3))] = 0
        label[((label==2) | (label==3))] = 1

    if label.shape != crop_size:
        image = cv2.resize(image,dsize=(crop_size[0], crop_size[1]), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label,dsize=(crop_size[0], crop_size[1]), interpolation=cv2.INTER_NEAREST)
    # label[(label==2)] = 1
        
    img_base_name = os.path.basename(img_name)
    img_f_names = img_base_name.replace(".nii.gz", "")
    lab_base_name = os.path.basename(lab_name)
    lab_f_names = lab_base_name.replace(".nii.gz", "")
    
    save_path_ims = os.path.join(save_dir_ims, img_f_names)
    save_path_labs = os.path.join(save_dir_labs, lab_f_names)
    np.save(save_path_ims, image)
    np.save(save_path_labs, label)
    