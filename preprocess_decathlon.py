import os 
import glob 
import numpy as np 
import nibabel as nib
from tqdm import tqdm
from functions import crop_CT
import cv2
datasets = 'Decathlon'
data_path='../data/'
im_path = os.path.join(data_path, datasets, "imagesTr")
label_path = os.path.join(data_path, datasets, "labelsTr")
im_list = glob.glob(os.path.join(im_path,"*.nii*"))
save_dir_ims = "../data/preprocessed_Decathlon/imagesTr"
save_dir_labs = "../data/preprocessed_Decathlon/labelsTr"

crop_size = [192,192,96]

diffs = []

for img_name in tqdm(im_list):
    
    lab_name = os.path.join(label_path, os.path.basename(img_name))

    img = nib.load(img_name)
    
    image = img.get_fdata()
    lab = nib.load(lab_name)
    label = lab.get_fdata()
    label_shape = label.shape
    nnz=np.array(label.nonzero())
    spleen_mean = np.array(label.nonzero()).mean(1).round().astype(int)
    # diff = nnz.max(1)-nnz.min(1)
    # diffs.append(diff)
    
    label_crop = label[
                        spleen_mean[0]-crop_size[0]//2 : spleen_mean[0]-crop_size[0]//2,
                        spleen_mean[1]-crop_size[1]//2 : spleen_mean[1]-crop_size[1]//2,
                        min(spleen_mean[2]-crop_size[2]//2,0):max(spleen_mean[2]+crop_size[2]//2,label_shape[2])
                        ]
    
    image = image.astype(float)
    im_min, im_max = np.quantile(image,[0.001,0.999])
    image = (np.clip((image-im_min)/(im_max-im_min),0,1)*255).astype(np.float32)
    
    image = crop_CT(image, 96, image.shape[2])
    label = crop_CT(label, 96, label.shape[2])

    if label.shape != (128,128,96):
        image = cv2.resize(image,dsize=(128, 128), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label,dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
    # label[(label==2)] = 1
        
    base_name = os.path.basename(img_name)
    f_names = base_name.replace(".nii.gz", "")
    save_path_ims = os.path.join(save_dir_ims, f_names)
    save_path_labs = os.path.join(save_dir_labs, f_names)
    np.save(save_path_ims, image)
    np.save(save_path_labs, label)
    