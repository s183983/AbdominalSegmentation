import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np  
import pydicom as dcm
import os 
from pydicom.errors import InvalidDicomError
from vtk.util.numpy_support import numpy_to_vtk
import vtk 
import torchio as tio
from functions import (
    num_of_params,
    sample_data,
    loss_func,
    requires_grad,
    get_transform_tr,
    load_state_dict_loose,
    crop_CT,
    get_augmentation
    )
from config import (
    get_args,
    update_args,
    )
from model_unet_3d import UNet3D
import cv2


#root = "C:/Users/lakri/Desktop/DTU/9.semester/Special Course/data/"
#%%

img = nib.load(root + "labels/label0001.nii")
data = img.get_fdata()
#%%
img = nib.load(root + "Decathlon/imagesTr/pancreas_117.nii")
data = img.get_fdata()
#%%
image = tio.ScalarImage(root + "Decathlon/imagesTr/pancreas_117.nii")
resize = tio.Resize([128,128,100])
resize_img = resize(image)

#%%
dmax1 = np.max(data[:,:,10])
print(dmax1)
plt.imshow(data[:,:,120]*255)
#%%
sumdat = np.sum(data,axis=2)
#%%
plt.imshow(sumdat)
#%%
ds = dcm.read_file(root + "Pancreas-CT/PANCREAS_0001/1-001.dcm", force=False)
slice1 = decompress.ds()
#%%
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
#%%
im_path = os.path.join(root,"Pancreas-CT/PANCREAS_0001/")
image_data = read_dicom_files_carefully(im_path)
#%%

im_path = 'C:/Users/lakri/Desktop/DTU/9.semester/Special Course/data/Decathlon/imagesTr/'
z_size = np.zeros([len(im_path)])
z_max = 0
z_min = 1000
i  = 0
for i, im in enumerate(os.listdir(im_path)):
    scan_im = nib.load(im_path+im)
    
    z_size[i] = scan_im.shape[2]
    
    if z_size[i] > z_max: 
        z_max = z_size[i]
        im_max_name = im
        
    if z_size[i] < z_min: 
        z_min = z_size[i]
        im_min_name = im
#%%
import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=z_size, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#%%
from functions import crop_CT
img = nib.load(root + "Decathlon/imagesTr/pancreas_074.nii")
data = img.get_fdata()
n_img = crop_CT(data, 100, data.shape[2])
n_img.shape
#%%
resized_img = cv2.resize(n_img,dsize=(128,128),interpolation=cv2.INTER_AREA)
resized_img.shape
#%%
fig, ax = plt.subplots(2)
ax[0].imshow(np.squeeze(n_img[:,:,20]))
ax[1].imshow(np.squeeze(resized_img[:,:,20]))
#%%
from functions import (
    num_of_params,
    sample_data,
    loss_func,
    requires_grad,
    get_transform_tr,
    )
from config import (
    get_args,
    update_args,
    )
from dataset_loader_v2 import CT_Dataset
import argparse
import torch
device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, 
                    default='default', 
                    help="name of the model")

args_name = parser.parse_args()

args = get_args(name=args_name.name)
args_dict = get_args(name=args_name.name,dict_mode=True)
transform_tr = get_transform_tr(ite=0,
    max_p=0.6,rampup_ite=args.training.aug_rampup) if args.training.augment else None

ds_tr = CT_Dataset(mode="train",
                     data_path='C:/Users/lakri/Desktop/DTU/9.semester/Special Course/data/',
                     transform=transform_tr,
                     reshape = args.training.reshape,
                     reshape_mode = args.training.reshape_mode,
                     datasets = args.training.datasets,
                     interp_mode = args.training.interp_mode)
dl_tr = torch.utils.data.DataLoader(ds_tr,batch_size=args.training.batch, drop_last=True,num_workers=0)
dl_tr = sample_data(dl_tr)
img, label_gt = next(dl_tr)
#%%
plt.imshow(np.squeeze(img[0,50,:,:]))
#%%
root = "C:/Users/lakri/Desktop/DTU/9.semester/Special Course/data/"
lab_name = root + "Decathlon/labelsTr/pancreas_069.nii"
lab = nib.load(lab_name)
label = lab.get_fdata()
#%%
plt.imshow(label[:,:,69]*255)
#%%
image = np.load(root+"preprocessed_Decathlon/imagesTr/pancreas_001.npy")
label = np.load(root+"preprocessed_Decathlon/labelsTr/pancreas_001.npy")
image = np.expand_dims(image,0)
label = np.expand_dims(label,0)

transforms_dict = {tio.RandomAffine(p=0.25),
                   tio.RandomElasticDeformation(p=0.25),
                   tio.RandomFlip((0, 1, 2), p=1)
                   }
transform = tio.Compose(transforms_dict)
aug_image = transform(image)
aug_label = transform(label)
fig, ax =  plt.subplots(2,2)
ax[0,0].imshow(np.squeeze(image[:,:,:,50]))
ax[0,1].imshow(np.squeeze(aug_image[:,:,:,50]))
ax[1,0].imshow(np.squeeze(label[:,:,:,50]))
ax[1,1].imshow(np.squeeze(aug_label[:,:,:,50]))
#%%
aug = get_augmentation((128, 128, 96))
#%%
from functions import (
    num_of_params,
    sample_data,
    loss_func,
    requires_grad,
    get_transform_tr,
    load_state_dict_loose,
    crop_CT,
    get_augmentation
    )
image = np.load("../data/preprocessed_Decathlon/imagesTr/pancreas_001.npy")
mask = np.load("../data/preprocessed_Decathlon/labelsTr/pancreas_001.npy")



data = {'image': image, 'mask': mask}
aug_data = aug(**data)
aug_image = aug_data['image']
aug_mask = aug_data['mask']
s = 50
fig, ax =  plt.subplots(2,2)
ax[0,0].imshow(np.squeeze(image[:,:,s]))
ax[0,1].imshow(np.squeeze(aug_image[:,:,s]))
ax[1,0].imshow(np.squeeze(mask[:,:,s]))
ax[1,1].imshow(np.squeeze(aug_mask[:,:,s]))
#%%
from functions import (
    num_of_params,
    sample_data,
    loss_func,
    requires_grad,
    get_transform_tr,
    load_state_dict_loose,
    crop_CT,
    get_augmentation
    )
image = np.load("../data/preprocessed_Synapse/imagesTr/img0001.npy")
mask = np.load("../data/preprocessed_Synapse/labelsTr/label0001.npy")



data = {'image': image, 'mask': mask}
aug_data = aug(**data)
aug_image = aug_data['image']
aug_mask = aug_data['mask']
s = 10
fig, ax =  plt.subplots(2,2)
ax[0,0].imshow(np.squeeze(image[:,:,s]))
ax[0,1].imshow(np.squeeze(aug_image[:,:,s]))
ax[1,0].imshow(np.squeeze(mask[:,:,s]))
ax[1,1].imshow(np.squeeze(aug_mask[:,:,s]))
#%%
from functions import pointSimulator2
import matplotlib.pyplot as plt 
lab_name = "../data/Synapse/labelsTr/label0003.nii.gz"
label = nib.load(lab_name).get_fdata()
label[~((label==2) | (label==3))] = 0
label[((label==2) | (label==3))] = 1
label = cv2.resize(label,dsize=(256,256), interpolation = cv2.INTER_NEAREST)
label = label[:,:,120]
pred = label.copy()
pred[150:180,80:120]=0
pred[180:200,120:140] = 1 
#plt.imshow(pred)
# label = torch.from_numpy(label).permute(2,0,1)

#%%
pointMaker = pointSimulator2(shape=label.shape,
                            range_sampled_points = [20,20])
pointMaker(label_pred = pred, label_gt = label)