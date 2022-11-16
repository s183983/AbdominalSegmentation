import argparse
import os
import torch
from torch import nn
from functions import (
    num_of_params,
    sample_data,
    loss_func,
    requires_grad,
    get_transform_tr,
    load_state_dict_loose,
    fxn,
    HiddenPrints,
    crop_CT, 
    pointSimulator2
    )
from config import (
    get_args,
    update_args,
    )
from model_unet_3d import UNet3D #AbstractUNet
from dataset_loader_v2 import CT_Dataset
import matplotlib.pyplot as plt 
with HiddenPrints():
    import numpy as np    
    
if __name__ ==  '__main__':
    device = "cuda"
    ROOT = "../"
    MODEL = '../runs/pls_learn/checkpoint/'
    check_point_name = os.path.join(ROOT, MODEL)
    name = os.path.join(MODEL, "030000.pt")
    net_name = 'pls_learn'
    device = "cuda"
    arg_name = ''.join(filter(lambda x: not x.isdigit(), net_name))
    #args = get_args(name=arg_name[:-1])
    args = get_args(name=arg_name)
    transform_tr = False   
    ds_tr = CT_Dataset(mode="train",
                         data_path='../data',
                         transform=args.training.augment,
                         reshape = args.training.reshape,
                         reshape_mode = args.training.reshape_mode,
                         datasets = args.training.datasets,
                         interp_mode = args.training.interp_mode,
                         tissue_range = args.training.tissue_range,
                         args = args)
    
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.training.batch, drop_last=True,num_workers=1)
    dl_tr = sample_data(dl_tr)
    img, label_gt = next(dl_tr)
    # for batch in dl_tr:
    #     img, label_gt = batch
    img = img.to(device, dtype=torch.float)
    with torch.no_grad():
        net = UNet3D(**vars(args.unet)).to(device)
        ckpt = torch.load(name, map_location=lambda storage, loc: storage)
        net.load_state_dict(ckpt["net"])
        net.eval()
        m = nn.Sigmoid()
        outpred = m(net(img)) 

#     s_n = 100
#     fig, ax = plt.subplots(1,3)    
#     ax[0].imshow(((np.squeeze(outpred[1,:,s_n,:,:].cpu().detach())).permute(1,0))>0.5)
#     ax[0].set_title('output pred')
#     ax[1].imshow((np.squeeze(label_gt[1,:,s_n,:,:].cpu().detach())).permute(1,0))
#     ax[1].set_title('GT')
#     ax[2].imshow((np.squeeze(img[1,:,s_n,:,:].cpu().detach())).permute(1,0))
#     ax[2].set_title('GT IMG')
 
# ps = pointSimulator2(shape = [128,128,128],
#                       radius = 1,
#                       sphere_size = (5,2), 
#                       range_sampled_points = [2, 10])
# #input dim: (B,C,D,W,H)
# ps(label_pred = outpred, label_gt = label_gt)














# transform_tr = False
# ds_tr = CT_Dataset(mode="train",
#                      data_path='C:/Users/lakri/Desktop/DTU/9.semester/Special Course/data/',
#                      transform=transform_tr,
#                      reshape = args.training.reshape,
#                      reshape_mode = args.training.reshape_mode,
#                      datasets = args.training.datasets,
#                      interp_mode = args.training.interp_mode)
# dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.training.batch, drop_last=True,num_workers=4)
# dl_tr = sample_data(dl_tr)
# #%%
# img, label_gt = next(dl_tr)




# #%%
# net = AbstractUNet(args).to(device)
  


# ckpt = torch.load(name, map_location=lambda storage, loc: storage)
# #args = ckpt['args']
# #args = update_args(args)
# net.load_state_dict(ckpt["net"])
# #%%
# ROOT = "C:/Users/lakri/Desktop/DTU/9.semester/Special Course/runs"
# MODEL = 'default/checkpoint'
# check_point_name = os.path.join(ROOT, MODEL)
# n_gpu_use = 1
# device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
# checkpoint = torch.load(check_point_name+'001500.pt', map_location=device)
# #%%
# parser = argparse.ArgumentParser()
# parser.add_argument("--name", type=str, 
#                     default='default', 
#                     help="name of the model")

# args_name = parser.parse_args()

# args = get_args(name=args_name.name)
# args_dict = get_args(name=args_name.name,dict_mode=True)

# net = UNet3D(**vars(args.unet))
# net.load_state_dict(checkpoint['model_state_dict'])
# net.eval()



# #%%
# n_gpu_use = 1
# device_ids = list(range(n_gpu_use))
# device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
# checkpoint = torch.load(check_point_name+'001500.pt', map_location=device)
# state_dict = []
# # Hack until all dicts are transformed
# if check_point_name.find('only_state_dict') == -1:
#     state_dict = checkpoint['state_dict']
# else:
#     state_dict = checkpoint

# if len(device_ids) > 1:
#     model = torch.nn.DataParallel(model, device_ids=device_ids)

# model.load_state_dict(state_dict)


# model = model.to(device)
# model.eval()
# #%%
# device = "cuda"
# ROOT = "C:/Users/lakri/Desktop/DTU/9.semester/Special Course/"
# MODEL = 'runs/default/checkpoint/'
# check_point_name = os.path.join(ROOT, MODEL)
# name = os.path.join(check_point_name,"001500.pt")
# #%%