import argparse
import os

import glob

import torch
import wandb
import copy
from torch import optim
from torchvision import utils, transforms
from tqdm import tqdm
import _pickle as cPickle
from dataset_loader_v2 import CT_Dataset

import warnings 
from functions import (
    num_of_params,
    sample_data,
    loss_func,
    requires_grad,
    get_transform_tr,
    load_state_dict_loose,
    fxn,
    HiddenPrints,
    saveTestSample,
    pointSimulator2
    )
from config import (
    get_args,
    update_args,
    )
from model_unet_3d import UNet3D #AbstractUNet
from point_sim import pointSimulator
with HiddenPrints():
    import numpy as np    

ROOT = "../"

def train_classifier(args, net, optim_net, start_iter, 
                     dl_tr, dl_va, ds_te, ds_tr):
    
    if not os.path.isdir(ROOT+"runs"):
        os.mkdir(ROOT+"runs")
    if not os.path.isdir(ROOT+"runs/"+args.name):
        os.mkdir(ROOT+"runs/"+args.name)
    if not os.path.isdir(ROOT+"runs/"+args.name+"/checkpoint"):
        os.mkdir(ROOT+"runs/"+args.name+"/checkpoint")
    if not os.path.isdir(ROOT+"runs/"+args.name+"/images"):
        os.mkdir(ROOT+"runs/"+args.name+"/images")
    
    
    device = "cuda"

    pbar = tqdm(range(args.training.max_iter), initial=start_iter, dynamic_ncols=True, smoothing=0.01)
    
    vali_every_ite = 200
    n_vali = max(2//args.training.batch,1)
    save_vali_every = 200
    ite_skeleton_p = 0
    n_vali_ite = int(np.ceil(ite_skeleton_p*n_vali))
    n_vali_ite_vec = ([True]*n_vali_ite)+([False]*(n_vali-n_vali_ite))
    save_checkpoint_every = 500
    
    eps = 1e-12
    assert np.abs(float(save_vali_every//vali_every_ite)
                  -save_vali_every/vali_every_ite)<eps
    
    # RGBexpand = torch.tensor([1,1,1]).view(1,-1,1,1).to(device)
    requires_grad(net, True)
    
    loss_vali = float('NaN')

    s = torch.nn.Sigmoid() if args.training.recon_mode=="BCE" else torch.nn.Identity()
    
    losses_10k_reset = {"loss_vali": {}, "loss_vali_ite": {}, "loss_vali_no_ite": {},
                   "loss_train": {}, "loss_train_ite": {}, "loss_train_no_ite": {},}
    losses_10k = copy.deepcopy(losses_10k_reset)
    pointMaker = pointSimulator2(**vars(args.pointSim))
    for idx in pbar:
        i = idx + start_iter
        
        if i > args.training.max_iter:
            print("Done!")
            break
        
        net.train()
        img, label_gt = next(dl_tr)
        # print("fetched data")
        img = img.to(device, dtype=torch.float)
        label_gt = label_gt.to(device, dtype=torch.float)
        ite_bool_train = np.random.rand()<0.1 and i>=1000
        
        with torch.no_grad():
            if ite_bool_train:
                label_pred = net(img)
                point_vol = torch.from_numpy(pointMaker(label_gt = label_gt, label_pred = label_pred))
                # img = torch.stack((img, point_vol), dim=1)
                img[:,1] = point_vol.squeeze(1) #.permute(0,4,1,2,3)
        
        
        
        label_gt = (label_gt>eps).type(torch.float)       
        label_pred = net(img)
        # print("doing pred")
        loss_net = loss_func(label_gt,label_pred,
                             recon_mode=args.training.recon_mode,
                             weight_mode=args.training.weight_mode_loss,
                             point = img[:,1].unsqueeze(1))
        # print("pred done")
        net.zero_grad()
        loss_net.backward()
        optim_net.step()
        # # print("optimized")
        # if i%vali_every_ite==0 and args.training.augment:
        #     if args.training.aug_rampup is not None:
        #         if i<=args.training.aug_rampup:
        #             ds_tr.transform = get_transform_tr(
        #                 ite=0,max_p=0.6,rampup_ite=args.training.aug_rampup)
        #             dl_tr = torch.utils.data.DataLoader(ds_tr,batch_size=args.training.batch,
        #                 sampler=ds_tr.data_sampler,
        #                 shuffle=ds_tr.data_sampler is None,drop_last=True,num_workers=4)
        #             dl_tr = sample_data(dl_tr)
        
        if (i%vali_every_ite)==0:
            net.eval()
            with torch.no_grad():
                loss_vali = 0
                loss_vali_ite = 0
                loss_vali_no_ite = 0
                for vali_i in range(n_vali):
                    #ite_bool = n_vali_ite_vec[vali_i]
                    
                    img, label_gt = next(dl_va)
                    img = img.to(device, dtype=torch.float)
                    label_gt = label_gt.to(device, dtype=torch.float)
                    if ite_bool_train:
                        label_pred = net(img)
                        point_vol = torch.from_numpy(pointMaker(label_gt = label_gt, label_pred = label_pred))
                        img = torch.stack(img, point_vol, dim=1)
                        #img[:,1,:,:] = point_vol.squeeze(1) #.permute(0,3,1,2)
                        
                    # img = torch.stack((img, point_vol)).permute(0,3,1,2)
                    label_gt = (label_gt>eps).type(torch.float)
                    label_pred = net(img)
                    
                    loss_vali_tmp = loss_func(label_gt, label_pred, 
                                              recon_mode=args.training.recon_mode,
                                              weight_mode=args.training.weight_mode_loss,
                                              point = img[:,1].unsqueeze(1)).item()
                    loss_vali += loss_vali_tmp

                    loss_vali_no_ite += loss_vali_tmp
                        
                loss_vali = loss_vali/n_vali
                loss_vali_no_ite = loss_vali_no_ite/(n_vali-n_vali_ite+1e-14)
                
                if (i%save_vali_every)==0:
                    n_im = min(10,label_pred.shape[0])
                    # utils.save_image(
                    #     torch.cat((img[:n_im]*0.5+0.5,
                    #                label_gt[:n_im]*RGBexpand,
                    #                s(label_pred[:n_im])*RGBexpand,
                    #                dim=0),
                    #               ROOT+"runs/"+args.name+f"/images/{str(i).zfill(6)}.png",
                    #               nrow=n_im,
                    #               ncol=4,
                    #               normalize=True,
                    #               range=(0, 1),
                    #               )
                
        pbar.set_description(
            (
                f"l_tr: {loss_net:.4f}; "
                f"l_vali: {loss_vali:.4f}; "
            )
        )

        
        log_dict = {"loss_train": loss_net}
        # if ite_bool_train:
        #     log_dict["loss_train_ite"] = loss_net
        # else:
        #     log_dict["loss_train_no_ite"] = loss_net 
            
        if (i%vali_every_ite)==0:
            log_dict["loss_vali"] = loss_vali
            log_dict["loss_vali_ite"] = loss_vali_ite
            log_dict["loss_vali_no_ite"] = loss_vali_no_ite
            
        if wandb and args.wandb: wandb.log(log_dict)
        
        for key,value in log_dict.items():
            losses_10k[key][i] = value

        if ((i+1)%save_checkpoint_every)==0:
            if i>save_checkpoint_every:
                try:
                    losses_save = torch.load(ROOT+"runs/"+args.name+
            f"/checkpoint/{str(i+1-save_checkpoint_every).zfill(6)}.pt", 
            map_location=lambda storage, loc: storage)["losses"]
                    losses_cat = {}
                    for key, value in losses_10k.items():
                        losses_cat[key] = {**losses_save[key],**value}
                    del losses_save
                except:
                    losses_cat = losses_10k
            else:
                losses_cat = losses_10k
                
            save_dict = {
                "net": net.state_dict(),
                "net_optim": net_optim.state_dict(),
                "args": args,
                "losses": losses_cat,
            }
            torch.save(save_dict,ROOT+"runs/"+args.name+f"/checkpoint/{str(i+1).zfill(6)}.pt")
            del save_dict
            del losses_cat
            losses_10k = copy.deepcopy(losses_10k_reset)
            
            saveTestSample(img, label_pred, label_gt, ROOT+"runs/"+args.name+"/images"+f"/{str(i+1).zfill(6)}.png")

if __name__ == "__main__":
    device = "cuda"
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy detected version 1.23.1')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, 
                        default='kidneyPointSniperDebug', 
                        help="name of the model")
    
    args_name = parser.parse_args()
    
    args = get_args(name=args_name.name)
    args_dict = get_args(name=args_name.name,dict_mode=True)

    net = UNet3D(**vars(args.unet)).to(device)
    num_of_params(net)
    net_optim = optim.Adam(
        net.parameters(),
        weight_decay=args.training.weight_decay,
        betas=(0.5, 0.999),
        lr=args.training.lr)
    
    load_list = sorted(glob.glob(ROOT+"runs/"+args.name+"/checkpoint/*"))
    
    if len(load_list)>0:
        ckpt = torch.load(load_list[-1], map_location=lambda storage, loc: storage)
        start_iter = int(os.path.basename(load_list[-1]).split('.')[0])
        #args = ckpt['args']
        #args = update_args(args)
        net.load_state_dict(ckpt["net"])
        net_optim.load_state_dict(ckpt["net_optim"])
    elif args.training.pretrain_name_ite is not None:
        start_iter = 0
        name_load = args.training.pretrain_name_ite.split("/")
        if len(name_load)>1:
            load_list = sorted(glob.glob(ROOT+"runs/"+name_load[0]+"/checkpoint/"+name_load[1]+"*"))
            ckpt = torch.load(load_list[-1], map_location=lambda storage, loc: storage)
        else:
            load_list = sorted(glob.glob(ROOT+"runs/"+name_load[0]+"/checkpoint/*"))
            ckpt = torch.load(load_list[-1], map_location=lambda storage, loc: storage)
            
        try:
            net.load_state_dict(ckpt["net"])
            net_optim.load_state_dict(ckpt["net_optim"])
        except:
            try:
                net.load_state_dict(ckpt["net"],strict=False)
                print("WARNING: using non-strict for Discriminator")
            except:
                net = load_state_dict_loose(net,ckpt["net"],verbose=True)
                print("WARNING: using loose loader for Discriminator")
    else:
        start_iter = 0
    
    # transform_tr = get_transform_tr(ite=0,
    #     max_p=0.6,rampup_ite=args.training.aug_rampup) if args.training.augment else None
    
    ds_tr = CT_Dataset(mode="train",
                         data_path='../data',
                         transform=args.training.augment,
                         reshape = args.training.reshape,
                         reshape_mode = args.training.reshape_mode,
                         datasets = args.training.datasets,
                         interp_mode = args.training.interp_mode,
                         tissue_range = args.training.tissue_range,
                         args = args)

    ds_va = CT_Dataset(mode="train",
                         data_path='../data',
                         transform=args.training.augment,
                         reshape = args.training.reshape,
                         reshape_mode = args.training.reshape_mode,
                         datasets = args.training.datasets,
                         interp_mode = args.training.interp_mode,
                         tissue_range = args.training.tissue_range,
                         args = args)
    ds_te = None
            
    
    dl_tr = torch.utils.data.DataLoader(ds_tr,
                                        batch_size=args.training.batch,
                                        drop_last=True,
                                        num_workers=args.training.batch)
    dl_va = torch.utils.data.DataLoader(ds_va,
                                        batch_size=args.training.batch,
                                        drop_last=True,
                                        num_workers=args.training.batch)
    
    dl_tr = sample_data(dl_tr)
    dl_va = sample_data(dl_va)
    
    if wandb is not None and args.wandb:
        if os.getlogin()=="lowes" or os.getlogin()=="s183983":
            wandb.init(project='3DUnet_onescan', entity='s183983',config=args_dict)
        else:
            wandb.init(project='3DUnet_onescan', entity='Bjonze',config=args_dict)
        
    train_classifier(args, net, net_optim, start_iter, dl_tr, dl_va, ds_te, ds_tr)