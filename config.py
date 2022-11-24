from types import SimpleNamespace
import json


def get_args(name="default",dict_mode=False):
    name_override=None
    
    args_dict = {"name": 'pls_learn',
                "wandb": True,
                "pointSim": {
                    "shape": [256,256,128],
                    "radius": 1,
                    "range_sampled_points": [2, 10],
                    "border_mean": 10,
                    "border_p": 0.4,
                    "sphere_size": (5,2)
                },
                "unet": {
                    "block": "ffmmm", #one of ['f','m']. m=MBConv (seperated conv),f=FusedMBConv (normal conv)
                    "act": "silu", #one of ['silu','relu']
                    "res_mode": "cat", #one of ['cat', 'add']
                    "init_mode": "effecientnetv2", #uses the init. from effecientnetv2 paper if 'effecientnetv2'
                    "downscale_mode": "avgpool", #one of ['maxpool','avgpool','conv']
                    "upscale_mode": "NDlinear", #one of ['nearest','bilinear','bicubic','trilinear','tconv','NDlinear']
                    #'bilinear' is only for 2D, "trilinear" is only for 3D. Will automatically find the correct one with 'NDlinear'
                    "input_channels": 1, #number of excepted input channels. E.g. 3 for RGB input, 1 for greyscale.
                    "num_blocks": 5, #One more than number of 2x downscales/upscales
                    "num_c": [4,8,16,32,64], #Number of channels in the blocks at different scales
                    "num_repeat": [1,2,2,4,4], #Number of repetitions of blocks at different scales
                    "expand_ratio": [1,4,4,6,6], #EffecientNetv2 expansion ratio in the blocks at different scales
                    "SE": [0,0,1,1,1],#bool defining if squeeze-and-excite layer be used at the end of blocks at different scales
                    "num_classes": 1,#Number of output channels (or classes)
                },
                "training": {
                    "reshape": [128,128,128], #reshape size for batch. If int instead of list then same reshape is used for spacial dimensions
                    "reshape_mode": None, # ['padding', 'fixed_size' or None]
                    "interp_mode": ["area","nearest"], #interpolation mode for rescaling of images
                    "max_iter": 30000, #Number of training iterations to complete training
                    "batch": 3, #Batch size
                    "lr": 1e-5,#Learning rate
                    "weight_decay": 1e-3,#Adam optimizer weight decay
                    "augment": True,
                    "aug_rampup": 10000, 
                    "recon_mode": "BCE", #one of ["L1", "L2", "BCE", "BCE_Point"]
                    "pretrain_name_ite": None,
                    "datasets": "Synapse", #"Pancreas-CT", Decathlon or preprocessed_Decathlon
                    "dataset_p": None,
                    "weight_mode_loss": None,
                    "do_pointSimulation": False,
                    "tissue_range": [-100,600]
                    
                }
            }
        
    if name is not None and name!="default":
        if name=="unet":
            args_mod = {}
        elif name=="train1":
            args_mod = {"training":{"datasets": "Decathlon1"}}
            args_mod["training"]["reshape_mode"] = 'padding'
        elif name=="just_learn":
            args_mod = {"training":{"datasets": "preprocessed_Decathlon"}}
            args_mod["training"]["max_iter"] = 15000
        elif name=="pls_learn":
            args_mod = {"training":{"datasets": "preprocessed_Synapse"}}
            args_mod["training"]["max_iter"] = 20000
        elif name=="kidneyPointSniper2000":
            shape = [192,192,128] #[256,256,128]
            args_mod = {"pointSim":{"shape": shape},
                        "training":{"reshape": shape},
                        "unet": {"input_channels": 2,
                                }}
            args_mod["training"]["max_iter"] = 20000
            args_mod["training"]["reshape_mode"] = "fixed_size"
            args_mod["training"]["do_pointSimulation"] = True
            args_mod["training"]["datasets"] = "Synapse"
            args_mod["training"]["batch"] = 2
            args_mod["training"]["lr"] = 1e-4
        elif name.find("kidneyPointSniper") != -1:
            shape = [192,192,128] #[256,256,128]
            args_mod = {"pointSim":{"shape": shape},
                        "training":{"reshape": shape},
                        "unet": {"input_channels": 2,
                                "block": "ffmmmm", #one of ['f','m']. m=MBConv (seperated conv),f=FusedMBConv (normal conv)
                                "num_blocks": 6, #One more than number of 2x downscales/upscales
                                "num_c": [4,8,16,32,64,128], #Number of channels in the blocks at different scales
                                "num_repeat": [1,2,2,4,4,6], #Number of repetitions of blocks at different scales
                                "expand_ratio": [1,4,4,6,6,8], #EffecientNetv2 expansion ratio in the blocks at different scales
                                "SE": [0,0,1,1,1,1]}}
            args_mod["training"]["max_iter"] = 25000
            args_mod["training"]["reshape_mode"] = "fixed_size"
            args_mod["training"]["do_pointSimulation"] = 0.4
            args_mod["training"]["datasets"] = "Synapse"
            args_mod["training"]["batch"] = 2
            args_mod["training"]["lr"] = 1e-4
            args_mod["training"]["recon_mode"] = "BCE_Point"
        else:
            raise ValueError('Invalid model name')
            
        if name.lower().find("debug") != -1:
            args_mod["wandb"] = False
            args_mod["unet"] = {"input_channels": 2,
                    "block": "ffmm", #one of ['f','m']. m=MBConv (seperated conv),f=FusedMBConv (normal conv)
                    "num_blocks": 4, #One more than number of 2x downscales/upscales
                    "num_c": [4,8,16,32], #Number of channels in the blocks at different scales
                    "num_repeat": [1,2,2,4], #Number of repetitions of blocks at different scales
                    "expand_ratio": [1,4,4,6], #EffecientNetv2 expansion ratio in the blocks at different scales
                    "SE": [0,0,1,1]}
            args_mod["training"]["batch"] = 2
        
        if name_override is not None:
            args_mod["name"] = name_override
        else:
            args_mod["name"] = name
        
        true_  = lambda *_: True
        for key1, value1 in args_mod.items():
            if type(value1)==dict:
                for key2, value2 in value1.items():
                    if type(value2)==dict:
                        for key3, value3 in value2.items():
                            if true_(args_dict[key1][key2][key3]):
                                args_dict[key1][key2][key3] = value3
                    else:
                        if true_(args_dict[key1][key2]):
                            args_dict[key1][key2] = value2
            else:
                if true_(args_dict[key1]):
                    args_dict[key1] = value1
            
    args = json.loads(json.dumps(args_dict), object_hook=lambda item: SimpleNamespace(**item))
    return args_dict if dict_mode else args

def update_args(args_namespace):
    args_namespace2 = args_namespace
    args_mod = get_args(dict_mode=True)
    for key1, value1 in args_mod.items():
        if type(value1)==dict:
            for key2, value2 in value1.items():
                if type(value2)==dict:
                    for key3, value3 in value2.items():
                        try:
                            getattr(getattr(getattr(args_namespace2,key1),key2),key3)
                        except AttributeError:
                            setattr(getattr(getattr(args_namespace2,key1),key2),key3,value3)
                            print("WARNING: added args."+key1+"."+key2+"."+key3+"="+str(value3))
                else:       
                    try:
                        getattr(getattr(args_namespace2,key1),key2)
                    except AttributeError:
                        setattr(getattr(args_namespace2,key1),key2,value2)
                        print("WARNING: added args."+key1+"."+key2+"="+str(value2))
        else:           
            try:
                getattr(args_namespace2,key1)
            except AttributeError:
                setattr(args_namespace2,key1,value1)
                print("WARNING: added args."+key1+"="+str(value1))
    
    return args_namespace2
