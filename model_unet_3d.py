import torch
from torch import nn
from torch.nn import SiLU
import math


class DownScaleLayer(nn.Module):
    def __init__(self, in_c, out_c, mode, factor=2):
        super(DownScaleLayer, self).__init__()
        if mode=="avgpool":
            self.downscale = nn.Sequential(nn.AvgPool3d(factor),
                                           nn.Conv3d(in_c,out_c,1))
        elif mode=="maxpool":    
            self.downscale = nn.Sequential(nn.MaxPool3d(factor),
                                           nn.Conv3d(in_c,out_c,1))
        elif mode=="conv":
            self.downscale = nn.Conv3d(in_c,out_c,factor,factor)
        else:
            raise ValueError("did not recognize mode: "+mode)

    def forward(self, x):
        return self.downscale(x)


class UpScaleLayer(nn.Module):
    def __init__(self, in_c, out_c, mode, factor=2):
        super(UpScaleLayer, self).__init__()
        if mode=="NDlinear":
            mode = 'trilinear'
            
        if mode in ['nearest','trilinear','bilinear']:
            self.upscale = nn.Sequential(nn.Conv3d(in_c,out_c,1),
                                         nn.Upsample(scale_factor=factor,mode=mode,align_corners=False))
        elif mode=="tconv":    
            self.upscale = nn.ConvTranspose3d(in_c,out_c,factor,factor)
        else:
            raise ValueError("did not recognize mode: "+mode)

    def forward(self, x):
        return self.upscale(x)


class SELayer(nn.Module):
    def __init__(self, in_c, hidden_c, reduction=4,act=SiLU()):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(in_c, hidden_c),
                act,
                nn.Linear(hidden_c, in_c),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

def get_act(act_name="relu"):
    if act_name.lower()=="relu":
        act = nn.ReLU()
    elif act_name.lower()=="silu":
        act = SiLU()
    else:
        raise ValueError("Did not recognize activation: "+act_name)
    return act


class MBConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, expand_ratio=4, use_se=False, use_bn=False, act=nn.ReLU(), stride=1):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_c * expand_ratio)
        self.identity = stride == 1 and in_c == out_c
        self.conv = nn.Sequential(*(
            [nn.Conv3d(in_c, hidden_dim, 1, 1, 0, bias=False)]+
            ([nn.BatchNorm3d(hidden_dim)] if use_bn else [])+
            [act]+
            
            [nn.Conv3d(hidden_dim, hidden_dim, k, stride, k//2, groups=hidden_dim, bias=False)]+
            ([nn.BatchNorm3d(hidden_dim)] if use_bn else [])+
            [act]+
            ([SELayer(hidden_dim, in_c//4)] if use_se else [])+
            
            [nn.Conv3d(hidden_dim, out_c, 1, 1, 0, bias=False)]+
            ([nn.BatchNorm3d(out_c)] if use_bn else [])
        ))
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FusedMBConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, expand_ratio=4, use_se=False, use_bn=False, act=nn.ReLU(), stride=1):
        super(FusedMBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_c * expand_ratio)
        self.identity = stride == 1 and in_c == out_c
        
        self.conv = nn.Sequential(*(
                [nn.Conv3d(in_c, hidden_dim, k, stride, k//2, bias=False)]+
                ([nn.BatchNorm3d(hidden_dim)] if use_bn else [])+
                [act]+
                ([SELayer(hidden_dim, in_c//4)] if use_se else [])+
                
                [nn.Conv3d(hidden_dim, out_c, 1, 1, 0, bias=False)]+
                ([nn.BatchNorm3d(out_c)] if use_bn else [])
            ))
    def forward(self, x):
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)


BLOCK_DICT = {"m": MBConv,
              "f": FusedMBConv}


class UNet3D(nn.Module):
    def __init__(self, block="ffmmm",
                       act="silu", 
                       res_mode = "add", #cat, add
                       init_mode = "effecientnetv2",
                       downscale_mode = "avgpool",
                       upscale_mode = "bilinear",
                       input_channels = 1,
                       num_blocks = 5,
                       num_c = [8,16,32,48,64],
                       num_repeat = [1,2,2,4,4], #num_repeat is the number of sequential mb convs
                       expand_ratio = [1,4,4,6,6],
                       SE = [0,0,1,1,1],
                       num_classes = 1):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.res_mode = res_mode
        #block="ffmmm"
        
        assert num_blocks == len(num_repeat)
        assert num_blocks == len(expand_ratio)
        assert num_blocks == len(SE)
        assert num_blocks == len(num_c)
        assert num_blocks == len(block)
        
        config = zip(range(num_blocks),
                     block,
                     num_c,
                     num_repeat,
                     expand_ratio,
                     SE)
        if isinstance(act,str):
            act = get_act(act_name=act)
        
        self.first_conv = nn.Conv3d(input_channels, num_c[0], 3, 1, 1, bias=False)
        self.last_conv = nn.Conv3d(num_c[0], num_classes, 3, 1, 1, bias=False)
        
        DownBlocks = []
        UpBlocks = []
        
        downscales = []
        upscales = []
        
        channels_prev = -1 #Not sure what this exactly do
        
        for i, block_type, channels, num_rep, ratio, use_se in config:
            if channels_prev>0:
                downscales.append(DownScaleLayer(channels_prev,
                                                 channels,
                                                 mode=downscale_mode))
                upscales.append(UpScaleLayer(channels,
                                             channels_prev,
                                             mode=upscale_mode))
        
            DownBlocks.append(nn.Sequential(*[BLOCK_DICT[block_type](channels, 
                                            channels, 
                                            k=3, 
                                            expand_ratio=ratio, 
                                            use_se=use_se, 
                                            use_bn=True, 
                                            act=act, 
                                            stride=1) for _ in range(num_rep)]))
            if res_mode=="cat":
                channels_r0 = channels*2 if i+1 < num_blocks else channels
            else:
                channels_r0 = channels
            
            UpBlocks.append(nn.Sequential(*[BLOCK_DICT[block_type](channels_r0 if r==0 else channels, 
                                            channels, 
                                            k=3, 
                                            expand_ratio=ratio, 
                                            use_se=use_se, 
                                            use_bn=True, 
                                            act=act, 
                                            stride=1) for r in range(num_rep)]))
            
            channels_prev = channels
        
        self.DownBlocks = nn.ModuleList(DownBlocks)
        self.UpBlocks = nn.ModuleList(UpBlocks)
        
        self.downscales = nn.ModuleList(downscales)
        self.upscales = nn.ModuleList(upscales)
        
        
        if init_mode=="effecientnetv2":
            self._initialize_weights_effecientnetv2()

    def _initialize_weights_effecientnetv2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        skip = []
        #print(x.shape)
        x = self.first_conv(x)
        for i in range(self.num_blocks):
            x = self.DownBlocks[i](x)
            
            if i+1 < self.num_blocks:
                skip.append(x)
                x = self.downscales[i](x)
                
        for i in reversed(range(self.num_blocks)):
            if i+1 < self.num_blocks:
                x = self.upscales[i](x)
                if self.res_mode=="cat":
                    # print("x",x.shape)
                    # print("skip", skip.pop().shape)
                    x = torch.cat((x,skip.pop()),dim=1)
                elif self.res_mode=="add":
                    x += skip.pop()
                
            x = self.UpBlocks[i](x)
            
        return self.last_conv(x)


class DownNet3D(nn.Module):
    def __init__(self, block="ffmmm",
                       act="silu",
                       init_mode = "effecientnetv2",
                       downscale_mode = "avgpool",
                       upscale_mode = "trilinear",
                       input_channels = 1,
                       num_blocks = 5,
                       num_c = [8,16,32,48,64],
                       num_repeat = [1,2,2,4,4],
                       expand_ratio = [1,4,4,6,6],
                       SE = [0,0,1,1,1],
                       num_classes = 1,
                       reshape_last = (5,5,5),
                       fc_c = (512,1)):
        super().__init__()
        
        self.fc_c = fc_c
        self.num_blocks = num_blocks
        
        assert num_blocks == len(num_repeat)
        assert num_blocks == len(expand_ratio)
        assert num_blocks == len(SE)
        assert num_blocks == len(num_c)
        assert num_blocks == len(block)
        
        config = zip(range(num_blocks),
                     block,
                     num_c,
                     num_repeat,
                     expand_ratio,
                     SE)
        
        if isinstance(act,str):
            act = get_act(act_name=act)
        
        self.first_conv = nn.Conv3d(input_channels, num_c[0], 3, 1, 1, bias=False)
        self.n_features = num_c[-1]*torch.prod(torch.tensor(reshape_last)).item()
        
        
        
        if len(fc_c)>0:
            fc = [nn.Linear(self.n_features,fc_c[0])]
            self.adaptive_pool = nn.AdaptiveAvgPool3d(reshape_last)
        else:
            fc = [nn.Conv3d(num_c[-1],1,1)]
            
        if len(fc_c)>1:
            for i in range(len(fc_c)-1):
                fc.append(act)
                fc.append(nn.Linear(fc_c[i],fc_c[i+1]))
        
        self.FC = nn.Sequential(*fc)
        
        DownBlocks = []
        
        downscales = []
        
        channels_prev = -1
        
        for i, block_type, channels, num_rep, ratio, use_se in config:
            if channels_prev>0:
                downscales.append(DownScaleLayer(channels_prev,
                                                 channels,
                                                 mode=downscale_mode))
        
            DownBlocks.append(nn.Sequential(*[BLOCK_DICT[block_type](channels, 
                                            channels, 
                                            k=3, 
                                            expand_ratio=ratio, 
                                            use_se=use_se, 
                                            use_bn=True, 
                                            act=act, 
                                            stride=1) for _ in range(num_rep)]))
            
            channels_prev = channels
        
        self.DownBlocks = nn.ModuleList(DownBlocks)
        
        self.downscales = nn.ModuleList(downscales)
        
        if init_mode=="effecientnetv2":
            self._initialize_weights_effecientnetv2()

    def _initialize_weights_effecientnetv2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        skip = []
        
        x = self.first_conv(x)
        for i in range(self.num_blocks):
            x = self.DownBlocks[i](x)
            
            if i+1 < self.num_blocks:
                skip.append(x)
                x = self.downscales[i](x)
                
        if len(self.fc_c)>0:
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            x = self.FC(x)
        else:
            x = self.FC(x)
            x = x.mean()
        
        return x
    
if __name__=="__main__":
    device = "cuda"
    from config import get_args
    
    args = get_args()
    net = UNet3D(**vars(args.unet))
    net = net.to(device)
    
    test_data = torch.rand(2,1,64,64,64).to(device)
    with torch.no_grad():
        output = net(test_data)
    
    print(test_data.shape, output.shape)
    
    """
    net = DownNet3D()
    net = net.to(device)
    with torch.no_grad():
        output = net(test_data)
    
    print(test_data.shape,output.shape)
    """