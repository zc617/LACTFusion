import os
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
from torch import Tensor
# from torch.nn import init,Linear,Module,GELU,LayerNorm
import torch.nn.functional as F
import numpy as np
from .conv_fea import *

drop = 0.0
patch_size = 4
stride = 1
padding = 0
embeding = 128
hid_layer = 4
# class Mlp(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.clr_fc = nn.Sequential(
#                                     nn.Linear(in_channels, hidden_channels),
#                                     nn.Dropout(drop),
#                                     nn.GELU(),
#                                     nn.Linear(hidden_channels, out_channels),
#                                     nn.Dropout(drop)
#                                     )
#     def forward(self, x):
#         return self.clr_fc(x)
# class PatchEmbedOverlapping(nn.Module):
#     def __init__(self, patch_size=patch_size, stride=stride, padding=padding, embed_dim=None):
#         super(PatchEmbedOverlapping, self).__init__()
#         self.patch_size = patch_size
#         self.stride = stride
#         self.padding = padding
#         self.embed_dim = embed_dim
#         self.conv = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
#         self.norm = nn.BatchNorm2d(embed_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         B, C, H, W = x.size()
        
#         # Nh = (H - self.patch_size + 2 * self.padding) // self.stride + 1
#         # Nw = (W - self.patch_size + 2 * self.padding) // self.stride + 1

#         if self.padding > 0:
#             x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

#         patches = self.conv(x)

#         patches = self.norm(patches)

#         return patches, H, W

# class ImageRestoration(nn.Module):
#     def __init__(self, patch_size=patch_size, stride=stride, padding=padding, embed_dim=None):
#         super(ImageRestoration, self).__init__()
#         self.patch_size = patch_size
#         self.stride = stride
#         self.padding = padding
#         self.embed_dim = embed_dim
#         self.deconv = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=patch_size, stride=stride, 
#                                          padding=padding, output_padding = 0)

#     def forward(self, patches):
#         B, C, Nh, Nw = patches.size()

#         # Calculate the output tensor size
#         output_H = (Nh - 1) * self.stride + self.patch_size - 2 * self.padding
#         output_W = (Nw - 1) * self.stride + self.patch_size - 2 * self.padding

#         x_reconstructed = self.deconv(patches)
#         x_reconstructed = x_reconstructed[:, :, :output_H, :output_W]

#         return x_reconstructed


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   


class Wave_Fuse(nn.Module):
    def __init__(self, dim, qkv_bias=False, proj_drop=0.,mode='fc'):
        super().__init__()
        
        self.fc_h = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias) 
        self.fc_c = nn.Conv2d(dim, dim, 1, 1,bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, (1,7), stride=1, padding=(0,7//2), groups=dim, bias=False) 
        self.tfc_w = nn.Conv2d(2*dim, dim, (7,1), stride=1, padding=(7//2,0), groups=dim, bias=False)  
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
        self.mode=mode
        
        if mode=='fc':
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 1, 1,bias=True),nn.BatchNorm2d(dim),nn.ReLU())  
        else:
            self.theta_h_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU())
            self.theta_w_conv=nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),nn.BatchNorm2d(dim),nn.ReLU()) 
                    
    def forward(self, ir, vi):
     
        B, C, H, W = ir.shape
        ir_res = ir
        vi_res = vi
        ir_theta_h=self.theta_h_conv(ir)
        ir_theta_w=self.theta_w_conv(ir)

        vi_theta_h=self.theta_h_conv(vi)
        vi_theta_w=self.theta_w_conv(vi)

        ir_h=self.fc_h(ir)
        ir_w=self.fc_w(ir) 

        vi_h=self.fc_h(vi)
        vi_w=self.fc_w(vi) 

        im_h = ir_h * torch.sin(ir_theta_h) + vi_h * torch.sin(vi_theta_h)
        im_w = ir_w * torch.sin(ir_theta_w) + vi_w * torch.sin(vi_theta_w)  # use Imaginary
         
        ir_h = torch.cat([ir_h * torch.cos(ir_theta_h), im_h],dim=1)
        ir_w = torch.cat([ir_w * torch.cos(ir_theta_w), im_w],dim=1)

        vi_h = torch.cat([vi_h * torch.cos(vi_theta_h), im_h],dim=1)
        vi_w = torch.cat([vi_w * torch.cos(vi_theta_w), im_w],dim=1)
    
        ir_h = self.tfc_h(ir_h)
        ir_w = self.tfc_w(ir_w)
        ir_c = self.fc_c(ir_res)
        ir_a = F.adaptive_avg_pool2d(ir_h + ir_w + ir_c,output_size=1)
        ir_a = self.reweight(ir_a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        ir = ir_h * ir_a[0] + ir_w * ir_a[1] + ir_c * ir_a[2] 
        ir = self.proj(ir)
        ir = self.proj_drop(ir) + ir_res 

        vi_h = self.tfc_h(vi_h)
        vi_w = self.tfc_w(vi_w)
        vi_c = self.fc_c(vi_res)
        vi_a = F.adaptive_avg_pool2d(vi_h + vi_w + vi_c,output_size=1)
        vi_a = self.reweight(vi_a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        vi = vi_h * vi_a[0] + vi_w * vi_a[1] + vi_c * vi_a[2] 
        vi = self.proj(vi)
        vi = self.proj_drop(vi) + vi_res

        return ir, vi
    
class WaveBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=hid_layer, qkv_bias=False, drop_path=drop):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        # self.act_layer=nn.GELU()
        self.attn = Wave_Fuse(dim, qkv_bias=qkv_bias, proj_drop=drop_path, mode='fc')
    
    def forward(self, ir, vi):
        ir1 = ir
        vi1 = vi
        ir, vi = self.attn(self.norm1(ir), self.norm1(vi))
        ir = ir + ir1
        vi = vi + vi1
   
        return ir, vi   


    
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()  
        embed_dims = [embeding, embeding, embeding, embeding]
        # self.patch_embeding = PatchEmbedOverlapping(embed_dim=embed_dims[0], padding=padding)
        self.Conv_block = Conv_block(outchannels=embed_dims[0])
        # self.patch_restore = ImageRestoration(embed_dim=embed_dims[3]*2, padding=padding)
        self.wave_fuse = nn.ModuleList([WaveBlock(dim=embed_dim) for embed_dim in embed_dims])
        # self.drop_path = DropPath(drop) if drop > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(embed_dims[0] * 2)
        mlp_hidden_dim = int(embed_dims[0] * hid_layer)
        self.mlp = Mlp(in_features=embed_dims[0] * 2, hidden_features=mlp_hidden_dim, out_features=embed_dims[0] * 2)
        self.reconstruction = Feature_reconstruction(dim= embed_dims[3]*2)

    def forward(self, ir, vi):
        
        # ir, _, _ = self.Conv_block(ir)
        # vi, _, _ = self.Conv_block(vi)
        ir = self.Conv_block(ir)
        vi = self.Conv_block(vi)
        # fea = self.patch_restore(torch.cat((ir, vi), dim=1))
        for i in range (4):
            ir, vi = self.wave_fuse[i](ir, vi)
        # fea = self.patch_restore(fea)
        fea_fuse1 = torch.cat((ir, vi),dim=1)
        fea_fuse1 = self.norm2(fea_fuse1)
        fea = self.mlp(fea_fuse1) + fea_fuse1
        fusion_image = self.reconstruction(fea)

        return fusion_image, ir, vi
   
if __name__ == "__main__":
    x = torch.tensor(np.random.rand(16, 1, 128, 128).astype(np.float32))
    b, c, H, W = x.size()
    img_size = (H, W)
    model = FusionNet()
    y = model(x, x)
    print('test ok!')
    
