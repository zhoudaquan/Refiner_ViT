"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import re
import numpy as np
import pdb
from torch.nn.parameter import Parameter
# from nn.modules import Module

import math
import torch
from torch import nn
from torch.nn import functional as F


########################################################################
######### HELPERS FUNCTIONS FOR Refiner-ViT MODEL ARCHITECTURE #########
########################################################################



def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output
class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim=head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, self.head_dim* self.num_heads * 2, bias=qkv_bias)
        self.q=nn.Linear(dim, self.head_dim* self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim* self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # B,heads,N,C
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x[:,:1,:]).reshape(B,self.num_heads,1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim* self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed
class ClassBlock(nn.Module):

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, group=1, skip_lam=1.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.skip_lam = skip_lam
        self.attn = ClassAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, atten=None):
        # import pdb;pdb.set_trace()
        cls_embed=x[:,:1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))/self.skip_lam
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))/self.skip_lam
        return torch.cat([cls_embed, x[:,1:]],dim=1)
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SpatialConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='same', 
                 reshape_dim = 1, token_num=14, chunk_size = 3):
        super(SpatialConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.padding_mode = padding_mode
        self.reshape_dim = reshape_dim
        self.token_num = token_num
        self.chunk_size = chunk_size
    def forward(self, x):
        shape = x.shape
        if self.reshape_dim == 1:
            input_data = x[:,:,:,1:].reshape(-1, shape[1], shape[2], self.token_num, self.token_num)
            x_cls = x[:,:,:,0:1]
        else:
            # import pdb;pdb.set_trace()
            input_data = x[:,:,1:,:].reshape(-1, shape[1], self.token_num, self.token_num, shape[3]).permute(0,1,4,2,3)
            x_cls = x[:,:,0:1,:]
        weight = self.weight
        if self.padding_mode == 'same':
            id, ih, iw= input_data.size()[-3:]
            kd, kh, kw = self.kernel_size
            sd, sh, sw= self.stride
            oh, ow, od = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(id / sd)
            pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * 1 + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * 1 + 1 - iw, 0)
            pad_d = max((od - 1) * self.stride[2] + (kd - 1) * 1 + 1 - id, 0)
            if pad_h > 0 or pad_w > 0 or pad_d > 0:
                input_data = F.pad(input_data, [pad_d//2, pad_d - pad_d //2, pad_h//2, pad_h - pad_h//2, pad_w//2, pad_w - pad_w//2])
        # import pdb;pdb.set_trace()
        output = F.conv3d(input_data, weight, self.bias, self.stride, self.padding, self.dilation, self.groups).reshape(shape[0], shape[1], shape[2], shape[3] - 1)
        return torch.cat([output.permute(0,1,3,2), x_cls], dim=2) if self.reshape_dim == 0 else torch.cat([output, x_cls], dim=2)
class SpatialConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='same', reshape_dim = 1, token_num=14):
        super(SpatialConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.padding_mode = padding_mode
        self.reshape_dim = reshape_dim
        self.token_num = token_num
    def forward(self, x):
        shape = x.shape
        if self.reshape_dim == 1:
            input_data = x[:,:,:,1:].reshape(-1, shape[1], self.token_num, self.token_num)
            x_cls = x[:,:,:,0:1]
        else:
            # import pdb;pdb.set_trace()
            input_data = x[:,:,1:,:].reshape(-1, shape[1], self.token_num, self.token_num)
            x_cls = x[:,:,0:1,:]
        weight = self.weight
        if self.padding_mode == 'same':
            ih, iw= input_data.size()[-2:]
            kh, kw = self.kernel_size
            sh, sw= self.stride
            oh, ow= math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * 1 + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * 1 + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                input_data = F.pad(input_data, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        # import pdb;pdb.set_trace()
        output = F.conv2d(input_data, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return torch.cat([output.reshape(shape[0], shape[1], shape[2], shape[3] - 1), x_cls], dim=3) if self.reshape_dim == 1 \
                else torch.cat([output.reshape(shape[0], shape[1], shape[2] - 1, shape[3]), x_cls], dim=2)

class SpatialSelfAttention(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, expand_ratio = 6, transform_mode='conv2d'):
        """
            This is used for ablation for adding spatial constraints when refining attention maps.
        """
        super(SpatialSelfAttention, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.inp, self.oup = inp, oup
        self.high_dim_id = False

        if self.expand_ratio != 1:
            self.conv_exp = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
        if transform_mode == 'none':
            self.depth_sep_conv = nn.Conv2d(hidden_dim, hidden_dim, (kernel_size,kernel_size), (stride,stride), (1,1), groups=hidden_dim, bias=False)
        elif transform_mode == 'conv3d':
            self.row_conv = nn.Sequential(
                SpatialConv3d(hidden_dim, hidden_dim, kernel_size, reshape_dim=0, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        elif transform_mode == 'conv2d':
            self.row_conv = nn.Sequential(
                SpatialConv2d(hidden_dim, hidden_dim, kernel_size, reshape_dim=0, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        elif transform_mode == 'conv2+1d':
            self.row_conv = nn.Sequential(
                SpatialConv2d(hidden_dim, hidden_dim, kernel_size, reshape_dim=0, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                SpatialConv2d(hidden_dim, hidden_dim, kernel_size=(kernel_size[0], 1), reshape_dim=1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        # self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv_pro = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        x= input
        if self.expand_ratio !=1:
            x = self.relu(self.bn1(self.conv_exp(x)))
        # x = self.relu(self.bn2(self.depth_sep_conv(x)))
        x = self.row_conv(x) #+ self.col_conv(x)
        x = self.bn3(self.conv_pro(x))
        
        if self.identity:
            return x + input
        else:
            return x
class DLA(nn.Module):
    def __init__(self, inp, oup, kernel_size = 3, stride=1, expand_ratio = 3, refine_mode='none'):
        super(DLA, self).__init__()
        """
            Distributed Local Attention used for refining the attention map.
        """

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.inp, self.oup = inp, oup
        self.high_dim_id = False
        self.refine_mode = refine_mode


        if refine_mode == 'conv':
            self.conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size,kernel_size), stride, (1,1), groups=1, bias=False)
        elif refine_mode == 'conv_exapnd':
            if self.expand_ratio != 1:
                self.conv_exp = Conv2dSamePadding(inp, hidden_dim, 1, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(hidden_dim)   
            self.depth_sep_conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size,kernel_size), stride, (1,1), groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)

            self.conv_pro = Conv2dSamePadding(hidden_dim, oup, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

            self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        x= input
        if self.refine_mode == 'conv':
            return self.conv(x)
        else:
            if self.expand_ratio !=1:
                x = self.relu(self.bn1(self.conv_exp(x)))
            x = self.relu(self.bn2(self.depth_sep_conv(x)))
            x = self.bn3(self.conv_pro(x))
            if self.identity:
                return x + input
            else:
                return x


if __name__ == '__main__':
    import torch
    sp_conv = SpatialSelfAttention(12,12,kernel_size=(3,3))
    data = torch.randn((3,12,197,197))
    output = sp_conv(data)
    print(output.shape)
