""" 
    Refined Vision Transformer (Refined-ViT) in PyTorch
    Full training hyper-parameters will be released.
"""
import torch
import torch.nn as nn
from functools import partial
from torch.nn.parameter import Parameter

# Need to install timm library first
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from .refiner_utils import DLA, ClassBlock
from torch.nn import functional as F

import numpy as np
import math


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # patch models
    'Refined_vit_small_patch16_224': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        crop_pct=.9,
    ),
    'Refined_vit_medium_patch16_224': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    'Refined_vit_large_patch16_224': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        crop_pct=1.13,
    ),

}

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., use_nes=False, expansion_ratio=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.scale_channel = nn.Parameter(1e-5*torch.ones(in_features))
        if use_nes:
            self.fc1 = WSLinear_fast(in_features, hidden_features, multiplier1=2, multiplier2=2)
            self.fc2 = WSLinear_fast(hidden_features, out_features, multiplier1=2, multiplier2 = 2)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x: (B,N,C)
        x = self.scale_channel.unsqueeze(0).unsqueeze(0)*x
        x = self.drop(x)
        return x
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.act = act_layer()
#         self.drop = nn.Dropout(drop)
# 
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

class Refined_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,expansion_ratio = 3, 
                        share_atten=False, apply_transform=True, refine_mode='conv_exapnd', kernel_size=3, head_expand=None):
        """
            refine_mode: "conv" represents only convolution is used for refining the attention map;
                         "conv-expand" represents expansion and conv are used together for refining the attention map; 
            share_atten: If set True, the attention map is not generated; use the attention map generated from the previous block
        """
        super().__init__()
        self.num_heads = num_heads
        self.share_atten = share_atten
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        
        self.scale = qk_scale or head_dim ** -0.5

        if self.share_atten:
            self.DLA = DLA(self.num_heads,self.num_heads, refine_mode=refine_mode)
            self.adapt_bn = nn.BatchNorm2d(self.num_heads)

            self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        elif apply_transform:
            self.DLA = DLA(self.num_heads,self.num_heads, kernel_size=kernel_size, refine_mode=refine_mode, expand_ratio=head_expand)
            self.adapt_bn = nn.BatchNorm2d(self.num_heads)
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, atten=None):
        B, N, C = x.shape
        if self.share_atten:
            attn = atten
            attn = self.adapt_bn(self.DLA(attn)) * self.scale 

            v = self.qkv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn_next = atten
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   

            q = math.sqrt(self.scale)*q
            k = math.sqrt(self.scale)*k
            attn = (q @ k.transpose(-2, -1)) # * self.scale
            attn = attn.softmax(dim=-1) + atten * self.scale if atten is not None else attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if self.apply_transform:
                attn = self.adapt_bn(self.DLA(attn))  
            attn_next = attn
        x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[-1], C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, expansion=3, 
                 group = False, share = False, share_atten=False, bs=False, expand_token=None, 
                 stride = 192, mode = 'overlap', apply_transform=False, head_expand=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.share_atten = share_atten
        # out_dim = dim if out_dim is None else out_dim
        self.expand_token = expand_token
        self.adjust_ratio = 0.5
        self.dim = dim
        self.attn = Refined_Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
                expansion_ratio = expansion, share_atten = share, apply_transform=apply_transform, head_expand=head_expand)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x, atten=None):
        x_new, atten = self.attn(self.norm1(x * self.adjust_ratio), atten)
        x = x + self.drop_path(x_new/self.adjust_ratio)
        x = x + self.drop_path(self.mlp(self.norm2(x * self.adjust_ratio))) / self.adjust_ratio
        return x, atten

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)#.flatten(2).transpose(1, 2)
        return x

class PatchEmbed_cnn(nn.Module):
    """ 
        Use three CNN layers for patch processing. Refer to T2T-ViT for more details.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,spp=42):
        super().__init__()

        new_patch_size = (patch_size // 2, patch_size // 2)

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(128)

        self.proj = nn.Conv2d(128, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, H, W]

        return x


class HybridEmbed(nn.Module):
    """ 
        CNN Feature Map Embedding
        Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

def get_points_single(size, stride=1, dtype=np.float32):
    """The vanilla version of positional encoding (2 channels)."""

    height, width = size
    x_range = np.arange(0, width * stride, stride, dtype=dtype)
    y_range = np.arange(0, height * stride, stride, dtype=dtype)
    y_channel, x_channel = np.meshgrid(y_range, x_range)
    points = (np.stack((x_channel, y_channel)) + stride // 2).transpose((1, 2, 0))
    points = (points - points.mean(axis=(0, 1))) / points.std(axis=(0, 1))
    return torch.cat((torch.zeros((1, 2)), torch.tensor(points.transpose((2, 1, 0)).reshape(height * width, -1))))
    # return points.transpose((2, 1, 0))

def Position_embedding(size, stride = 1):
    return get_points_single(size)

class Refiner_ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, group = False, share_atten=False, cos_reg = False,
                 use_cnn_embed=True, apply_transform=None, interpolate_pos_embedding=False, head_expand=3,
                 mix_token=True, return_dense=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.interpolate_pos_embedding = interpolate_pos_embedding
        # use cosine similarity as a regularization term
        self.cos_reg = cos_reg

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if use_cnn_embed:
                self.patch_embed = PatchEmbed_cnn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, 196 + 1, embed_dim)) # only used for fintuning
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        d = depth if isinstance(depth, int) else len(depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, d)]  # stochastic depth decay rule
        if isinstance(depth, int):
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group = group)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, share=depth[i], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    share_atten=share_atten, apply_transform=apply_transform[i], head_expand=head_expand)
                for i in range(len(depth))])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.return_dense=return_dense
        self.mix_token=mix_token
        if return_dense:
            self.aux_head=nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def resize_pos_embed(self, x, h, w):
        B,_,C = x.size()
        ct = x[:,0].unsqueeze(2)
        ts = x[:,1:].transpose(1, 2).reshape(B, C, 14, 14)
        ts = F.interpolate(ts, (h, w), mode='bilinear', align_corners=True)
        ts = ts.flatten(2)
        x = torch.cat([ct, ts], dim=2).transpose(1, 2)
        return x
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    def forward_cls(self, x):
        B,N,C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x
    def forward_features(self, x):
        if self.cos_reg:
            atten_list = []
        B = x.shape[0]
        x = self.patch_embed(x)

        # dense prediction and relabbeling from Zihang
        patch_h, patch_w = 0,0
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[2],x.shape[3]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            temp_x = x.clone()
            temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0,0,0,0

        if self.interpolate_pos_embedding:
            # used for calculating dim for pos_embed interpolation
            B, C, H, W = x.size()
        x = x.flatten(2).transpose(1, 2) # B, N, C
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.interpolate_pos_embedding:
            x = x + self.resize_pos_embed(self.pos_embed, H, W)
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        attn = None
        for blk in self.blocks:
            x, attn = blk(x, attn)
            if self.cos_reg:
                atten_list.append(attn)
        # x = self.forward_cls(x)
        x = self.norm(x)
        if self.cos_reg and self.training:
            return x, (bbx1, bby1, bbx2, bby2), patch_h, patch_w, atten_list
        else:
            return x, (bbx1, bby1, bbx2, bby2), patch_h, patch_w

    def forward(self, x):
        if self.cos_reg and self.training:
            x, (bbx1, bby1, bbx2, bby2), patch_h, patch_w, atten = self.forward_features(x)
            x_cls = self.head(x[:, 0])
            if self.return_dense:
                x_aux = self.aux_head(x[:,1:])
                if not self.training:
                    return x_cls
                if self.mix_token and self.training:
                    x_aux = x_aux.reshape(x_aux.shape[0],patch_h, patch_w,x_aux.shape[-1])

                    temp_x = x_aux.clone()
                    temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                    x_aux = temp_x

                    x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1])
                # import pdb; pdb.set_trace()
                return (x_cls, x_aux, (bbx1, bby1, bbx2, bby2)), atten
            return x_cls, atten
        else:
            x, (bbx1, bby1, bbx2, bby2), patch_h, patch_w, = self.forward_features(x)
            x_cls = self.head(x[:, 0])
            if self.return_dense:
                x_aux = self.aux_head(x[:,1:])
                if not self.training:
                    return x_cls + 0.5*x_aux.max(1)[0]
                if self.mix_token and self.training:
                    x_aux = x_aux.reshape(x_aux.shape[0],patch_h, patch_w,x_aux.shape[-1])

                    temp_x = x_aux.clone()
                    temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                    x_aux = temp_x

                    x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1])
                return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
            return x_cls


@register_model
def Refiner_ViT_S(pretrained=False, **kwargs):
    apply_transform = [False] * 16 + [True] * 0
    stage = [False,True] * 8
    model = Refiner_ViT(
        patch_size=16, embed_dim=384, depth=stage, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), share_atten=True, head_expand=6, **kwargs)
    model.default_cfg = default_cfgs['Refined_vit_small_patch16_224']
    return model

@register_model
def Refiner_ViT_M(pretrained=False, **kwargs):
    apply_transform = [False] * 32 + [True] * 0
    stage = [False,True] * 16
    model = Refiner_ViT(
        patch_size=16, embed_dim=420, depth=stage, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), share_atten=True, head_expand=3, **kwargs)
    model.default_cfg = default_cfgs['Refined_vit_medium_patch16_224']
    return model

@register_model
def Refiner_ViT_L(pretrained=False, **kwargs):
    apply_transform = [False] * 32 + [True] * 0
    stage = [False,True] * 16
    model = Refiner_ViT(
        patch_size=16, embed_dim=512, depth=stage, apply_transform=apply_transform, num_heads=16, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), share_atten=True, head_expand=3, **kwargs)
    model.default_cfg = default_cfgs['Refined_vit_large_patch16_224']
    return model
