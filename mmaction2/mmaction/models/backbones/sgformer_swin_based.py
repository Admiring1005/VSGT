from functools import lru_cache, reduce
from operator import mul
from typing import Dict, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint

from mmaction.registry import MODELS

try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

import warnings
 
warnings.filterwarnings("ignore")

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x

class Mlp(BaseModule):
    """Multilayer perceptron.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features.
            Defaults to None.
        out_features (int, optional): Number of output features.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_cfg: Dict = dict(type='GELU'),
                 drop: float = 0.,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        """Forward function."""
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, D, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x: torch.Tensor,
                     window_size: Sequence[int],
                    D, H, W) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): The input features of shape :math:`(B, num_heads, D*H*W, C)`.
        window_size (Sequence[int]): The window size, :math:`(w_d, w_h, w_w)`.

    Returns:
        torch.Tensor: The partitioned windows of shape
            :math:`(B*num_windows, w_d*w_h*w_w, C)`.
    """
    
    B, num_heads, N, C = x.shape
    x = x.contiguous().view(B*num_heads, N, C).contiguous().view(B*num_heads, D, H, W, C)
    B, D, H, W, C = x.shape
    wd, wh, ww = window_size
    # print('[DEBUG] ', B, D, H, W, C)
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows  #(B*numheads*num_windows, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: Sequence[int], D: int, H: int, W: int, head: int) -> torch.Tensor:
    """
    Args:
        windows (torch.Tensor): Input windows of shape
            :meth:`(B*num_windows, w_d, w_h, w_w, C)`.
        window_size (Sequence[int]): The window size, :meth:`(w_d, w_h, w_w)`.
        D (int): Temporal length of feature maps.
        H (int): Height of feature maps.
        W (int): Width of feature maps.
        head (int)

    Returns:
        torch.Tensor: The feature maps reversed from windows of
            shape :math:`(B, D, H, W, C)`.
    """
    wd, wh, ww = window_size
    Bhead = int(windows.shape[0] / (D * H * W / wd / wh / ww))
    x = windows.view(Bhead, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(Bhead, D, H, W, -1).view(Bhead//head, head, D, H, W, -1)\
        .contiguous().permute(0,2,3,4,1,5).contiguous().view(Bhead//head, D, H, W, -1).view(Bhead//head, D*H*W, -1)
    return x
    
    
# class PatchMerging(BaseModule):
#     """Patch Merging Layer.

#     Args:
#         embed_dims (int): Number of input channels.
#         norm_cfg (dict): Config dict for norm layer.
#             Defaults to ``dict(type='LN')``.
#         init_cfg (dict, optional): Config dict for initialization.
#             Defaults to None.
#     """

#     def __init__(self,
#                  embed_dims: int,
#                  norm_cfg: Dict = dict(type='LN'),
#                  init_cfg: Optional[Dict] = None) -> None:
#         super().__init__(init_cfg=init_cfg)
#         self.embed_dims = embed_dims
#         # print('[DEBUG] embed_dims ', self.embed_dims)
#         self.mid_embed_dims = 4 * embed_dims
#         self.out_embed_dims = 2 * embed_dims
#         self.reduction = nn.Linear(
#             self.mid_embed_dims, self.out_embed_dims, bias=False)
#         self.norm = build_norm_layer(norm_cfg, self.mid_embed_dims)[1]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Perform patch merging.

#         Args:
#             x (torch.Tensor): Input feature maps of shape
#                 :math:`(B, D, H, W, C)`.

#         Returns:
#             torch.Tensor: The merged feature maps of shape
#                 :math:`(B, D, H/2, W/2, 2*C)`.
#         """
#         B, D, H, W, C = x.shape
#         # print("[DEBUG] before PatchMerging: ", x.shape)
#         # padding
#         pad_input = (H % 2 == 1) or (W % 2 == 1)
#         if pad_input:
#             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

#         x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
#         x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
#         x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
#         x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x)

#         return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        out_dim = dim*2
        self.out_dim = out_dim
        self.act = nn.GELU()
        self.conv1 = Conv3d_BN(dim, out_dim, (1,1,1), (1,1,1), (0,0,0))
        self.conv2 = Conv3d_BN(out_dim, out_dim, (1,3,3), (1,2,2), (0,1,1), groups=out_dim)
        self.conv3 = Conv3d_BN(out_dim, out_dim, (1,1,1), (1,1,1), (0,0,0))

    def forward(self, x):
        # x B C D H W 
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        # output [B D*H*W C]
        return x, D, H, W

class Conv3d_BN(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 4, 4), stride=(2, 4, 4), padding=(0, 1, 1),
                dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        bn = nn.GroupNorm(1, out_channels)#torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class Head(BaseModule):
    def __init__(self, n):
        super(Head, self).__init__()
        self.conv = nn.Sequential(
            Conv3d_BN(3, n, (1,3,3), (1,2,2), (0,1,1)),
            nn.GELU(),
            Conv3d_BN(n, n, (3,3,3), (2,1,1), (1,1,1)),
            nn.GELU(),
            Conv3d_BN(n, n, (1,3,3), (1,2,2), (0,1,1)),
        )
        self.norm = nn.LayerNorm(n)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, D, H, W
    
# class PatchEmbed3D(BaseModule):
#     """Video to Patch Embedding.

#     Args:
#         patch_size (Sequence[int] or int]): Patch token size.
#             Defaults to ``(2, 4, 4)``.
#         in_channels (int): Number of input video channels. Defaults to 3.
#         embed_dims (int): Dimensions of embedding. Defaults to 128.
#         conv_cfg: (dict): Config dict for convolution layer.
#             Defaults to ``dict(type='Conv3d')``.
#         norm_cfg (dict, optional): Config dict for norm layer.
#             Defaults to None.
#         init_cfg (dict, optional): Config dict for initialization.
#             Defaults to None.
#     """

#     def __init__(self,
#                  patch_size: Union[Sequence[int], int] = (2, 4, 4),
#                  in_channels: int = 3,
#                  embed_dims: int = 128,
#                  norm_cfg: Optional[Dict] = None,
#                  conv_cfg: Dict = dict(type='Conv3d'),
#                  init_cfg: Optional[Dict] = None) -> None:
#         super().__init__(init_cfg=init_cfg)
#         self.patch_size = patch_size
#         self.in_channels = in_channels
#         self.embed_dims = embed_dims

#         # print(embed_dims)
        
#         self.proj = build_conv_layer(
#             conv_cfg,
#             in_channels,
#             embed_dims,
#             kernel_size=patch_size,
#             stride=patch_size)

#         if norm_cfg is not None:
#             self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
#         else:
#             self.norm = None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Perform video to patch embedding.

#         Args:
#             x (torch.Tensor): The input videos of shape
#                 :math:`(B, C, D, H, W)`. In most cases, C is 3.

#         Returns:
#             torch.Tensor: The video patches of shape
#                 :math:`(B, embed_dims, Dp, Hp, Wp)`.
#         """
#         _, _, D, H, W = x.size()
#         # 将D，W和H通过pad操作，扩充为patch_size的整数
#         if W % self.patch_size[2] != 0:
#             x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
#         if H % self.patch_size[1] != 0:
#             x = F.pad(x,
#                       (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
#         if D % self.patch_size[0] != 0:
#             x = F.pad(x, (0, 0, 0, 0, 0,
#                           self.patch_size[0] - D % self.patch_size[0]))
#         # 使用3D卷积将x分成大小为128的patch
#         x = self.proj(x)  # B C Dp Wp Wp
#         if self.norm is not None:
#             Dp, Hp, Wp = x.size(2), x.size(3), x.size(4)
#             x = x.flatten(2).transpose(1, 2)  # B Dp*Hp*Wp C
#             x = self.norm(x)
#             x = x.transpose(1, 2).view(-1, self.embed_dims, Dp, Hp, Wp)
#         # 最后x=[B, C, Dp, Hp, Wp]

#         return x

def local_conv(dim):
    return nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
    
#TODO：【核心】SGAttention设计，参照sgformer的Attention，swin的WindowAttention3D
class SGAttention(BaseModule):
    def __init__(self, dim, mask, window_size=(8,7,7), num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio=sr_ratio
        if sr_ratio>1:
            if mask:
                # Significance-Guided Transformer Block
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                if self.sr_ratio==8:
                    f1, f2, f3 = 14*14, 56, 28
                elif self.sr_ratio==4:
                    f1, f2, f3 = 49, 14, 7
                elif self.sr_ratio==2:
                    f1, f2, f3 = 2, 1, None
                self.f1 = nn.Linear(f1, 1)
                self.f2 = nn.Linear(f2, 1)
                if f3 is not None:
                    self.f3 = nn.Linear(f3, 1)
            else:
                # Multi-Scale Transformer Block
                self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
                self.act = nn.GELU()

                self.q1 = nn.Linear(dim, dim//2, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.q2 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = local_conv(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, D, H, W, mask):
        # x: [B,D*H*W,C]
        B, N, C = x.shape
        # B, N, C = x.shape
        lepe = self.lepe_conv(
            self.lepe_linear(x).transpose(1, 2).contiguous().view(B, C, D, H, W)).view(B, C, -1).transpose(-1, -2).contiguous()
        if self.sr_ratio > 1:
            if mask is None:
                # global
                q1 = self.q1(x).reshape(B, N, self.num_heads//2, C // self.num_heads).permute(0, 2, 1, 3)
                x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
                #TODO 修改sr卷积
                x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_1 = self.act(self.norm(x_1))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1] #B head N C

                attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale #B head Nq Nkv
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).contiguous().reshape(B, N, C//2)

                global_mask_value = torch.mean(attn1.detach().mean(1), dim=1) # B Nk  #max ?  mean ?
                #RECORD：增加D维度
                global_mask_value = F.interpolate(global_mask_value.view(B,1,D//self.sr_ratio,H//self.sr_ratio,W//self.sr_ratio),
                                                  (D, H, W), mode='nearest')[:, 0]

                # local
                q2 = self.q2(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3) #B head N C
                kv2 = self.kv2(x_.reshape(B, C, -1).permute(0, 2, 1)).reshape(B, -1, 2, self.num_heads // 2,
                                                                          C // self.num_heads).permute(2, 0, 3, 1, 4)
                k2, v2 = kv2[0], kv2[1]
                # q_window = 7
                # window_size= 7
                #TODO 【重要】修改成3D窗口划分
                
                q2, k2, v2 = window_partition(q2, self.window_size, D, H, W), window_partition(k2, self.window_size, D, H, W), \
                             window_partition(v2, self.window_size, D, H, W)
                attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
                # (B*numheads*num_windows, wd*wh*ww, wd*wh*ww)
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)

                x2 = (attn2 @ v2)  # B*numheads*num_windows, wd*wh*ww, C   .transpose(1, 2).reshape(B, N, C)
                x2 = window_reverse(x2, self.window_size, D, H, W, self.num_heads // 2)

                wd, wh, ww = self.window_size
                
                local_mask_value = torch.mean(attn2.detach().view(B, self.num_heads//2, D//wd*H//wh*W//ww, wd*wh*ww, wd*wh*ww).mean(1), dim=2)
                local_mask_value = local_mask_value.view(B, D // wd, H // wh, W // ww, wd, wh, ww)
                local_mask_value=local_mask_value.permute(0,1,4,2,5,3,6).contiguous().view(B, D, H, W)

                # mask B H W
                x = torch.cat([x1, x2], dim=-1)
                x = self.proj(x+lepe)
                x = self.proj_drop(x)
                # cal mask
                mask = local_mask_value+global_mask_value
                mask_1 = mask.view(B, D * H * W)
                #TODO 【重要】mask2生成前交换了H和W，是否需要动D
                mask_2 = mask.permute(0, 1, 3, 2).reshape(B, D * H * W)
                mask = [mask_1, mask_2]
            else:
                q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                # mask [local_mask global_mask]  local_mask [value index]  value [B, H, W]
                # use mask to fuse
                mask_1, mask_2 = mask
                mask_sort1, mask_sort_index1 = torch.sort(mask_1, dim=1)
                mask_sort2, mask_sort_index2 = torch.sort(mask_2, dim=1)
                token_num = D * H * W
                if self.sr_ratio == 8:
                    token1, token2, token3 = token_num // (14 * 14), token_num // 56, token_num // 28
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 4:
                    token1, token2, token3 = token_num // 49, token_num // 14, token_num // 7
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 2:
                    token1, token2 = token_num // 2, token_num // 1
                    token1, token2 = token1 // 2, token2 // 2
                if self.sr_ratio==4 or self.sr_ratio==8:
                    p1 = torch.gather(x, 1, mask_sort_index1[:, :token_num // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, token_num // 4:token_num // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3 = torch.gather(x, 1, mask_sort_index1[:, token_num // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C

                    x_ = x.view(B, D, H, W, C).permute(0, 1, 3, 2, 4).reshape(B, D * H * W, C)
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, :token_num // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, token_num // 4:token_num // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3_ = torch.gather(x_, 1, mask_sort_index2[:, token_num // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3_.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C
                elif self.sr_ratio==2:
                    p1 = torch.gather(x, 1, mask_sort_index1[:, :token_num // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, token_num // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                    x_ = x.view(B, D, H, W, C).permute(0, 1, 3, 2, 4).reshape(B, D * H * W, C)
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, :token_num // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, token_num // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                kv1 = self.kv1(seq1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4) # kv B heads N C
                kv2 = self.kv2(seq2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv = torch.cat([kv1, kv2], dim=2)
                k, v = kv[0], kv[1]
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
                x = self.proj(x+lepe)
                x = self.proj_drop(x)
                mask=None

        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
            x = self.proj(x+lepe)
            x = self.proj_drop(x)
            mask=None

        return x, mask
    
class SGFormerBlock3D(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 is_mask: bool,
                 num_heads: int,
                 window_size: Sequence[int] = (8, 7, 7),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.1,
                 proj_drop: float = 0.,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='LN'),
                 with_cp: bool = False,
                 init_cfg: Optional[Dict] = None,
                 sr_ratio: int = 1,
                 linear: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size #TODO: windows_size选取什么值
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        
        _attn_cfg = {
            'dim': embed_dims,
            'mask': is_mask,
            'window_size': self.window_size,
            'num_heads': self.num_heads,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'attn_drop': attn_drop,
            'proj_drop': proj_drop,
            'sr_ratio': sr_ratio,
            'linear': linear
        }
        self.attn = SGAttention(**_attn_cfg)
        
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        
        _mlp_cfg = {
            'in_features': embed_dims,
            'hidden_features': int(embed_dims * mlp_ratio),
            'act_cfg': act_cfg,
            'drop': drop
        }
        self.mlp = Mlp(**_mlp_cfg)
    
    def forward_part1(self, x: torch.Tensor,
                      D: int, H: int, W: int,
                      mask_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function part1."""
        x, mask = self.attn(self.norm1(x), D, H, W, mask_matrix)
        return x, mask
    
    def forward_part2(self, x: torch.Tensor,
                     D: int, H: int, W: int) -> torch.Tensor:
        """Forward function part2."""
        return self.drop_path(self.mlp(self.norm2(x), D, H, W))
        
    def forward(self, x: torch.Tensor,
                D: int, H: int, W: int,
                mask_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO: 重要：实现注意力Block的forward流程，参照swin中的SwinTransformerBlock3D
        """
        Args:
            x0 (torch.Tensor): Input features of shape :math:`(B, D*H*W, C)`.
            mask_matrix (torch.Tensor): Attention mask.
        """
        #TODO: 之后学习checkpoint的用法
        shortcut = x
        if self.with_cp:
            x = checkpoint.checkpoint(self.forward_part1, x, D, H, W, mask_matrix)
        else:
            x, mask = self.forward_part1(x, D, H, W, mask_matrix)
        dpx = self.drop_path(x)
        # print(shortcut.shape, dpx.shape)
        x = shortcut + dpx

        if self.with_cp:
            x = x + checkpoint.checkpoint(self.forward_part2, x, D, H, W)
        else:
            x = x + self.forward_part2(x, D, H, W)

        return x, mask

class BasicLayer(BaseModule):
    """A basic Swin Transformer layer for one stage.

    Args:
        embed_dims (int): Number of feature channels.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (Sequence[int]): Local window size.
            Defaults to ``(8, 7, 7)``.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        attn_drop (float): Attention dropout rate. Defaults to 0.0.
        drop_paths (float or Sequence[float]): Stochastic depth rates.
            Defaults to 0.0.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict, optional): Config dict for norm layer.
            Defaults to ``dict(type='LN')``.
        downsample (:class:`PatchMerging`, optional): Downsample layer
            at the end of the layer. Defaults to None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will
            save some memory while slowing down the training speed.
            Defaults to False.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 depth: int,
                 num_heads: int,
                 window_size: Sequence[int] = (8, 7, 7),
                 sr_ratio: int = 1,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_paths: Union[float, Sequence[float]] = 0.,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='LN'),
                 downsample: Optional[PatchMerging] = None,
                 with_cp: bool = False,
                 init_cfg: Optional[Dict] = None,
                 is_last_stage: bool = False) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.with_cp = with_cp
        self.is_last_stage = is_last_stage

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        # build blocks
        self.blocks = ModuleList()
        for i in range(depth):
            #TODO: SGFormerBlock3D的参数定义
            #TODO: 关键，需要间隔设置mask参数为True和False，以标识不同类型的Block
            _block_cfg = {
                'embed_dims': embed_dims,
                'is_mask': True if (i%2==1 and not is_last_stage) else False,
                'num_heads': num_heads,
                'window_size': window_size,
                'mlp_ratio': mlp_ratio,
                'qkv_bias': qkv_bias,
                'qk_scale': qk_scale,
                'drop': drop,
                'attn_drop': attn_drop,
                'drop_path': drop_paths[i],
                'act_cfg': act_cfg,
                'norm_cfg': norm_cfg,
                'with_cp': with_cp,
                'sr_ratio': sr_ratio,
            }

            block = SGFormerBlock3D(**_block_cfg)
            self.blocks.append(block)
        self.norm = nn.LayerNorm(embed_dims)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(
                dim=embed_dims)

    def forward(self,
                x: torch.Tensor,
                D: int, H: int, W: int,
                do_downsample: bool = True) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input feature maps of shape
                :math:`(B, D*H*W, C)`.
            注：这里的x已经分成patch，C代表patch的长度，DHW为x生成patch后三个维度的值
            do_downsample (bool): Whether to downsample the output of
                the current layer. Defaults to True.
        """
        # calculate attention mask for SW-MSA
        B, _, C = x.shape
        attn_mask = None
        # x0 = rearrange(x0, 'b c d h w -> b d h w c')
        # # print(x0.shape)
        # B, D, H, W, C = x0.shape
        # x = x0.permute(0,4,1,2,3).view(B,C,D*H*W).transpose(-1,-2)
        #TODO: compute_mask计算的是用于局部注意力的掩码，本模型不采用该函数
        # x [B,D*H*W,C]
        for blk in self.blocks:
            x, attn_mask  = blk(x, D, H, W, attn_mask)
        # if not self.is_last_stage:
        # x [B,D*H*W,C]
        x = self.norm(x)
        # x = x.reshape(B, D, H, W, -1).contiguous()
        # x = rearrange(x0, 'b d h w c -> b c d h w')
        if self.downsample is not None and do_downsample:
            x = self.downsample(x)
        return x

    @property
    def out_embed_dims(self):
        if self.downsample is not None:
            return self.downsample.out_embed_dims
        else:
            return self.embed_dims


@MODELS.register_module()
class SGFormer3D(BaseModule):
    """Video SG Transformer backbone"""
    
    # arch_zoo = {
    #     **dict.fromkeys(['t', 'tiny'],
    #                     {'embed_dims': 96,
    #                      'depths': [2, 2, 6, 2],
    #                      'num_heads': [3, 6, 12, 24]}),
        # **dict.fromkeys(['s', 'small'],
        #                 {'embed_dims': 96,
        #                  'depths': [2, 2, 18, 2],
        #                  'num_heads': [3, 6, 12, 24]}),
        # **dict.fromkeys(['b', 'base'],
        #                 {'embed_dims': 128,
        #                  'depths': [2, 2, 18, 2],
        #                  'num_heads': [4, 8, 16, 32]}),
        # **dict.fromkeys(['l', 'large'],
        #                 {'embed_dims': 192,
        #                  'depths': [2, 2, 18, 2],
        #                  'num_heads': [6, 12, 24, 48]}),
    # }
    
    def __init__(
        self,
        # SG-Former
        img_size: int = 224,
        linear: bool = False,
        # num_patches: int = img_size//4,
        
        # 核心构建参数
        sr_ratios: Sequence[int] = [8, 4, 2, 1],
        embed_dims: int = 128,
        num_heads: Sequence[int] = [2, 4, 8, 16],
        depths: Sequence[int] = [2, 4, 16, 1], # depth：每个stage的Block数量
        
        # 加载2D权重，执行inflate得到3D的预训练权重
        pretrained: Optional[str] = None,
        pretrained2d: bool = False,
        # 原swin-transformer是4(然后tuple到4x4),而这里是4x4x4,多了一个时间维度
        patch_size: Union[int, Sequence[int]] = (2, 4, 4),
        in_channels: int = 3,
        window_size: Sequence[int] = (8, 7, 7),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        act_cfg: Dict = dict(type='GELU'),
        norm_cfg: Dict = dict(type='LN'),
        patch_norm: bool = True,
        frozen_stages: int = -1,
        with_cp: bool = False,
        out_indices: Sequence[int] = (3, ),
        out_after_downsample: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d

        # if isinstance(arch, str):
        #     arch = arch.lower()
        #     assert arch in set(self.arch_zoo), \
        #         f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
        #     self.arch_settings = self.arch_zoo[arch]
        # else:
        #     essential_keys = {'embed_dims', 'depths', 'num_heads'}
        #     assert isinstance(arch, dict) and set(arch) == essential_keys, \
        #         f'Custom arch needs a dict with keys {essential_keys}'
        #     self.arch_settings = arch

        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.sr_ratios = sr_ratios
        
        assert len(self.depths) == len(self.num_heads)
        self.num_layers = len(self.depths)
        assert 1 <= self.num_layers <= 4
        self.out_indices = out_indices
        assert max(out_indices) < self.num_layers
        self.out_after_downsample = out_after_downsample
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        #TODO PatchEmbed3D -> Head
        # _patch_cfg = {
        #     'patch_size': patch_size,
        #     'in_channels': in_channels,
        #     'embed_dims': self.embed_dims,
        #     'norm_cfg': norm_cfg if patch_norm else None,
        #     'conv_cfg': dict(type='Conv3d')
        # }
        # self.patch_embed = PatchEmbed3D(**_patch_cfg)
        self.patch_embed = Head(self.embed_dims)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        # build layers
        self.layers = ModuleList()
        embed_dims = [self.embed_dims]
        cur_dims = self.embed_dims
        for i, (depth, num_heads, sr_ratio) in \
                enumerate(zip(self.depths, self.num_heads, self.sr_ratios)):
            # print(depth, num_heads, sr_ratio)
            if i >= 1:
                pass
            
            downsample = PatchMerging if i < self.num_layers - 1 else None
            _layer_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'sr_ratio': sr_ratio,
                'window_size': window_size,
                'mlp_ratio': mlp_ratio,
                'qkv_bias': qkv_bias,
                'qk_scale': qk_scale,
                'drop': drop_rate,
                'attn_drop': attn_drop_rate,
                'drop_paths': dpr[:depth],
                'act_cfg': act_cfg,
                'norm_cfg': norm_cfg,
                'downsample': downsample,
                'with_cp': with_cp,
                'is_last_stage': False if i < self.num_layers - 1 else True
            }

            layer = BasicLayer(**_layer_cfg)
            self.layers.append(layer)

            #TODO
            # embed_dims.append(layer.out_embed_dims)
            cur_dims = cur_dims * 2
            embed_dims.append(cur_dims)

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        # for i in out_indices:
        #     if norm_cfg is not None:
        #         norm_layer = build_norm_layer(norm_cfg,
        #                                       self.num_features[i])[1]
        #     else:
        #         norm_layer = nn.Identity()

            # self.add_module(f'norm{i}', norm_layer)
        
        #TODO _freeze_stages实现
        # self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the sgformer2d parameters to sgformer3d.
        
        Args:
            logger (MMLogger): The logger used to print debugging information.
        """
        checkpoint = _load_checkpoint(self.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict_ema']
        # print(state_dict.keys())
        # for k, v in self.state_dict().items():
        #     print(f'{k}\t\t{v.shape}')
        # 参照videoswin，不使用position embedding，因为实验效果提升有限
        patch_embed_keys = [
            k for k in state_dict.keys() if 'patch_embed' in k
        ]
        norm_keys = [
            k for k in state_dict.keys() if k.startswith('norm')
        ]
        block_keys = [
            k for k in state_dict.keys() if 'block' in k
        ]
        
        for k in patch_embed_keys:
            k_parts = k.split('.')
            patch_idx = int(k_parts[0][-1])
            name_current = ""
            if patch_idx == 1:
                # patch_embed
                k_parts[0] = "patch_embed"
                name_current = ".".join(k_parts)
                
            else:
                # patch_merging
                k_parts[0] = f'layers.{patch_idx-2}.downsample'
                name_current = ".".join(k_parts)
            pretrained_2d_weight = state_dict[k]
            current_3d_weight = self.state_dict()[name_current]
            if pretrained_2d_weight.ndim != current_3d_weight.ndim:
                # repeat
                dim2_num = current_3d_weight.shape[2]
                state_dict[name_current] = pretrained_2d_weight.unsqueeze(2).\
                repeat(1,1,dim2_num,1,1) / dim2_num
            else:
                state_dict[name_current] = pretrained_2d_weight
            del state_dict[k]
        
        for k in block_keys:
            k_parts = k.split('.')
            block_idx = int(k_parts[0][-1])
            k_parts[0] = f"layers.{block_idx-1}.blocks"
            name_current = ".".join(k_parts)
            pretrained_2d_weight = state_dict[k]
            current_3d_weight = self.state_dict()[name_current]
            if pretrained_2d_weight.ndim != current_3d_weight.ndim:
                # repeat
                dim2_num = current_3d_weight.shape[2]
                state_dict[name_current] = pretrained_2d_weight.unsqueeze(2).\
                repeat(1,1,dim2_num,1,1) / dim2_num
            else:
                state_dict[name_current] = pretrained_2d_weight
            del state_dict[k]
        
        for k in norm_keys:
            k_parts = k.split('.')
            norm_idx = int(k_parts[0][-1])
            k_parts[0] = f"layers.{norm_idx-1}.norm"
            name_current = ".".join(k_parts)
            state_dict[name_current] = state_dict[k]
            del state_dict[k]

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
    def process_weights(self, logger: MMLogger) -> None:
        checkpoint = _load_checkpoint(self.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # print('[DEBUG] state_dict.keys(): ', state_dict.keys())
        backbone_keys = [
            k for k in state_dict.keys() if 'backbone.' in k
        ]
        for k in backbone_keys:
            k_parts = k.split('.')
            name_current = ".".join(k_parts[1:])
            # print(name_current)
            state_dict[name_current] = state_dict[k]
            del state_dict[k]
        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
    def init_weights(self) -> None:
        """Initialize the weights in backbone."""
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            # Inflate 2D model into 3D model.
            self.inflate_weights(logger)
        else:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            self.process_weights(logger)
            # if self.pretrained:
            #     self.init_cfg = dict(
            #         type='Pretrained', checkpoint=self.pretrained)
            # super().init_weights()

    def forward(self, x: torch.Tensor) -> \
            Union[Tuple[torch.Tensor], torch.Tensor]:
        """Forward function for SGFormer3D Transformer."""
        # print('[DEBUG]', x.shape)
        x, D, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        B, _, _ = x.shape
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous(), D, H, W, do_downsample=self.out_after_downsample)
            # x: [B D*H*W C]
            # if i in self.out_indices:
            #     # norm_layer = getattr(self, f'norm{i}')
            #     # out = norm_layer(x)
            #     # out = rearrange(out, 'b d h w c -> b c d h w').contiguous()
            #     # out = x.transpose(-1,-2).view(B, -1, D, H, W).contiguous()
            #     outs.append(out)

            # if i != self.num_layers - 1:
            x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
            if i in self.out_indices:
                outs.append(x)
                
            if layer.downsample is not None and not self.out_after_downsample:
                x, D, H, W = layer.downsample(x)
            # x [B C D H W]
            # if i == self.num_layers - 1:
            #     x = rearrange(x, 'b c d h w -> b d h w c')
        # print('[DEBUG] out.shape', outs[0].shape)
        # return x
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(SGFormer3D, self).train(mode)
        # self._freeze_stages()
    

if mmdet_imported:
    MMDET_BACKBONES.register_module()(SGFormer3D)