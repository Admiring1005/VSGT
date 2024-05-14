# from functools import lru_cache, reduce
# from operator import mul
# from typing import Dict, List, Optional, Sequence, Tuple, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# from einops import rearrange
# from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
# from mmcv.cnn.bricks import DropPath
# from mmengine.logging import MMLogger
# from mmengine.model import BaseModule, ModuleList
# from mmengine.model.weight_init import trunc_normal_
# from mmengine.runner.checkpoint import _load_checkpoint

# from mmaction.registry import MODELS

# try:
#     from mmdet.models import BACKBONES as MMDET_BACKBONES
#     mmdet_imported = True
# except (ImportError, ModuleNotFoundError):
#     mmdet_imported = False
    
# #TODO: 3D MLP，直接用swin的，在调用的时候适配参数 
# class Mlp(BaseModule):
#     """Multilayer perceptron.

#     Args:
#         in_features (int): Number of input features.
#         hidden_features (int, optional): Number of hidden features.
#             Defaults to None.
#         out_features (int, optional): Number of output features.
#             Defaults to None.
#         act_cfg (dict): Config dict for activation layer.
#             Defaults to ``dict(type='GELU')``.
#         drop (float): Dropout rate. Defaults to 0.0.
#         init_cfg (dict, optional): Config dict for initialization.
#             Defaults to None.
#     """

#     def __init__(self,
#                  in_features: int,
#                  hidden_features: Optional[int] = None,
#                  out_features: Optional[int] = None,
#                  act_cfg: Dict = dict(type='GELU'),
#                  drop: float = 0.,
#                  init_cfg: Optional[Dict] = None) -> None:
#         super().__init__(init_cfg=init_cfg)
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = build_activation_layer(act_cfg)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward function."""
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# def window_partition(x: torch.Tensor,
#                      window_size: Sequence[int]) -> torch.Tensor:
#     """
#     Args:
#         x (torch.Tensor): The input features of shape :math:`(B, D, H, W, C)`.
#         window_size (Sequence[int]): The window size, :math:`(w_d, w_h, w_w)`.

#     Returns:
#         torch.Tensor: The partitioned windows of shape
#             :math:`(B*num_windows, w_d*w_h*w_w, C)`.
#     """
#     B, D, H, W, C = x.shape
#     x = x.view(B, D // window_size[0], window_size[0], H // window_size[1],
#                window_size[1], W // window_size[2], window_size[2], C)
#     windows = x.permute(0, 1, 3, 5, 2, 4, 6,
#                         7).contiguous().view(-1, reduce(mul, window_size), C)
#     return windows


# def window_reverse(windows: torch.Tensor, window_size: Sequence[int], B: int,
#                    D: int, H: int, W: int) -> torch.Tensor:
#     """
#     Args:
#         windows (torch.Tensor): Input windows of shape
#             :meth:`(B*num_windows, w_d, w_h, w_w, C)`.
#         window_size (Sequence[int]): The window size, :meth:`(w_d, w_h, w_w)`.
#         B (int): Batch size of feature maps.
#         D (int): Temporal length of feature maps.
#         H (int): Height of feature maps.
#         W (int): Width of feature maps.

#     Returns:
#         torch.Tensor: The feature maps reversed from windows of
#             shape :math:`(B, D, H, W, C)`.
#     """
#     x = windows.view(B, D // window_size[0], H // window_size[1],
#                      W // window_size[2], window_size[0], window_size[1],
#                      window_size[2], -1)
#     x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
#     return x



# class Conv2d_BN(torch.nn.Sequential):
#     def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
#                  groups=1, bn_weight_init=1):
#         super().__init__()
#         self.add_module('c', torch.nn.Conv2d(
#             a, b, ks, stride, pad, dilation, groups, bias=False))
#         bn = nn.GroupNorm(1, b)#torch.nn.BatchNorm2d(b)
#         torch.nn.init.constant_(bn.weight, bn_weight_init)
#         torch.nn.init.constant_(bn.bias, 0)
#         self.add_module('bn', bn)

# class Block(nn.Module):
#     def __init__(self, dim, mask, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, mask,
#             num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W, mask):
#         x_, mask = self.attn(self.norm1(x), H, W, mask)
#         x = x + self.drop_path(x_)
#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

#         return x, mask
    
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

#         return x, Dp, Hp, Wp


# #TODO: sg-former的Block，封装了两个Transformer block
# class Block(nn.Module):

#     def __init__(self, dim, mask, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, mask,
#             num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     #TODO: attn，mlp都要增加D维度
#     def forward(self, x, D, H, W, mask):
#         x_, mask = self.attn(self.norm1(x), H, W, mask)
#         x = x + self.drop_path(x_)
#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

#         return x, mask

    
# @MODELS.register_module()
# class SGFormer3D(BaseModule):
#     """Video SG Transformer backbone"""
#     def __init__(
#         self,
        
#         # SG-Former
#         img_size: int = 224,
#         #patch_size
#         #in_chans->in_channels
#         num_stages: int = 4,
#         linear: bool = False,
#         sr_ratios: Sequence[int] = (8,1),
#         num_classes: int = 1000,
#         num_patches: int = img_size//4,
#         embed_dims: Sequence[int] = (128,256,512,1024),
#         num_heads: Sequece[int] = (1,2,4,8),
#         mlp_ratios: Sequence[int] = (4,4,4,4),
#         depths: Sequence[int] = (3,4,6,3), # depth：每个stage的Block数量
        
#         # 加载2D权重，执行inflate得到3D的预训练权重
#         pretrained: Optional[str] = None,
#         pretrained2d: bool = True,
#         patch_size: Union[int, Sequence[int]] = (2, 4, 4),
#         in_channels: int = 3,
#         window_size: Sequence[int] = (8, 7, 7),
#         mlp_ratio: float = 4.,
#         qkv_bias: bool = True,
#         qk_scale: Optional[float] = None,
#         drop_rate: float = 0.,
#         attn_drop_rate: float = 0.,
#         drop_path_rate: float = 0.1,
#         act_cfg: Dict = dict(type='GELU'),
#         norm_cfg: Dict = dict(type='LN'),
#         patch_norm: bool = True,
#         frozen_stages: int = -1,
#         with_cp: bool = False,
#         out_indices: Sequence[int] = (3, ),
#         out_after_downsample: bool = False,
#         init_cfg: Optional[Union[Dict, List[Dict]]] = [
#             dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
#             dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
#         ]
#     ) -> None:
        
        
#         # 第一个Transformer Block前：输入图片，生成patch
#         # 其他Transformer Block：将上一个Block的输出进行PatchMerging降采样
#         cur = 0
#         for i in range(num_stages):
#             if i==0:
#                 patch_embed = PatchEmbed3D(patch_size = (2,4,4))
#             else:
#                 patch_embed = PatchMerging(dim=embed_dims[i - 1],
#                                            out_dim=embed_dims[i])
#             # 每个stage
#             block = nn.ModuleList([Block(
#                 dim=embed_dims[i], mask=True if (j%2==1 and i<num_stages-1) else False, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
#                 sr_ratio=sr_ratios[i], linear=linear)
#                 for j in range(depths[i])])
#             norm = norm_layer(embed_dims[i])
#             cur += depths[i]
#             setattr(self, f"patch_embed{i + 1}", patch_embed)
#             setattr(self, f"block{i + 1}", block)
#             setattr(self, f"norm{i + 1}", norm)
#         # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches*self.num_patches, embed_dims[0]))  # fixed sin-cos embedding
#         # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

#         # self.apply(self._init_weights)
    
#     #TODO: 增加D维度
#     def forward(self, x):
#         B = x.shape[0]
#         mask=None
#         for i in range(self.num_stages):
#             patch_embed = getattr(self, f"patch_embed{i + 1}")
#             block = getattr(self, f"block{i + 1}")
#             norm = getattr(self, f"norm{i + 1}")
#             x, D, H, W = patch_embed(x)
#             if i==0:
#                 x+=self.pos_embed
#             for blk in block:
                
#                 x, mask = blk(x, D, H, W, mask)
#             x = norm(x)
#             if i != self.num_stages - 1:
#                 x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         return x.mean(dim=1)