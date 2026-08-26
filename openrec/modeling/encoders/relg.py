import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import DropPath, Identity, Mlp
import torch.nn.functional as F
import torchvision.ops as ops
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import Fconv_PCA, Fconv_PCA_out

class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.flatten(2, 3).transpose(1, 2)
        
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class FlattenBlockRe2D(Block):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0,
                 attn_drop=0,
                 drop_path=0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 eps=0.000001):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer, eps)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class ConvBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
        num_conv=2,
        kernel_size=3,
        last=False,
    ):
        super().__init__()
        tranNum = 4
        self.last = last
        self.conv1 = Fconv_PCA(sizeP=3, inNum=dim//tranNum, outNum=dim//tranNum, stride=1, tranNum=tranNum, padding=1, ifIni=0)
        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=dim, affine=False)
        self.conv2 = Fconv_PCA(sizeP=3, inNum=dim//tranNum, outNum=dim//tranNum, stride=1, tranNum=tranNum, padding=1, ifIni=0)
        self.bn2 = nn.GroupNorm(num_groups=1, num_channels=dim, affine=False)
        if last==False:
            self.conv3 = Fconv_PCA(sizeP=3, inNum=dim//tranNum, outNum=dim//tranNum, stride=1, tranNum=tranNum, padding=1, ifIni=0)
        else:
            self.conv3 = Fconv_PCA_out(sizeP=3, inNum=dim//tranNum, outNum=dim, stride=1, tranNum=tranNum, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=1, num_channels=dim, affine=False)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        if self.last:
            x += residual
        x = self.act(x)
        # if self.last:
        #     x += residual
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if not self.last:
            x += residual
        x = self.act(x)
        return x

class SubSample2D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
        last_conv=False,
    ):
        super().__init__()
        tranNum = 4
        if last_conv == False:
            self.conv = Fconv_PCA(sizeP=3, inNum=in_channels//tranNum, outNum=out_channels//tranNum, stride=2, tranNum=tranNum, padding=1, ifIni=0)
        else:
            self.conv = Fconv_PCA_out(sizeP=3, inNum=in_channels//tranNum, outNum=out_channels, stride=2, tranNum=tranNum, padding=1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels, affine=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class IdentitySize(nn.Module):

    def forward(self, x):
        return x

class Stage(nn.Module):

    def __init__(self,
                 dim=64,
                 out_dim=256,
                 depth=3,
                 mixer=['Conv'] * 2,
                 kernel_sizes=3,
                 num_heads=2,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path=[0.1] * 3,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 eps=1e-6,
                 downsample=None,
                 **kwargs):
        super().__init__()
        self.dim = dim

        self.blocks = nn.Sequential()
        for i in range(depth):
            if mixer[i] == 'Conv' or mixer[i] == 'ConvOut':
                self.blocks.append(
                    ConvBlock(dim=dim,
                              kernel_size=kernel_sizes,
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              drop=drop_rate,
                              act_layer=act,
                              drop_path=drop_path[i],
                              norm_layer=norm_layer,
                              eps=eps,))
            elif mixer[i] == 'ConvLast':
                self.blocks.append(
                    ConvBlock(dim=dim,
                              kernel_size=kernel_sizes,
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              drop=drop_rate,
                              act_layer=act,
                              drop_path=drop_path[i],
                              norm_layer=norm_layer,
                              eps=eps,
                              last=True,))
            else:
                if mixer[i] == 'Attn':
                    block = Block
                self.blocks.append(
                    block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        act_layer=act,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path[i],
                        norm_layer=nn.LayerNorm,
                        eps=eps,
                    ))    

        if downsample:
            if mixer[-1] == 'Conv':
                self.downsample = SubSample2D(dim, out_dim)
            elif mixer[-1] == 'ConvOut':
                self.downsample = SubSample2D(dim, out_dim, last_conv=True)
        else:
            self.downsample = IdentitySize()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x

class RELG(nn.Module):

    def __init__(self,
                 max_sz=[129, 129],
                 in_channels=3,
                 out_channels=192,
                 depths=[2, 2, 2, 3],
                 dims=[64, 128, 256, 384],
                 mixer=[['Conv'] * 2, ['Conv'] * 2 + ['Conv'] * 2,
                        ['Attn'] * 3],
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.1,
                 last_drop=0.1,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=lambda c: nn.GroupNorm(num_groups=1, num_channels=c, affine=False),
                 act=nn.GELU,
                 eps=1e-6,
                 tranNum=4,
                 **kwargs):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        dpr = np.linspace(0, drop_path_rate,
                          sum(depths))  # stochastic depth decay rule
        self.convIn = Fconv_PCA(sizeP=3, inNum=in_channels, outNum=dims[0]//tranNum, stride=1, tranNum=tranNum, padding=1, ifIni=1)
        self.bnIn = nn.GroupNorm(num_groups=1, num_channels=dims[0], affine=False)
        self.act = nn.GELU()

        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = Stage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                mixer=mixer[i_stage],
                kernel_sizes=3,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
            )
            self.stages.append(stage)

        self.out_channels = self.num_features
        self.tranNum = tranNum
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_embed', 'downsample', 'pos_embed'}

    def forward(self, x):
        x = self.convIn(x)
        x = self.bnIn(x)
        x = self.act(x)
        
        for stage in self.stages:
            x = stage(x)
        if len(x.shape) == 4:
            x = x.flatten(2, 3).transpose(1, 2)
        return x
