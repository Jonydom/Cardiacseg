"""Zhuang Jiaxin
lincolnz9511@gmail.com
Reference:
https://github.com/facebookresearch/mae/blob/main/models_vit.py
https://github.com/Project-MONAI/MONAI/blob/b61db797e2f3bceca5abbaed7b39bb505989104d/monai/networks/nets/vit.py
https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/vision_transformer.py
"""

from typing import Sequence, Union
from functools import partial
import torch.nn as nn
import torch
import math

from timm.models.vision_transformer import Block
from utils.patch_embed import PatchEmbed


class VisionTransformer3D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 img_size: Union[Sequence[int], int],
                 patch_size: Union[Sequence[int], int],
                 embed_dim: int = 1024,
                 # mlp_dim: int = 3072
                 mlp_ratio=4,
                 depth: int = 24,
                 num_heads: int=16,
                 qkv_bias: bool=True,
                 norm_layer=nn.LayerNorm,
                 global_pool=False,
                 drop_rate: float=0,
                 classification: bool=False,
                 num_classes: int=0, 
                 args=None
                 ) -> None:

        super().__init__()

        # define structure
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.global_pool = global_pool

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.classification = classification
        if self.classification:
            self.head = nn.Linear(embed_dim, num_classes)

        self.args = args
        if args and hasattr(args, 'finetune'):
            if args.finetune:
                for name, param in self.named_parameters():
                    if not name.startswith('head'):
                        param.requires_grad = False
                        print('{} requires no grad'.format(name))
                    else:
                        param.requires_grad = True
                        print('{} requires grad'.format(name))

    def forward(self, x):
        B, nc, d, w, h = x.shape
        x = self.patch_embed(x)

        if self.classification:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.classification:
            x = x + self.interpolate_pos_encoding(x, d, w, h, self.pos_embed)
        else:
            x = x + self.interpolate_pos_encoding(x, d, w, h, self.pos_embed)[:, 1:]

        x = self.pos_drop(x)

        hidden_states_out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in [3, 6, 9]:
                hidden_states_out.append(x)

        if self.global_pool:
            x = x.mean(dim=1)
            outcome = self.norm(x)
        else:
            x = self.norm(x)
            outcome = x

        if self.classification:
            outcome = self.head(x[:, 0])
            return outcome
        else:
            return outcome, hidden_states_out

    def patchify(self, imgs):
        """jiaxin
        imgs: (N, C, D, H, W), C=1
        x: (N, L, patch_size**3 *1)

        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[3] % p == 0
        assert imgs.shape[4] % p == 0

        d = h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, d, p, h, p, w, p))
        x = torch.einsum('ncdzhpwq->ndhwzpqc', x)
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p**3 * 1))
        return x

    def unpatchify(self, x):
        """jiaxin
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, D, H, W)

        # x: (N, L, patch_size**2 *3)
        # imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        d = h = w = round(x.shape[1]**(1/3.0))
        assert d * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, 1))
        x = torch.einsum('ndhwzpqc->ncdzhpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, d * p, h * p, h * p))

        return imgs

    def interpolate_pos_encoding(self, x, d, w, h, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h and h == d:
            return pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        d0 = d // self.patch_embed.patch_size[0]
        w0 = w // self.patch_embed.patch_size[1]
        h0 = h // self.patch_embed.patch_size[2]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        d0, w0, h0 = d0 + 0.1, w0 + 0.1, h0 + 0.1
        pos_embed_d = round(math.pow(N, 1/3.0))
        pos_embed_h = round(math.pow(N, 1/3.0))
        pos_embed_w = round(math.pow(N, 1/3.0))

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, 
                                    pos_embed_d,
                                    pos_embed_h, 
                                    pos_embed_h,
                                    dim).permute(0, 4, 1, 2, 3),
            scale_factor=(d0 / pos_embed_d, w0 / pos_embed_h, h0 / pos_embed_w), mode='trilinear',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1] and int(d0) == patch_pos_embed.shape[-3]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


# jiaxin
def vit_tiny_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=1152, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch16(**kwargs):
    model = VisionTransformer3D(
        patch_size=16, embed_dim=1344, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model