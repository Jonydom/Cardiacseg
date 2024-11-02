""" Cube to Patch Embedding using Conv3d

A convolution based approach to patchifying a 3D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
import sys
from torch import nn as nn

sys.path.insert(0, '/home/jianglei/VCL-Project/data/2022Jianglei/CardiacSeg/utils/')
from utils.helpers import to_3tuple
from utils.trace_utils import _assert


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # jia-xin
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # jia-xin
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.flatten = flatten
        # jiaxin
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # jiaxin
        B, C, D, H, W = x.shape
        _assert(D == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(H == self.img_size[1], f"Input image height ({H}) doesn't match model ({self.img_size[1]}).")
        _assert(W == self.img_size[2], f"Input image width ({W}) doesn't match model ({self.img_size[2]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNC
        x = self.norm(x)
        return x
