from typing import Sequence, Tuple, Union

import torch.nn as nn
import torch

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep

import models_3dvit


class CardiacSeg(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        args = None
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.hidden_size = hidden_size

        patch_size = 16
        arch = '{}_patch{}'.format(args.arch, patch_size)

        assert in_channels == 1
        model = models_3dvit.__dict__[arch](
            img_size=img_size,
            in_channels=in_channels,
            global_pool=False,
        )

        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.vit = model
        self.args = args

        if args.finetune:
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            for param in self.vit.parameters():
                param.requires_grad = True
                
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, feature_size*2, kernel_size=2, stride=2),
            nn.BatchNorm3d(feature_size*2),
            nn.ReLU(),
            nn.ConvTranspose3d(feature_size*2, feature_size*2, kernel_size=2, stride=2),
            nn.BatchNorm3d(feature_size*2),
            nn.ReLU(),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, feature_size*4, kernel_size=2, stride=2),
            nn.BatchNorm3d(feature_size*4),
            nn.ReLU(),
        )

        self.fpn3 = nn.Sequential(
            nn.Conv3d(hidden_size, feature_size*8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm3d(feature_size*8),
            nn.ReLU(),
        )

        self.fpn4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(feature_size*2, feature_size*2, kernel_size=2, stride=2),
            nn.BatchNorm3d(feature_size*2),
            nn.ReLU(),
        )


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        if self.args.finetune:
            with torch.no_grad():
                x, hidden_states_out = self.vit(x_in)
        else:
            x, hidden_states_out = self.vit(x_in)

        x_out = self.proj_feat(x, self.hidden_size, self.feat_size)
        enc1 = self.encoder1(x_in)
        enc2 = self.fpn1(x_out)
        enc3 = self.fpn2(x_out)
        enc4 = self.fpn3(x_out)
        enc5 = self.fpn4(x_out)
        dec4 = self.decoder5(enc5, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.deconv(dec2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)