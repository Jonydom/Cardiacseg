import torch
import argparse
from networks.cardiacseg import CardiacSeg
from networks.unet3d import UNet3D
from networks.cotr3d import ResTranUnet

def choose_model(args):
    
    hidden_size, num_heads = vit_params(args)
    
    if args.model_name == 'cardiacseg':
        model = CardiacSeg(in_channels = args.in_channels,
                            out_channels = args.num_classes,
                            img_size = args.image_size,
                            feature_size = 16,    ####
                            hidden_size = hidden_size,
                            mlp_dim = 3072,
                            num_heads = num_heads,
                            pos_embed = "conv",
                            norm_name = args.norm,
                            conv_block = True,
                            res_block = True,
                            dropout_rate = 0.0,
                            spatial_dims = 3,
                            args = args)
    elif args.model_name == 'unet3d':
        model = UNet3D(
            in_channels = args.in_channels,
            num_classes = args.num_classes
        )
    elif args.model_name == 'cotr3d':
        model = ResTranUnet(
            norm_cfg = "BN",
            activation_cfg = "ReLU",
            img_size = args.image_size,
            num_classes = args.num_classes,
            weight_std = True,
            deep_supervision = False
        )
    return model


def vit_params(args):
    
    if args.arch == 'vit_tiny':
        hidden_size = 192
        num_heads = 3
    elif args.arch == 'vit_base':
        hidden_size = 768
        num_heads = 12
    elif args.arch == 'vit_large':
        hidden_size = 1152
        num_heads = 16
    elif args.arch == 'vit_huge':
        hidden_size = 1344
        num_heads = 16

    return hidden_size, num_heads


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cardiac segmentation pipeline")
    parser.add_argument("--arch", default='vit_base', type=str, help="type of ViT")
    parser.add_argument("--model_name", default='cotr3d', type=str, help="network used for segmrntation")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--num_classes", default=7+1, type=int, help="number of segmentation classes, including background")
    parser.add_argument("--input_size", default=96, type=int, help="image size for network input")
    parser.add_argument("--norm", default='batch', type=str, help="network used for segmrntation")
    parser.add_argument("--finetune", action="store_true", help="finetune a pretrained model, else train from scratch")
    args = parser.parse_args()
    args.image_size = (args.input_size, args.input_size, args.input_size)
    
    input_tensor = torch.randn(2, 1, 96, 96, 96)
    model = choose_model(args)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)