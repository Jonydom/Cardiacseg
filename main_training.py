import os
import argparse
import torch
from torch.optim.lr_scheduler import PolynomialLR, MultiStepLR
from monai.utils import set_determinism
# from detectron2.solver.lr_scheduler import WarmupMultiStepLR

# yzy's code
from dataset import get_loader
from models import choose_model
from trainer import training


def parse_args():
    parser = argparse.ArgumentParser(description="Cardiac segmentation pipeline")
    
    parser.add_argument("--adapterformer", action="store_true", help="use adapterformer or not")
    parser.add_argument("--finetune", action="store_true", help="finetune a pretrained model, else train from scratch")
    parser.add_argument("--lora", action="store_true", help="use LoRA adapter or not")
    parser.add_argument("--not_froze_encoder", action="store_true", help="freeze encoder's parameters or not")
    parser.add_argument("--res_adpter", action="store_true", help="use residual adapter or not")

    parser.add_argument("--arch", default='vit_base', type=str, help="type of ViT")
    parser.add_argument("--a_min", default=500.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=2000.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--cache_rate", default=1.0, type=float, help="ceche rate in CacheDataset")
    parser.add_argument("--dataset", default="imagechd", type=str, help="dataset")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/ImageCHD_split_sdf", type=str, help="dataset directory")
    parser.add_argument("--demo_interval", default=30, type=int, help="the interval for plotting demo")
    parser.add_argument("--epoch_end", default=500, type=int, help="the end epoch of training")
    parser.add_argument("--epoch_start", default=0, type=int, help="the start epoch of training")
    parser.add_argument("--fold", default=None, type=int, help="the fold of 5-fold validation")
    parser.add_argument("--gpu", default="0", type=str, help="gpu id")
    parser.add_argument("--n_gpu", default=1, type=int, help="the number of gpu")

    parser.add_argument("--input_size", default=96, type=int, help="image size for network input")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--lr_decay_epoch", default=450, type=int, help="epoch learning rate decay")
    #### loss ####
    parser.add_argument("--loss", nargs='+', default=["dice", "ce", "rmi"], type=str, help="loss function to use")
    parser.add_argument("--lossw_dice", default=1.0, type=float, help="weight for dice loss")
    parser.add_argument("--lossw_ce", default=0.5, type=float, help="weight for ce loss")
    parser.add_argument("--lossw_rmi", default=0.1, type=float, help="weight for rmi loss")
    parser.add_argument("--sigmoid_rmi", action="store_true", help="adding rmi loss in sigmoid manner")
    parser.add_argument("--rmi_ds", default='avg', type=str, help="downsampling method for rmi loss")
    parser.add_argument("--rmi_epoch", default=450, type=int, help="the epoch rmi loss adds")
    parser.add_argument("--rmi_stride", default=2, type=int, help="downsampling stride for rmi loss")
    parser.add_argument("--rmi_radius", default=3, type=int, help="radius for rmi loss")

    parser.add_argument('--local_rank', default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--model_name", default='cardiacseg', type=str, help="network used for segmrntation")
    parser.add_argument("--norm", default='batch', type=str, help="network used for segmrntation")
    parser.add_argument("--num_pos", default=1, type=int, help="number of positive samples for RandCropByPosNegLabeld")
    parser.add_argument("--num_neg", default=3, type=int, help="number of negative samples for RandCropByPosNegLabeld")
    parser.add_argument("--num_samples", default=4, type=int, help="number of samples for RandCropByPosNegLabeld")
    parser.add_argument("--num_classes", default=7+1, type=int, help="number of segmentation classes, including background")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/output/sdf-b2bn-cardiacseg", type=str, help="directory to save the outputs")
    parser.add_argument("--plot_col", default=8, type=int, help="number of columns in demo")
    parser.add_argument("--plot_row", default=4, type=int, help="number of rows in demo")
    parser.add_argument("--plot_slices", default=3, type=int, help="number of slice interval in demo")
    parser.add_argument("--pretrained_model", default="", type=str, help="pretrained model path")

    

    parser.add_argument("--resume_ckpt", default='', type=str, help="resume training from pretrained checkpoint")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
    parser.add_argument("--val_interval", default=10, type=int, help="number of intervals to validate and save models")
    parser.add_argument("--warm_up", default=20, type=int, help="warm up epochs")
    parser.add_argument("--workers", default=4, type=int, help="number of workers")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_determinism(seed=args.seed)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_DEVICE_ORDER"] = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    args.image_size = (args.input_size, args.input_size, args.input_size)
    model = choose_model(args)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).cuda()
    elif args.n_gpu == 1:
        model = model.cuda()
    
    if args.finetune:
        checkpoint = torch.load(args.pretrained_model, map_location='cpu')
        ckpt = {}
                        
        for key, value in checkpoint['model'].items():
            if key not in ['pos_embed']:
                new_key = 'module.vit.{}'.format(key)
                ckpt[new_key] = value
        out = model.load_state_dict(ckpt, strict=False)

        print(out)
        print("=> loaded checkpoint '{}'".format(args.pretrained_model))

    params_all = [p for p in model.parameters()]
    params_bp = [p for p in model.parameters() if p.requires_grad]
    print("Total parameters to train:", len(params_bp))
    print("Total parameters count:", len(params_all))
    
    optimizer = torch.optim.AdamW(
                                    params_bp,
                                    lr = args.lr, 
                                    weight_decay = 1e-5,
                                )
    # lr_scheduler = WarmupMultiStepLR(
    #                                     optimizer = optimizer,
    #                                     milestones = [args.lr_decay_epoch],
    #                                     gamma = 0.1,
    #                                     warmup_factor = args.lr,
    #                                     warmup_iters = args.warm_up,
    #                                     warmup_method = "linear",
    #                                     last_epoch = -1,
    #                                 )
    lr_scheduler = MultiStepLR(
        optimizer = optimizer,
        milestones = [args.lr_decay_epoch],
        gamma = 0.1,
        last_epoch = -1
    )
    
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
        optimizer_state_dict = checkpoint['optimizer']
        scheduler_state_dict = checkpoint['scheduler']
        optimizer.load_state_dict(optimizer_state_dict)
        lr_scheduler.load_state_dict(scheduler_state_dict)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume_ckpt))
    
    train_loader, val_loader = get_loader(args)
    print("Get dataloader!")
    training(model, train_loader, val_loader, optimizer, lr_scheduler, args)

            
            
if __name__ == '__main__':
    main()