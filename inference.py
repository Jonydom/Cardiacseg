import matplotlib.pyplot as plt
import glob
import os
import torch
import sys
import numpy as np
from numpy import *
import SimpleITK as sitk
import scipy.ndimage as ndimage

from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Resized,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
    CropForegroundd,
    EnsureTyped,
    EnsureType
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
sys.path.insert(0, '/tmp/code/cardiacseg/networks')
from cardiacseg import CardiacSeg


def load_model(args, device, model_path):
    model = CardiacSeg(
                        in_channels = 1,
                        out_channels = args.num_classes,
                        img_size = args.image_size,
                        feature_size = 16,    ####
                        hidden_size = 768,
                        mlp_dim = 3072,
                        num_heads = 12,
                        norm_name = "batch",
                        res_block = True,
                        dropout_rate = 0.0,
                        args = args
                    )
    model = torch.nn.DataParallel(model).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model


test_transform = Compose(
[
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1, 1, 1), mode="bilinear"),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRangePercentilesd(
                                        keys=["image"], lower=0, upper=98,
                                        b_min=0.0, b_max=1.0, clip=True, relative=False
                                    ),
    # CropForegroundd(keys=["image"], source_key="image"),
    EnsureTyped(keys=["image"]),
])


def inference(args, test_dir, model, output_dir):
    test_images = sorted(glob.glob(f'{test_dir}/*_image.nii.gz'))
    test_files = [{"image": image_name} for image_name in test_images][:2]
    print('Number of cases:', len(test_files))

    # test_files
    test_ds = CacheDataset(data=test_files, transform=test_transform, cache_rate=0.0, num_workers=8,)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=8,)
    
    sw_batch_size = 4
    
    with torch.no_grad():
        n = 0
        for test_data in test_loader:
            test_inputs = test_data['image'].to(device)
            test_outputs = sliding_window_inference(test_inputs, args.image_size, sw_batch_size, model, overlap=0.25)
            outputs = torch.softmax(test_outputs, 1).cpu().numpy()
            outputs = np.argmax(outputs, axis=1).astype(np.uint8)[0]
            save_result(n, test_files, test_inputs, outputs, output_dir)
            n += 1
    return


def save_result(n, files, test_inputs, val_outputs, output_dir):
    raw_img_p = files[n]['image']
    raw_name = raw_img_p.split('/')[-1]
    save_name = raw_name.replace('image.nii.gz', 'predlabel.nii.gz')
    
    raw_img = sitk.ReadImage(raw_img_p)
    raw_img_arr = sitk.GetArrayFromImage(raw_img)
    raw_size = raw_img_arr.shape
    pred_shape = val_outputs.shape
    
    zoom = (raw_size[2]/pred_shape[0], raw_size[1]/pred_shape[1], raw_size[0]/pred_shape[2])
    pred_arr = ndimage.zoom(val_outputs, zoom, output=np.uint8, order=0, mode='nearest', prefilter=False)
    pred_arr = pred_arr.transpose(2,1,0)
#     pred_arr = np.flip(pred_arr, 2)

    out = sitk.GetImageFromArray(pred_arr)
    out.SetDirection(raw_img.GetDirection())
    out.SetOrigin(raw_img.GetOrigin())
    out.SetSpacing(raw_img.GetSpacing())
    save_pred = f'{output_dir}/{save_name}'
    sitk.WriteImage(out, save_pred)
    print('Done: {}'.format(raw_name))
            
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", default=8, type=int, help="number of segmentation classes, including background")
    parser.add_argument("--ckpt_url", default='/tmp/pretrainmodel/CardiacSeg_model_o5l2/CardiacSeg_CT_model.pth', type=str, help="Optional input file, read from stdin if not given", nargs="?")
    parser.add_argument("--in_file", default='/tmp/dataset/CHD_Seg/test/images', help="Optional input file, read from stdin if not given", nargs="?")
    parser.add_argument("--out_file", default='/tmp/output', help="Optional output file, write to stdout if not given", nargs="?")
    parser.add_argument("--arch", default="vit_base", type=str)
    parser.add_argument("--finetune", action="store_true", help="finetune a pretrained model, else train from scratch")
    args = parser.parse_args()

    args.image_size = (128, 128, 128)

    # load the network, assigning it to the selected device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model_path = args.ckpt_url
    model = load_model(args, device, model_path)
    
    data_dir = args.in_file
    out_dir = args.out_file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # inference
    inference(args, data_dir, model, out_dir)
