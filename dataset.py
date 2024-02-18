import glob
import numpy as np

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    RandCropByPosNegLabeld,
)


class CHD2CHD:
    """
       remove labels out of 1-7
    """
    def operation(self, data):
        new = np.where(data>=8, 0, data)
        return new

    def __call__(self, data):
        label = data['label']
        label = self.operation(label)
        data['label'] = label
        return data

    
class Lymph2CHD:
    """
       lymph_node: 1,    2,    3,    4,    5,    6,    7,    0
       chd:        5,    3,    1,    4,    2,    6,    7,    0
    """
    def operation(self, data):
        origin_labels = {
            1 : 5,
            2 : 3,
            3 : 1,
            4 : 4,
            5 : 2,
            6 : 6,
            7 : 7,
        }
        new = np.zeros(data.shape)
        for k,v in origin_labels.items():
            new = np.where(data==k, v, new)
        return new

    def __call__(self, data):
        label = data['label']
        label = self.operation(label)
        data['label'] = label
        return data
    
    
class MMWHS2CHD:
    """
       mm-whs raw: 205., [420., 421.], 500., 550., 600., 820., 850., TO
       correspond: 1,    2,            3,    4,    5,    6,    7,    0
       chd:        5,    3,            1,    4,    2,    6,    7,    0
    """

    def operation(self, data):
        origin_labels = {
            205 : 5,
            420 : 3, 421 : 3,
            500 : 1,
            550 : 4,
            600 : 2,
            820 : 6,
            850 : 7,
        }
        new = np.zeros(data.shape)
        for k,v in origin_labels.items():
            new = np.where(data==k, v, new)
        return new

    def __call__(self, data):
        label = data['label']
        label = self.operation(label)
        data['label'] = label
        return data
    

def get_data(args):
    if 'imagechd' in args.dataset:
        assert args.num_classes == 8
        if 'fewshot' in args.dataset:
            train_images = sorted(glob.glob(f'{args.data_dir}/train_8/images/*image.nii.gz'))
            train_labels = sorted(glob.glob(f'{args.data_dir}/train_8/labels/*label.nii.gz'))
        else:
            train_images = sorted(glob.glob(f'{args.data_dir}/train/images/*image.nii.gz'))
            train_labels = sorted(glob.glob(f'{args.data_dir}/train/labels/*label.nii.gz'))
            
        val_images = sorted(glob.glob(f'{args.data_dir}/valid/images/*image.nii.gz'))
        val_labels = sorted(glob.glob(f'{args.data_dir}/valid/labels/*label.nii.gz'))
        
    elif args.dataset == 'lymph':
        assert args.num_classes == 8
        train_images = sorted(glob.glob(f'{args.data_dir}/train/images/*.nii.gz'))
        train_labels = sorted(glob.glob(f'{args.data_dir}/train/labels/*.nii.gz'))
        val_images = sorted(glob.glob(f'{args.data_dir}/valid/images/*.nii.gz'))
        val_labels = sorted(glob.glob(f'{args.data_dir}/valid/labels/*.nii.gz'))
    
    elif 'mmwhs' in args.dataset:
        assert args.num_classes == 8
        images = sorted(glob.glob(f'{args.data_dir}/images/*_image.nii.gz'))
        labels = sorted(glob.glob(f'{args.data_dir}/labels/*_label.nii.gz'))
        if args.fold == 1:
            train_images, val_images = images[:16], images[16:]
            train_labels, val_labels = labels[:16], labels[16:]
        elif args.fold == 2:
            train_images, val_images = images[:12]+images[16:], images[12:16]
            train_labels, val_labels = labels[:12]+labels[16:], labels[12:16]
        elif args.fold == 3:
            train_images, val_images = images[:8]+images[12:], images[8:12]
            train_labels, val_labels = labels[:8]+labels[12:], labels[8:12]
        elif args.fold == 3:
            train_images, val_images = images[:8]+images[12:], images[8:12]
            train_labels, val_labels = labels[:8]+labels[12:], labels[8:12]
        elif args.fold == 4:
            train_images, val_images = images[:4]+images[8:], images[4:8]
            train_labels, val_labels = labels[:4]+labels[8:], labels[4:8]
        elif args.fold == 5:
            train_images, val_images = images[4:], images[:4]
            train_labels, val_labels = labels[4:], labels[:4]

    elif args.dataset == 'msd':
        assert args.num_classes == 2
        images = sorted(glob.glob(f'{args.data_dir}/imagesTr/*.nii.gz'))
        labels = sorted(glob.glob(f'{args.data_dir}/labelsTr/*.nii.gz'))
        if args.fold == 1:
            train_images, val_images = images[:16], images[16:]
            train_labels, val_labels = labels[:16], labels[16:]
        elif args.fold == 2:
            train_images, val_images = images[:12]+images[16:], images[12:16]
            train_labels, val_labels = labels[:12]+labels[16:], labels[12:16]
        elif args.fold == 3:
            train_images, val_images = images[:8]+images[12:], images[8:12]
            train_labels, val_labels = labels[:8]+labels[12:], labels[8:12]
        elif args.fold == 3:
            train_images, val_images = images[:8]+images[12:], images[8:12]
            train_labels, val_labels = labels[:8]+labels[12:], labels[8:12]
        elif args.fold == 4:
            train_images, val_images = images[:4]+images[8:], images[4:8]
            train_labels, val_labels = labels[:4]+labels[8:], labels[4:8]
        elif args.fold == 5:
            train_images, val_images = images[4:], images[:4]
            train_labels, val_labels = labels[4:], labels[:4]

    train_files = [
                    {"image": image_name, "label": label_name} 
                    for image_name, label_name in zip(train_images, train_labels)
                  ][:]
    val_files = [
                    {"image": image_name, "label": label_name}
                    for image_name, label_name in zip(val_images, val_labels)
                ][:]

    print('Train files: {}'.format(len(train_files)))
    print('Val files: {}'.format(len(val_files)))
    
    return train_files, val_files
    

def get_transforms(args):
    common_transform1 = [
                            LoadImaged(keys=["image", "label"]),
                            EnsureChannelFirstd(keys=["image", "label"]),
                            Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
                            Orientationd(keys=["image", "label"], axcodes='RAS'),
                        ]
    common_transform2 = [
                            CropForegroundd(keys=["image", "label"], source_key="image"),
                        ]
    common_transform3 = [
                            RandCropByPosNegLabeld(
                                                        keys=["image", "label"],
                                                        label_key="label",
                                                        spatial_size=args.image_size,
                                                        pos=args.num_pos, neg=args.num_neg,
                                                        num_samples=args.num_samples,
                                                        image_key="image",
                                                        image_threshold=0,
                                                        allow_smaller=True
                                                    ),
                            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=args.image_size),
                        ]
    enhance_transform = [
                            RandGaussianNoised(keys="image", prob=0.5, std=0.05),
                            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                        ]
                        

    if 'chdseg' in args.dataset:
        intensity_transform = [
                                    ScaleIntensityRanged(
                                                            keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                                            b_min=0.0, b_max=1.0, clip=True,
                                                        ),
                              ]
        label_transform = [
                                CHD2CHD(),
                                EnsureTyped(keys=["image", "label"]),
                            ]
        
    elif args.dataset == 'lymph':
        intensity_transform = [
                                    ScaleIntensityRanged(
                                                            keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                                            b_min=0.0, b_max=1.0, clip=True,
                                                        ),
                              ]
        label_transform = [
                                Lymph2CHD(),
                                EnsureTyped(keys=["image", "label"]),
                            ]
        
    elif args.dataset == 'mmwhs_mri':
        intensity_transform = [
                                    ScaleIntensityRangePercentilesd(
                                                    keys=["image"], lower=0, upper=98,
                                                    b_min=0.0, b_max=1.0, clip=True, relative=False
                                                ),
                              ]
        label_transform = [
                                MMWHS2CHD(),
                                EnsureTyped(keys=["image", "label"]),
                            ]
        
    elif args.dataset == 'msd':
        intensity_transform = [
                                    ScaleIntensityRangePercentilesd(
                                                    keys=["image"], lower=0, upper=98,
                                                    b_min=0.0, b_max=1.0, clip=True, relative=False
                                                ),
                              ]
        label_transform = [EnsureTyped(keys=["image", "label"]),]
        
    train_transform = Compose(common_transform1 + intensity_transform + common_transform2 + common_transform3 
                                + enhance_transform + label_transform)
    val_transform = Compose(common_transform1 + intensity_transform + common_transform2 + label_transform)
    
    return train_transform, val_transform
    
    
def get_loader(args):
    train_files, val_files = get_data(args)
    train_transform, val_transform = get_transforms(args)
    train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=args.cache_rate, num_workers=args.workers)
    val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=args.cache_rate, num_workers=args.workers)
    
    train_loader = DataLoader(
                                train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=True,
                            )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.workers)
    
    return train_loader, val_loader