import pandas as pd 
import gc
from PIL import Image
import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# xla 
try:
    import torch_xla.core.xla_model as xm
except:
    pass

#aug
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

class ImageDataset:
    def __init__(
        self,
        image_paths,
        targets,
        resize,
        augmentations=None,
        channel_first=True,
    ):
        """
        :param image_paths: list of paths to images
        :param targets: numpy array
        :param resize: tuple or None
        :param augmentations: albumentations augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.channel_first = channel_first

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        targets = self.targets[item]
        image = Image.open(self.image_paths[item])
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image),torch.tensor(targets) 
        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets),
        }

def get_default_sampler(ds, aim='train'):
    if aim is 'train':
        return torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=xm.xrt_world_size(), #divide dataset among this many replicas
            rank=xm.get_ordinal(), #which replica/device/core
            shuffle=True
            )
    elif aim is 'valid':
        return torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False
            )

def get_default_transforms(aim='train'):
    if aim=='train':
        return Compose([
            # RandomResizedCrop(self.img_size, self.img_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            # ToTensorV2(p=1.0),
        ], p=1.)
    elif aim == 'valid' : 
        return Compose([
            # Resize(int(self.img_size*1.1), int(self.img_size*1.1)),
            # CenterCrop(flags['img_size'], flags['img_size'], p=1.),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # ToTensorV2(p=1.0),
        ], p=1.)

def get_dl(flags, fold=1, aim='train', transforms=None, sampler=None):

    df = pd.read_csv(flags['train_folds_path'])
    if aim == 'train':
        df = df[df.kfold != fold].reset_index(drop=True)
    else:
        df = df[df.kfold == fold].reset_index(drop=True)
    
    images = df.image_id.values.tolist()
    images = [
        os.path.join(flags['training_data_path'], i) for i in images
    ]
    targets = df.label.values

    if transforms is None:
        transforms = get_default_transforms(aim=aim)
    elif transforms is 'NO':
        transforms = None
    
    ds = ImageDataset(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=transforms,
    )

    if sampler is None:
        sampler = get_default_sampler(ds, aim=aim)
    elif 'NO':
        sampler = None
    
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=flags['batch_size'],
        sampler=sampler,
        num_workers=flags['num_workers'],
        drop_last=True
        )
    del df, images, targets, transforms, sampler, ds
    gc.collect()
    return dl

class Data:
    def __init__(self, training_data_path, num_workers=4, batch_size=128, img_size=256, transforms=None, sampler=None, tpu=0, train_folds_path='./train_folds.csv'):
        
        self.training_data_path = training_data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.img_size = img_size
        self.transforms = transforms
        self.sampler = sampler
        self.tpu = tpu
        self.train_folds_path = train_folds_path

        del training_data_path, transforms, sampler, tpu
        gc.collect()

    def get_dl(self, fold, aim='train'):
        '''
        refactor 
        '''
        df = pd.read_csv(self.train_folds_path)
        if aim == 'train':
            df = df[df.kfold != fold].reset_index(drop=True)
        else:
            df = df[df.kfold == fold].reset_index(drop=True)
        
        images = df.image_id.values.tolist()
        images = [
            os.path.join(self.training_data_path, i) for i in images
        ]
        targets = df.label.values

        transforms = self.get_transforms(aim)
        
        ds = ImageDataset(
            image_paths=images,
            targets=targets,
            resize=None,
            augmentations=transforms,
        )
        
        sampler = self.get_sampler(aim)

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True
            )
        del df, images, targets, transforms, sampler, ds
        gc.collect()
        return dl

    def get_transforms(self,  aim='train'):
        if self.transforms is None:
            t = {
                'train' : Compose([
                    # RandomResizedCrop(self.img_size, self.img_size),
                    Transpose(p=0.5),
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    ShiftScaleRotate(p=0.5),
                    HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                    RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                    CoarseDropout(p=0.5),
                    Cutout(p=0.5),
                    # ToTensorV2(p=1.0),
                ], p=1.),
                'valid' : Compose([
                    # Resize(int(self.img_size*1.1), int(self.img_size*1.1)),
                    CenterCrop(self.img_size, self.img_size, p=1.),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                    # ToTensorV2(p=1.0),
                ], p=1.)
                }
            return t[aim]
        elif self.transforms is 'NO':
            t = {
                'train' : None,
                'valid' : None
            }
            return t[aim]
        else:
            return self.transforms[aim]
    
    def get_sampler(self, aim='train'):
        if self.sampler:
            return self.sampler
        elif self.tpu:
            s = {}
            s['train'] = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=xm.xrt_world_size(), #divide dataset among this many replicas
                rank=xm.get_ordinal(), #which replica/device/core
                shuffle=True
                )
            s['valid'] = torch.utils.data.distributed.DistributedSampler(
                valid_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False
                )
            return s 
        else:
            t = {
                'train' : None,
                'valid' : None
            }
            gc.collect()
            return t[aim]
