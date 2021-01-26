import pandas as pd
from skimage import io
import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils

#aug
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

import pandas as pd
from sklearn.model_selection import StratifiedKFold   
from sklearn import metrics
from sklearn import model_selection

class CassavaDataset():
    
    def __init__(self, data_csv, data_root, transforms=None, augmentation=None):
        
        self.data_source = data_root + '/train_images/'
        self.transforms = transforms 
        self.map_classes = {}
        
        from json import load
        with open(data_root+'/label_num_to_disease_map.json') as f:
             mm = load(f)
        for key in mm.keys():
            self.map_classes[int(key)] = mm[key]
        
        self.train = data_csv
        self.train['disease'] = self.train["label"].map(self.map_classes)     
    
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.read_img(self.train.iloc[idx, 0])
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        #image = image.transpose((2, 0, 1))
        label = self.train.iloc[idx, 1]
        
        return image.double(), int(label)
    
    def read_img(self, img_id):
        return io.imread(self.data_source+img_id)
        
    def print_random(self, label=None):
        
        idx = np.random.randint(low=0, high=self.__len__())
        img, label = self.__getitem__(idx)
        print(str(label), self.map_classes[label])
        print('Image shape:', img.shape, '. Therefore, unable to print.')
        #plt.imshow(img)
        
class TestDataset():
    
    def __init__(self, data_root, transforms=None):
        
        self.data_source = data_root + '/test_images/'
        self.transforms = transforms   
        self.test = os.listdir(self.data_source)
    
    def __len__(self):
        return len(self.test)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.read_img(self.test[idx])
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image
    
    def read_img(self, img_id):
        return io.imread(self.data_source+img_id)
        
    def print_random(self):
        from random import choice
        img = self.read_img(choice(self.test))
        plt.imshow(img)


def get_train_transforms(flags):
        return Compose([
                RandomResizedCrop(flags['img_size'], flags['img_size']),
#                 Transpose(p=0.5),
                HorizontalFlip(p=0.5),
#                 VerticalFlip(p=0.5),
#                 ShiftScaleRotate(p=0.5),
#                 HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#                 RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#                 CoarseDropout(p=0.5),
#                 Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)


def get_valid_transforms(flags):
    return Compose([
            Resize(int(flags['img_size']*1.1), int(flags['img_size']*1.1)),
            CenterCrop(flags['img_size'], flags['img_size'], p=1.),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_dl(flags):  
#     wandb_run = wandb.init(project=flags['project'], name=flags['run_name'], resume=True)
    
    data_transforms = {
        'train': get_train_transforms(flags),
        'val': get_valid_transforms(flags),
    }

    train = pd.read_csv(flags['data_root']+'/train.csv')
    k_fold = StratifiedKFold(n_splits=5).split(train, train['label'])
    train_idx, val_idx = list(k_fold)[0]
    train_idx = val_idx if flags['debug'] else train_idx
    
    ds = {
          'train' : CassavaDataset(train.loc[train_idx,:], flags['data_root'], transforms=data_transforms['train'])
        , 'val' : CassavaDataset(train.loc[val_idx,:], flags['data_root'], transforms=data_transforms['val'])
        , 'test' : TestDataset(flags['data_root'], transforms=data_transforms['val'])
    }

    # sampler = {
    #     'train' : torch.utils.data.distributed.DistributedSampler(
    #         ds['train'],
    #         num_replicas=xm.xrt_world_size(), #divide dataset among this many replicas
    #         rank=xm.get_ordinal(), #which replica/device/core
    #         shuffle=True),
    #     'val' : torch.utils.data.distributed.DistributedSampler(
    #         ds['val'],
    #         num_replicas=xm.xrt_world_size(),
    #         rank=xm.get_ordinal(),
    #         shuffle=False)
    # }

    dl = {
        'train': torch.utils.data.DataLoader(
            ds['train'],
            batch_size=flags['batch_size'],
            # sampler=sampler['train'],
            num_workers=flags['num_workers'],
            drop_last=True),
        'val' :  torch.utils.data.DataLoader(
            ds['val'],
            batch_size=flags['batch_size'],
            # sampler=sampler['val'],
            num_workers=flags['num_workers'],
            drop_last=False)
    }
    return dl
