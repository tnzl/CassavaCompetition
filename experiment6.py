import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch_xla.distributed.parallel_loader as pl
import time
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import wandb
import pandas as pd
from sklearn.model_selection import StratifiedKFold   
from sklearn import metrics
from sklearn import model_selection
from data import CassavaDataset, TestDataset

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

    sampler = {
        'train' : torch.utils.data.distributed.DistributedSampler(
            ds['train'],
            num_replicas=xm.xrt_world_size(), #divide dataset among this many replicas
            rank=xm.get_ordinal(), #which replica/device/core
            shuffle=True),
        'val' : torch.utils.data.distributed.DistributedSampler(
            ds['val'],
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
    }

    dl = {
        'train': torch.utils.data.DataLoader(
            ds['train'],
            batch_size=flags['batch_size'],
            sampler=sampler['train'],
            num_workers=flags['num_workers'],
            drop_last=True),
        'val' :  torch.utils.data.DataLoader(
            ds['val'],
            batch_size=flags['batch_size'],
            sampler=sampler['val'],
            num_workers=flags['num_workers'],
            drop_last=False)
    }
    return dl

from learner import Learner

def map_fn(index, flags, wandb_run):
        
    net = torchvision.models.resnet18(pretrained=True).double()
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, 5)
    optimizer=torch.optim.SGD(net.parameters(), lr=0.001*xm.xrt_world_size(), momentum=0.9)
    
    xm.rendezvous('barrier-1')
    learner = Learner(net, 
                      optimizer=optimizer, 
                      loss_fn=torch.nn.CrossEntropyLoss(), 
                      dl=get_dl(flags), 
                      device=xm.xla_device(), 
                      num_epochs=flags['num_epochs'], 
                      bs=flags['batch_size'], 
                      verbose=True, 
                      tpu=index+1, 
                      seed=1234, 
                      metrics=None, 
                      lr_schedule=lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
                      wandb_run=wandb_run)
    learner.fit()
 #   xm.rendezvous('barrier-2')

flags = {
    'project' : "cassava-leaf-disease-classification",
    'run_name' : 'try-xxx',
    'pin_memory': True,
    'data_root' : '/kaggle/input/cassava-leaf-disease-classification',
    'img_size' : 320,   
    'fold': 0,
    'model': 'resnext50_32x4d',
    'pretrained': True,
    'batch_size': 64,
    'num_workers': 0,
    'lr': 0.001,
    'seed' : 1234,
    'verbose' : True
}
flags['img_size'] = 320
flags['batch_size'] = 32
flags['num_workers'] = 4
flags['seed'] = 1234
flags['debug'] = True
flags['num_epochs'] = 2 if flags['debug'] else 25

# wandb_run = wandb.init(project=flags['project'], name=flags['run_name'], config=flags)
wandb_run = None
xmp.spawn(map_fn, args=(flags,wandb_run,), nprocs=8, start_method='fork')