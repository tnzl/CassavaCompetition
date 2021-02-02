from learner import Learner
from data import Data
from callbacks import WandbCallback
import wandb

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

import torchvision 
from torch import nn, optim

#aug
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

def map_fn(index, flags, wandb_run):
    
    t = {
        'train' : Compose([
            # RandomResizedCrop(self.img_size, self.img_size),
            # Transpose(p=0.5),
            # HorizontalFlip(p=0.5),
            # VerticalFlip(p=0.5),
            # ShiftScaleRotate(p=0.5),
            # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # CoarseDropout(p=0.5),
            # Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.),
    'val' : Compose([
        # Resize(int(self.img_size*1.1), int(self.img_size*1.1)),
        # CenterCrop(self.img_size, self.img_size, p=1.),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)
    }

    data = Data(
        data_root=flags['data_root'], 
        num_workers=flags['num_workers'], 
        bs=flags['batch_size'], 
        debug=flags['debug'], 
        sampler=None, 
        transforms=t, 
        fold=0, 
        num_folds=5, 
        img_size=flags['img_size'], 
        tpu=True
        )

    data.ds['train'].data_source = '/kaggle/input/cassava-jpeg-256x256/kaggle/train_images_jpeg/'
    data.ds['val'].data_source = '/kaggle/input/cassava-jpeg-256x256/kaggle/train_images_jpeg/'

    net = torchvision.models.resnet18(pretrained=True).double()
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, 5)
    optimizer = optim.SGD(net.parameters(), lr=0.001*xm.xrt_world_size(), momentum=0.9)
    
    xm.rendezvous('barrier-1')
    learner = Learner(net, 
                      optimizer=optimizer, 
                      loss_fn=nn.CrossEntropyLoss(), 
                      dl=data.get_dl(), 
                      device=xm.xla_device(), 
                      num_epochs=flags['num_epochs'], 
                      bs=flags['batch_size'], 
                      verbose=True, 
                    #   cbs = [WandbCallback(wandb_run)],
                      tpu=index+1, 
                      seed=1234, 
                      metrics=None, 
                      lr_schedule=optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                    )
    learner.fit()
    learner.verboser("Complete!")
 #   xm.rendezvous('barrier-2')

flags = {
    'project' : "cassava-leaf-disease-classification",
    'run_name' : 'try-xxx',
    'pin_memory': True,
    'data_root' : '/kaggle/input/cassava-leaf-disease-classification',
    'img_size' : 256,   
    'fold': 0,
    'model': 'resnet18',
    'pretrained': True,
    'batch_size': 16,
    'num_workers': 2,
    'lr': 0.001,
    'seed' : 1234,
    'verbose' : True
}
flags['img_size'] = 256
flags['batch_size'] = 16
flags['num_workers'] = 4
flags['seed'] = 1234
flags['debug'] = False
flags['num_epochs'] = 2 if flags['debug'] else 10

# wandb_run = wandb.init(project=flags['project'], name=flags['run_name'], config=flags)
wandb_run = None
xmp.spawn(map_fn, args=(flags,wandb_run,), nprocs=8, start_method='fork')

# Conclusion:
# dataloaders are the bottleneck of pipeline. Reading large images takes a lot of time.