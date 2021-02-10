# import timm 
import gc
import os
import time
import torch
import albumentations
import wandb
import numpy as np
import pandas as pd

import cv2
from PIL import Image

import torch.nn as nn
from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F
from torch.optim import Adam

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser

import torchvision
import pandas as pd 
import gc
import os
import torch
import numpy as np
import warnings


from data import ImageDataset, get_default_sampler, get_default_transforms, get_dl
from learner import fit
from callbacks import CallbackManager, PrintCallback, UnfreezePattern, ModelSaver

os.environ['XLA_USE_BF16']="1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

def run(rank, flags):
    global FLAGS
    torch.set_default_tensor_type('torch.FloatTensor')
    
    train_dict = {
        'flags' : flags,
        'device' : xm.xla_device(),
        'loss_fn' : nn.CrossEntropyLoss()
    }
    train_dict['train_loader'] = pl.MpDeviceLoader(get_dl(flags, fold=flags['fold'], aim='train'), train_dict['device'])
    train_dict['valid_loader'] = pl.MpDeviceLoader(get_dl(flags, fold=flags['fold'], aim='valid'), train_dict['device'])
    train_dict['model'] = MX.to(train_dict['device'])
    train_dict['optimizer'] = Adam(train_dict['model'].parameters(), lr=flags['lr']*xm.xrt_world_size()) 
    train_dict['lr_schedule'] = torch.optim.lr_scheduler.CosineAnnealingLR(train_dict['optimizer'], len(train_dict['train_loader'])*flags['epochs'])
    train_dict['cb_manager'] = CallbackManager(train_dict)
    train_dict['cbs'] = [PrintCallback(logger=xm.master_print), UnfreezePattern(flags['unfreeze_pattern']), ModelSaver()]
    gc.collect()
    
    xm.master_print(f'========== training fold {FLAGS["fold"]} for {FLAGS["epochs"]} epochs ==========')

    fit(train_dict)

    xm.rendezvous('save_model')
    
    # xm.master_print('save model')
    
    xm.save(train_dict['model'].state_dict(), f'xla_trained_model_{FLAGS["epochs"]}_epochs_fold_{FLAGS["fold"]}.pth')

FLAGS = {
    'project' : "cassava-leaf-disease-classification",
    'run_name' : 'resnext',
    'train_folds_path' : './train_folds.csv',
    'training_data_path' : "/kaggle/input/cassava-jpeg-256x256/kaggle/train_images_jpeg",
    'fold': 0,
    'model': 'resnext50_32x4d',
    'unfreeze_pattern' : [2]*3+list(range(5,12,3))+list(range(17, 72, 6))+list(range(80, 162, 9))+[161],
    'pretrained': True,
    'batch_size': 32,
    'num_workers': 4,
    'lr': 3e-4,
    'epochs': 10, 
    'seed':1111
}

# create folds
df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
df["kfold"] = -1    
df = df.sample(frac=1).reset_index(drop=True)
y = df.label.values
kf = model_selection.StratifiedKFold(n_splits=5)
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f
df.to_csv("train_folds.csv", index=False)

# model
net = torchvision.models.resnext50_32x4d(pretrained=True).double()
for param in net.parameters():
    param.requires_grad = False
net.fc = nn.Linear(net.fc.in_features, 5)
MX = xmp.MpModelWrapper(net)

# Spawn processes
# wandb_run = wandb.init(project=FLAGS['project'], name=FLAGS['run_name'], config=FLAGS)
start_time = time.time()
xmp.spawn(run, args=(FLAGS,), nprocs=8, start_method='fork')
print('time taken: ', time.time()-start_time)