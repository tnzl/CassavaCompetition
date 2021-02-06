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

import pandas as pd 
import gc
from PIL import Image
import os
import torch
import numpy as np
import warnings

from data import ImageDataset, get_default_sampler, get_default_transforms, get_dl
from learner import train_loop_fn, eval_loop_fn

warnings.filterwarnings("ignore")
os.environ['XLA_USE_BF16']="1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

def run(rank, flags):
    xm.master_print('Starting run')
    global FLAGS
    torch.set_default_tensor_type('torch.FloatTensor')
    device = xm.xla_device() #device, will be different for each core on the TPU
    epochs = flags['epochs']
    fold = flags['fold']
    training_data_path = "/kaggle/input/cassava-jpeg-256x256/kaggle/train_images_jpeg" #define the dataset path
    # define DataLoader with the defined sampler    
    train_loader = pl.MpDeviceLoader(get_dl(flags, fold=1, aim='train'), device) # puts the train data onto the current TPU core
    valid_loader = pl.MpDeviceLoader(get_dl(flags, fold=1, aim='valid'), device) # puts the valid data onto the current TPU core

    model = MX.to(device) # put model onto the current TPU core
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=FLAGS['lr']*xm.xrt_world_size()) # often a good idea to scale the learning rate by number of cores
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*FLAGS['epochs']) #let's use a scheduler

    gc.collect()
    
    xm.master_print(f'========== training fold {FLAGS["fold"]} for {FLAGS["epochs"]} epochs ==========')
    for i in range(flags['epochs']):
        xm.master_print(f'EPOCH {i}:')
        # train one epoch
        train_loop_fn(train_loader, loss_fn, model, optimizer, device, scheduler)
                
        # validation one epoch
        eval_loop_fn(valid_loader, loss_fn, model, device)

        gc.collect()
    
    xm.rendezvous('save_model')
    
    xm.master_print('save model')
    
    xm.save(model.state_dict(), f'xla_trained_model_{FLAGS["epochs"]}_epochs_fold_{FLAGS["fold"]}.pth')

# create folds
df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
df["kfold"] = -1    
df = df.sample(frac=1).reset_index(drop=True)
y = df.label.values
kf = model_selection.StratifiedKFold(n_splits=5)
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f
df.to_csv("train_folds.csv", index=False)

import torchvision
net = torchvision.models.resnext50_32x4d(pretrained=True).double()
for param in net.parameters():
    param.requires_grad = False
net.fc = nn.Linear(net.fc.in_features, 5)
MX = xmp.MpModelWrapper(net)

FLAGS = {
    'project' : "cassava-leaf-disease-classification",
    'run_name' : 'resnext',
    'train_folds_path' : './train_folds.csv',
    'training_data_path' : "/kaggle/input/cassava-jpeg-256x256/kaggle/train_images_jpeg",
    'fold': 0,
    'model': 'resnext50_32x4d',
    'pretrained': True,
    'batch_size': 128,
    'num_workers': 4,
    'lr': 3e-4,
    'epochs': 10
}
wandb_run = wandb.init(project=FLAGS['project'], name=FLAGS['run_name'], config=FLAGS)
start_time = time.time()
xmp.spawn(run, args=(FLAGS,), nprocs=8, start_method='fork')
print('time taken: ', time.time()-start_time)