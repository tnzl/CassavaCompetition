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
# from learner import Learner, train_loop_fn, eval_loop_fn
from callbacks import WandbCallback, PrintCallback

os.environ['XLA_USE_BF16']="1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

def train_loop_fn(train_dict):
    train_dict['model'].train() # put model in training mode
    for bi, d in enumerate(train_dict['train_loader']): # enumerate through the dataloader
        
        images, targets = d

        # pass image to model
        train_dict['optimizer'].zero_grad()
        outputs = train_dict['model'](images)
        
        # calculate loss
        loss = train_dict['loss_fn'](outputs, targets)
        
        # backpropagate
        loss.backward()
        
        # Use PyTorch XLA optimizer stepping
        xm.optimizer_step(train_dict['optimizer'])
        
        # Step the scheduler
        if train_dict['lr_schedule'] is not None: 
            train_dict['lr_schedule'].step()
    
    # since the loss is on all 8 cores, reduce the loss values and print the average
    loss_reduced = xm.mesh_reduce('loss_reduce',loss, lambda x: sum(x) / len(x)) 
    # master_print will only print once (not from all 8 cores)
    xm.master_print(f'bi={bi}, train loss={loss_reduced}')
        
    train_dict['model'].eval() # put model in eval mode for later use
    
def eval_loop_fn(train_dict):
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(train_dict['valid_loader']): # enumerate through dataloader
        
        images, targets = d

        # pass image to model
        with torch.no_grad(): outputs = train_dict['model'](images)

        # Add the outputs and targets to a list 
        targets_np = targets.cpu().detach().numpy().tolist()
        outputs_np = outputs.cpu().detach().numpy().tolist()
        fin_targets.extend(targets_np)
        fin_outputs.extend(outputs_np)    
        del targets_np, outputs_np
        gc.collect() # delete for memory conservation
                
    o,t = np.array(fin_outputs), np.array(fin_targets)
    
    # calculate loss
    loss = train_dict['loss_fn'](torch.tensor(o), torch.tensor(t))
    # since the loss is on all 8 cores, reduce the loss values and print the average
    loss_reduced = xm.mesh_reduce('loss_reduce',loss, lambda x: sum(x) / len(x)) 
    # master_print will only print once (not from all 8 cores)
    xm.master_print(f'val. loss={loss_reduced}')
    
    acc = metrics.accuracy_score(t,o.argmax(axis=1))
    acc_reduced = xm.mesh_reduce('acc_reduce', acc, lambda x: sum(x) / len(x))
        
    xm.master_print(f'val. accuracy = {acc_reduced}')

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
    
    gc.collect()
    
    xm.master_print(f'========== training fold {FLAGS["fold"]} for {FLAGS["epochs"]} epochs ==========')
    for i in range(FLAGS['epochs']):
        xm.master_print(f'EPOCH {i}:')
        # train one epoch
        train_loop_fn(train_dict)
                
        # validation one epoch
        eval_loop_fn(train_dict)

        gc.collect()
    
    xm.rendezvous('save_model')
    
    xm.master_print('save model')
    
    xm.save(train_dict['model'].state_dict(), f'xla_trained_model_{FLAGS["epochs"]}_epochs_fold_{FLAGS["fold"]}.pth')

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
    'epochs': 30, 
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