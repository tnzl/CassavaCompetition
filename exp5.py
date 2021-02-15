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
from callbacks import CallbackManager, PrintCallback, UnfreezePattern, ModelSaver, RLPSchduler

os.environ['XLA_USE_BF16']="1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0.1, dim=-1): 
        super(LabelSmoothing, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = 5 
        self.dim = dim 
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, pred, target): 
        """Taylor Softmax and log are already applied on the logits"""
        #pred = pred.log_softmax(dim=self.dim) 
        return self.loss_fn(self.confidence*pred+self.smoothing/5, target)
    
from efficientnet_pytorch import EfficientNet
class Net(nn.Module):
    def __init__(self,model_name='efficientnet-b3',pool_type=F.adaptive_avg_pool2d):
        super().__init__()
        self.pool_type = pool_type
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_features = getattr(self.backbone,'_fc').in_features
        self.classifier = nn.Linear(in_features,5)
    def forward(self,x):
        features = self.pool_type(self.backbone.extract_features(x),1)
        features = features.view(x.size(0),-1)
        return self.classifier(features)

def run(rank, flags):
    global FLAGS
    torch.set_default_tensor_type('torch.FloatTensor')
    
    train_dict = {
        'flags' : flags,
        'device' : xm.xla_device(),
        'loss_fn' : LabelSmoothing()
    }
    train_dict['train_loader'] = pl.MpDeviceLoader(get_dl(flags, fold=flags['fold'], aim='train'), train_dict['device'])
    train_dict['valid_loader'] = pl.MpDeviceLoader(get_dl(flags, fold=flags['fold'], aim='valid'), train_dict['device'])
    train_dict['model'] = MX.to(train_dict['device'])
    train_dict['optimizer'] = Adam(train_dict['model'].parameters(), lr=flags['lr']*xm.xrt_world_size()) 
    # train_dict['lr_schedule'] = torch.optim.lr_scheduler.CosineAnnealingLR(train_dict['optimizer'], len(train_dict['train_loader'])*flags['epochs'])
    train_dict['lr_schedule'] = None
    train_dict['cb_manager'] = CallbackManager(train_dict)
    train_dict['cbs'] = [PrintCallback(logger=xm.master_print), ModelSaver(epoch_freq=5), RLPSchduler()]
    gc.collect()
    
    xm.master_print(f'========== training fold {FLAGS["fold"]} for {FLAGS["epochs"]} epochs ==========')

    fit(train_dict, continue_from=flags['continue_from'])

    xm.rendezvous('save_model')
    
    # xm.master_print('save model')
    
    xm.save(train_dict['model'].state_dict(), f'xla_trained_model_{FLAGS["epochs"]}_epochs_fold_{FLAGS["fold"]}.pth')

FLAGS = {
    'project' : "cassava-leaf-disease-classification",
    'run_name' : 'resnext',
    'train_folds_path' : './train_folds.csv',
    'training_data_path' : "/kaggle/input/cassava-jpeg-256x256/kaggle/train_images_jpeg",
    'fold': 0,
    'model': 'effnet:b3',
    'unfreeze_pattern' : [2]*3+list(range(5,12,3))+list(range(17, 72, 6))+list(range(80, 162, 9))+[161]*2,
    'pretrained': True,
    'batch_size': 16,
    'num_workers': 4,
    'lr': 3e-4,
    'epochs': 20, 
    'seed':1111,
    'continue_from' : 0
}
net = Net()
MX = xmp.MpModelWrapper(net)

# Spawn processes
# wandb_run = wandb.init(project=FLAGS['project'], name=FLAGS['run_name'], config=FLAGS)
start_time = time.time()
xmp.spawn(run, args=(FLAGS,), nprocs=8, start_method='fork')
print('time taken: ', time.time()-start_time)