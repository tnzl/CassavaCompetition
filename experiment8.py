from learner import Learner
from data import data
import wandb

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

def map_fn(index, flags, wandb_run):
    
    data = Data(
        data_root=flags['data_root'], 
        num_workers=flags['num_workers'], 
        bs=flags['batch_size'], 
        debug=flags['debug'], 
        sampler=None, 
        transforms=None, 
        fold=0, 
        num_folds=5, 
        img_size=flags['img_size'], 
        tpu=True
        )

    net = torchvision.models.resnet18(pretrained=True).double()
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, 5)
    optimizer=torch.optim.SGD(net.parameters(), lr=0.001*xm.xrt_world_size(), momentum=0.9)
    
    xm.rendezvous('barrier-1')
    learner = Learner(net, 
                      optimizer=optimizer, 
                      loss_fn=torch.nn.CrossEntropyLoss(), 
                      dl=data.get_dl(), 
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
    learner.verboser("Complete!")
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
    'num_workers': 2,
    'lr': 0.001,
    'seed' : 1234,
    'verbose' : True
}
flags['img_size'] = 320
flags['batch_size'] = 32
flags['num_workers'] = 4
flags['seed'] = 1234
flags['debug'] = False
flags['num_epochs'] = 2 if flags['debug'] else 5

# wandb_run = wandb.init(project=flags['project'], name=flags['run_name'], config=flags)
wandb_run = None
xmp.spawn(map_fn, args=(flags,wandb_run,), nprocs=8, start_method='fork')
