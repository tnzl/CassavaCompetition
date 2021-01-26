import torch
import torchvision
from data import get_dl
from learner import Learner

flags = {}
flags['project'] = "cassava-leaf-disease-classification"
flags['run_name'] = 'try-xxx'
flags['data_root'] = '/kaggle/input/cassava-leaf-disease-classification'
flags['pin_memory'] = True
flags['img_size'] = 320  
flags['fold'] = 0
flags['model'] = 'resnext50_32x4d'
flags['pretrained'] = True
flags['lr'] = 0.001,
flags['verbose'] : True
flags['img_size'] = 320
flags['batch_size'] = 32
flags['num_workers'] = 4
flags['seed'] = 1234
flags['debug'] = True
flags['num_epochs'] = 2 if flags['debug'] else 5

net = torchvision.models.resnet18(pretrained=True).double()
count = 0
for param in net.parameters():
    count +=1
    # print(param.shape,count)
    param.requires_grad = False
net.fc = torch.nn.Linear(net.fc.in_features, 5)

optimizer=torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
wandb_run = None

learner = Learner(net, 
                optimizer=optimizer, 
                loss_fn=torch.nn.CrossEntropyLoss(), 
                dl=get_dl(flags), 
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
                num_epochs=flags['num_epochs'], 
                bs=flags['batch_size'], 
                verbose=True, 
                tpu=False, 
                seed=1234, 
                metrics=None, 
                lr_schedule=torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
                wandb_run=wandb_run)

learner.fit()