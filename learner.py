import torch 
import torch_xla.core.xla_model as xm
import numpy as np
import gc
import time
from sklearn import metrics
from callbacks import CallbackManager

def train_loop_fn(train_dict):
    train_dict['model'].train() # put model in training mode
    for bi, d in enumerate(train_dict['train_loader']): # enumerate through the dataloader
        train_dict['cb_manager'].on_batch_begin(bi, state_dict=None)
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
        
        with torch.no_grad():
            train_dict['cb_manager'].on_batch_end(bi, state_dict={'loss': loss.detach().cpu().item()})
    
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

def fit(train_dict):
    train_dict['cb_manager'].on_fit_begin(state_dict=None)
    for i in range(train_dict['flags']['epochs']):
        train_dict['cb_manager'].on_epoch_begin(i, state_dict=None)
        es = time.time()
        xm.master_print(f'EPOCH {i}:')
        # train one epoch
        train_loop_fn(train_dict)
                
        # validation one epoch
        eval_loop_fn(train_dict)

        gc.collect()
        xm.master_print(f'Epoch {i} time = {time.time()-es}')
        train_dict['cb_manager'].on_epoch_end(i, state_dict=None)
    train_dict['cb_manager'].on_fit_end(state_dict=None)
        