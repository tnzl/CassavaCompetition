import torch 
import torch_xla.core.xla_model as xm
import numpy as np
import gc
import time
from sklearn import metrics
from callbacks import CallbackManager

def train_loop_fn(train_dict):
    train_dict['model'].train() # put model in training mode
    fin_correct = []
    fin_loss = []
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
            fin_loss.append(loss.detach().cpu().item())
            fin_correct.append(torch.eq(targets, torch.argmax(outputs, 1)).sum().item())
            train_dict['cb_manager'].on_batch_end(bi, state_dict={'train_batch_loss':fin_loss[-1], 'train_batch_num_corrects':fin_correct[-1]})
    
    sd = {
        'train_epoch_loss' : sum(fin_loss)/len(fin_loss),
        'train_epoch_accuracy' : sum(fin_correct)/(len(fin_correct)*train_dict['flags']['batch_size'])
    }
    # since the loss is on all 8 cores, reduce the loss values and print the average
    sd['train_epoch_loss'] = xm.mesh_reduce('loss_reduce',sd['train_epoch_loss'], lambda x: sum(x) / len(x)) 
    sd['train_epoch_accuracy'] = xm.mesh_reduce('acc_reduce',sd['train_epoch_accuracy'], lambda x: sum(x) / len(x)) 
    # master_print will only print once (not from all 8 cores)
    # xm.master_print(f'bi={bi}, train loss={loss_reduced}')
    return sd
    
    
def eval_loop_fn(train_dict):
    train_dict['model'].eval() # put model in eval mode for later use
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
    # xm.master_print(f'val. loss={loss_reduced}')
    
    acc = metrics.accuracy_score(t,o.argmax(axis=1))
    acc_reduced = xm.mesh_reduce('acc_reduce', acc, lambda x: sum(x) / len(x))
        
    # xm.master_print(f'val. accuracy = {acc_reduced}')
    sd = {
        'valid_loss' : loss_reduced.item(),
        'valid_accuracy' : acc_reduced
    }
    return sd

def fit(train_dict):
    train_dict['cb_manager'].on_fit_begin(state_dict=None)
    for i in range(train_dict['flags']['epochs']):
        train_dict['cb_manager'].on_epoch_begin(i, state_dict=None)
        es = time.time()
        # xm.master_print(f'EPOCH {i}:')
        # train one epoch
        train_sd = train_loop_fn(train_dict)
                
        # validation one epoch
        valid_sd = eval_loop_fn(train_dict)

        gc.collect()
        # xm.master_print(f'Epoch {i} time = {time.time()-es}')
        train_sd.update(valid_sd)
        train_dict['cb_manager'].on_epoch_end(i, state_dict=train_sd)
    train_dict['cb_manager'].on_fit_end(state_dict=None)
        