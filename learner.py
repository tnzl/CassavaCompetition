import torch 
import torch_xla.core.xla_model as xm
import numpy as np
import gc
import time
from sklearn import metrics
from callbacks import CallbackManager

class Learner:
    def __init__(self, model, optimizer, loss_fn, device, num_epochs, bs, train_dl, valid_dl=None, cbs=[], run_name="NA", verbose=True, tpu=False, seed=None, metrics=None, lr_schedule=None):
        '''
        Initialize.
        '''
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.metrics = metrics
        self.lr_schedule = lr_schedule
        self.seed = seed
        self.tpu = tpu
        self.num_epochs = num_epochs
        self.epoch = None
        self.verbose = verbose
        self.device = device
        self.run_name = run_name
        self.cb_manager = CallbackManager(self)
        self.cbs = cbs
        self.bs = bs
        self.logger = print if not self.tpu else xm.master_print
        
        if self.seed:
            self.seed_everything(self.seed)
        
        # self.model = self.model.to(self.device).double()
        
        self.verboser('Training on : '+str(self.device))
        # self.verboser('Number of callbacks = '+str(len(self.cbs)))

    def seed_everything(self,seed):
        '''
        Seed everything.
        '''
        self.verboser("seed_everything function not implemented...") 

    def verboser(self, msg):
        if self.verbose:
            self.logger(f'[Process {self.tpu}]:'+msg)
            return 
        return

    def save_model(self, name='NA.pth'):
        xm.rendezvous('save_model')
        xm.master_print('save model')
        xm.save(model.state_dict(), name)
        return

    def train_loop_fn(self):
        start_time = time.time()
        self.model.train() # put model in training mode
        fin_correct = []
        fin_loss = []
        for bi, d in enumerate(self.train_dl): # enumerate through the dataloader
            # xm.master_print(f'bi={bi}')
            # images = d['image'] # obtain the ids
            # targets = d['targets'] # obtain the target
            self.cb_manager.on_batch_begin(bi)
            images, targets = d 

            # pass image to model
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # calculate loss
            loss = self.loss_fn(outputs, targets)
            
            # backpropagate
            loss.backward()
            
            # Use PyTorch XLA optimizer stepping
            xm.optimizer_step(self.optimizer)
            
            # Step the scheduler
            if self.lr_schedule is not None: self.lr_schedule.step()
            
            #log
            fin_loss.append(loss.detach().cpu().item())
            fin_correct.append(torch.eq(targets, torch.argmax(outputs, 1)).sum().detach().cpu().item())
            sd = {
                'train_batch_loss': fin_loss[-1],
                'train_batch_corrects': fin_correct[-1]
            }

            self.cb_manager.on_batch_end(bi, sd)
        
        # since the loss is on all 8 cores, reduce the loss values and print the average
        loss_reduced = xm.mesh_reduce('loss_reduce',loss, lambda x: sum(x) / len(x)) 
        # master_print will only print once (not from all 8 cores)
        # xm.master_print(f'bi={bi}, train loss={loss_reduced}, train time={time.time()-start_time}')
        self.verboser(f'bi={bi}, train loss={loss_reduced}, train time={time.time()-start_time}')
            
        # model.eval() # put model in eval mode for later use

    def eval_loop_fn(self):
        start_time = time.time()
        fin_targets = []
        fin_outputs = []
        for bi, d in enumerate(self.valid_dl): # enumerate through dataloader
            
            # images = d['image'] # obtain the ids
            # targets = d['targets']# # obtain the targets

            images, targets = d 

            # pass image to model
            with torch.no_grad(): outputs = self.model(images)

            # Add the outputs and targets to a list 
            targets_np = targets.cpu().detach().numpy().tolist()
            outputs_np = outputs.cpu().detach().numpy().tolist()
            fin_targets.extend(targets_np)
            fin_outputs.extend(outputs_np)    
            del targets_np, outputs_np
            gc.collect() # delete for memory conservation
                    
        o,t = np.array(fin_outputs), np.array(fin_targets)
        
        # calculate loss
        loss = self.loss_fn(torch.tensor(o), torch.tensor(t))
        # since the loss is on all 8 cores, reduce the loss values and print the average
        loss_reduced = xm.mesh_reduce('loss_reduce',loss, lambda x: sum(x) / len(x)) 
        # master_print will only print once (not from all 8 cores)
        # xm.master_print(f'val. loss={loss_reduced}')
        self.verboser(f'val. loss={loss_reduced}')
        
        acc = metrics.accuracy_score(t,o.argmax(axis=1))
        acc_reduced = xm.mesh_reduce('acc_reduce', acc, lambda x: sum(x) / len(x))
            
        # xm.master_print(f'val. accuracy = {acc_reduced},\n val_time = {start_time-time.time()}')
        self.verboser(f'val. accuracy = {acc_reduced},\n val_time = {time.time()-start_time}')

    def fit(self):

        for i in range(self.num_epochs):
            self.epoch = i
            # sd = {'epoch' : i}
            # wandb_run.log(sd)
            self.cb_manager.on_epoch_begin(self.epoch)
            xm.master_print(f'EPOCH {i}:')
            # train one epoch
            self.train_loop_fn()
                    
            # validation one epoch
            self.eval_loop_fn()

            # val_stats.update(train_stats)
            self.cb_manager.on_epoch_end(self.epoch, state_dict=None)

            gc.collect()