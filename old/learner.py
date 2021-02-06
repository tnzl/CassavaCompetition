#misc
import time
import gc

#torch
import torch
import numpy as np
from sklearn import metrics

#my
from callbacks import Callback, CallbackManager

# xla 
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
except:
    pass

class Learner:
    def __init__(self, net, optimizer, loss_fn, dl, device, num_epochs, bs, cbs=[], run_name="NA", verbose=True, tpu=False, seed=None, metrics=None, lr_schedule=None):
        '''
        Initialize.
        '''
        self.net = net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dl = dl
        self.metrics = metrics
        self.lr_schedule = lr_schedule
        self.seed = seed
        self.tpu = tpu
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.device = device
        self.run_name = run_name
        self.epoch = None
        self.cb_manager = CallbackManager(self)
        self.cbs = cbs
        self.bs = bs
        
        if self.seed:
            self.seed_everything(self.seed)
        
        self.net = self.net.to(self.device).double()
        
        self.verboser('Training on : '+str(self.device))
        self.verboser('Number of callbacks = '+str(len(self.cbs)))
        
    
    def verboser(self, msg, logger= print):

        if self.verbose:
            logger(f'[Process {self.tpu}]:'+msg)
            return 
        return
            
    def seed_everything(self,seed):
        '''
        Seed everything.
        '''
        self.verboser("seed_everything function not implemented...")
    
    def save_model(self):
        name = self.run_name+f'_{self.num_epochs}_epochs.pth'
        if self.tpu:
            xm.rendezvous('save_model')
            # if self.verbose:
            #     xm.master_print('save model')
            xm.save(self.net.state_dict(), name)
        else:
            torch.save(self.net.state_dict(), name)
        return 
    
    def load_model(self, path):
        pass

    def train_one_epoch(self):
        self.net = self.net.train() # put model in training mode
        dl = self.dl['train']
        fin_correct = []
        fin_loss = []
        start_time = time.time()

        if self.tpu:
            dl = pl.MpDeviceLoader(dl, self.device)

        for bi, batch in enumerate(dl):
            self.cb_manager.on_batch_begin(bi)
            images, targets = batch

            # pass image to model
            self.optimizer.zero_grad()
            outputs = self.net(images)
            
            # calculate loss
            loss = self.loss_fn(outputs, targets)
            
            # backpropagate
            loss.backward()
            
            # Use PyTorch XLA optimizer stepping
            if self.tpu:
                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()
            
            # Step the scheduler
            if self.lr_schedule is not None: self.lr_schedule.step()
            
            #log
            fin_loss.append(loss.detach().cpu().item())
            fin_correct.append(torch.eq(targets, torch.argmax(outputs, 1)).sum().detach().cpu().item())
            sd = {
                'train_batch_loss': fin_loss[-1],
                'train_batch_corrects': fin_correct[-1]
            }

            #clear mem
            del loss, outputs, targets, images
            gc.collect()

            self.cb_manager.on_batch_end(bi, sd)

        loss = sum(fin_loss) / len(fin_loss)
        acc = sum(fin_correct) / (self.bs * (bi + 1))
        if self.tpu:
            loss = xm.mesh_reduce('loss_reduce',loss, lambda x: sum(x) / len(x)) 
            acc = xm.mesh_reduce('acc_reduce',acc, lambda x: sum(x) / len(x))
            self.verboser(f'Finished epoch {self.epoch} train. Train time was: {time.time()-start_time} and bi: {bi}', logger=xm.master_print)
        stats = {
            'train_accuracy' : acc * 100,
            'train_loss' : loss
        }
        #clear mem
        del acc, loss, fin_correct, fin_loss, dl
        gc.collect()
        
        return stats

    # def train_one_epoch(self):
    #     '''
    #     Train the model for one epoch.
    #     '''
    #     self.net = self.net.train()
        
    #     #Dataloader
    #     dl = self.dl['train']
    #     if self.tpu:
    #         dl = pl.ParallelLoader(dl, [self.device]).per_device_loader(self.device)
        
    #     #Stats init
    #     num_correct = 0
    #     total_guesses = 0
    #     cum_loss = 0
        
    #     #train
    #     for batch_num, batch in enumerate(dl):
    #         self.cb_manager.on_batch_begin(batch_num)
    #         data, targets = batch 
    #         if not self.tpu:
    #             data = data.to(self.device)
    #             targets = targets.to(self.device)
    #         output = self.net(data)
    #         loss = self.loss_fn(output, targets)
    #         self.optimizer.zero_grad()
    #         loss.backward()
            
    #         if self.tpu:
    #             xm.optimizer_step(self.optimizer)
    #         else:
    #             self.optimizer.step()
                
    #         cum_loss += loss.detach().item()
    #         best_guesses = torch.argmax(output, 1)
    #         batch_corrects = torch.eq(targets, torch.argmax(output, 1)).sum().item()
    #         num_correct += batch_corrects
    #         total_guesses += len(targets)
    #         sd = {
    #             'train_batch_loss' : loss.detach().item(),
    #             'train_batch_corrects' : batch_corrects
    #         }
    #         self.cb_manager.on_batch_end(batch_num, sd)
    #         del data, targets, output, best_guesses, batch_corrects, loss, sd
    #         gc.collect()
    #     #stats
    #     stats = {
    #         'train_accuracy' : num_correct/total_guesses * 100,
    #         'train_loss' : cum_loss/total_guesses
    #     }
    #     #clear mem
    #     del num_correct, cum_loss, total_guesses, dl
    #     gc.collect()
    #     return stats
    
    def validate(self):
        '''
        Validate
        '''
        self.net = self.net.eval()
        eval_start = time.time()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            dl = self.dl['val']
            if self.tpu:
                dl = pl.ParallelLoader(dl, [self.device]).per_device_loader(self.device)

            for batch_num, batch in enumerate(dl):
                data, targets = batch
                if not self.tpu:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                output = self.net(data)

                targets_np = targets.cpu().detach().numpy().tolist()
                outputs_np = output.cpu().detach().numpy().tolist()

                fin_targets.extend(targets_np)
                fin_outputs.extend(outputs_np)

                del targets_np, outputs_np
                gc.collect()
        
        o,t = np.array(fin_outputs), np.array(fin_targets)
        
        loss = self.loss_fn(torch.tensor(o), torch.tensor(t))
        if self.tpu:
            loss = xm.mesh_reduce('loss_reduce',loss, lambda x: sum(x) / len(x))

        acc = metrics.accuracy_score(t,o.argmax(axis=1))
        if self.tpu:
            acc = xm.mesh_reduce('acc_reduce', acc, lambda x: sum(x) / len(x))
        
        stats = {
            'val_loss' : loss,
            'val_accuracy' : acc*100
        }

        self.verboser(f"Finished evaluation. Evaluation time was: {time.time() - eval_start}") 
        del acc, loss, o, t, fin_outputs, fin_targets
        gc.collect()

        return stats
    
    def fit(self):
        '''
        Run the complete training loop.
        '''
        self.cb_manager.on_fit_begin()

        self.verboser('Starting to fit...')
            
        train_start = time.time()
        torch.manual_seed(self.seed)    
        
        for epoch in range(self.num_epochs):
            self.cb_manager.on_epoch_begin(epoch)

            epoch_start = time.time()
            self.epoch = epoch

            #train
            train_stats = self.train_one_epoch()
            
            #validate
            val_start = time.time()
            val_stats = self.validate()
            epoch_end = time.time()
            
            if self.lr_schedule:
                self.lr_schedule.step()
            self.verboser(f"Finished epoch {epoch} "+
                          f"\n Epoch train time was: {epoch_end-epoch_start}"+
                          f"\n Train acc : {train_stats['train_accuracy']}"+
                          f"\n Train loss : {train_stats['train_loss']}"+
                          f"\n Val acc : {val_stats['val_accuracy']}"+
                          f"\n Val loss : {val_stats['val_loss']}") 
            
            val_stats.update(train_stats)
            self.cb_manager.on_epoch_end(epoch, state_dict=val_stats)
            
            del epoch_end, train_stats, val_start, val_stats, epoch_start
            gc.collect()
        
        #save model
        self.save_model()
        
        self.verboser(f"Finished training. Train time was: {time.time() - train_start}") 

        self.cb_manager.on_fit_end()

        gc.collect()
        return