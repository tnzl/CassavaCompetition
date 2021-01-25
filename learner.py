#misc
import time

#torch
import torch

# xla 
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

class Learner():
    
    def __init__(self, net, optimizer, loss_fn, dl, device, num_epochs, bs=8, run_name="NA", verbose=True, tpu=False, seed=None, metrics=None, lr_schedule=None, wandb_run=None):
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
        self.wandb_run = wandb_run
        self.device = device
        self.run_name = run_name
        
        if self.seed:
            self.seed_everything(self.seed)
        
        self.net = self.net.to(self.device)
        
        if self.tpu:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
    
    def verboser(self, msg):
        if self.wandb_run:
            self.wandb_run.log({f'[Process {self.tpu}]': msg})
        if self.verbose:
            print(f'[Process {self.tpu}]:'+msg)
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
            if self.verbose:
                xm.master_print('save model')
            xm.save(self.net.state_dict(), name)
        else:
            torch.save(self.net.state_dict(), name)
        return 
    
    def load_model(self, path):
        pass
    
    def train_one_epoch(self):
        '''
        Train the model for one epoch.
        '''
        self.net = self.net.train()
        
        #Dataloader
        dl = self.dl['val']
        if self.tpu:
            dl = pl.ParallelLoader(dl, [self.device]).per_device_loader(self.device)
        
        #Stats init
        num_correct = 0
        total_guesses = 0
        cum_loss = 0
        
        #train
        for batch_num, batch in enumerate(dl):
            data, targets = batch 
            if not self.tpu:
                data = data.to(self.device)
                targets = targets.to(self.device)
            output = self.net(data)
            loss = self.loss_fn(output, targets)
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.tpu:
                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()
                
            cum_loss += loss.item()
            best_guesses = torch.argmax(output, 1)
            num_correct += torch.eq(targets, best_guesses).sum().item()
            total_guesses += len(targets)
        
        #stats
        stats = {
            'train_accuracy' : num_correct/total_guesses * 100,
            'train_loss' : cum_loss/total_guesses
        }
        return stats
    
    def validate(self):
        '''
        Validate the model.
        '''
        self.net = self.net.eval()
        eval_start = time.time()
        with torch.no_grad():
            #stats
            num_correct = 0
            total_guesses = 0
            cum_loss = 0
            
            dl = self.dl['val']
            if self.tpu:
                dl = pl.ParallelLoader(dl, [self.device]).per_device_loader(self.device)

            for batch_num, batch in enumerate(dl):
                data, targets = batch
                if not self.tpu:
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                output = self.net(data)
                best_guesses = torch.argmax(output, 1)
                cum_loss += self.loss_fn(output, targets).item()
                num_correct += torch.eq(targets, best_guesses).sum().item()
                total_guesses += len(targets)

        elapsed_eval_time = time.time() - eval_start
        self.verboser(f"Finished evaluation. Evaluation time was: {elapsed_eval_time}" )
        self.verboser(f"Guessed{num_correct} of {total_guesses} correctly for {num_correct/total_guesses * 100}% accuracy and loss of {cum_loss/total_guesses}.")

        stats = {
            'val_accuracy' : num_correct/total_guesses * 100,
            'val_loss' : cum_loss/total_guesses
        }
        return stats
    
    def fit(self):
        '''
        Run the complete training loop.
        '''
        self.verboser('Starting to fit...')
            
        train_start = time.time()
        torch.manual_seed(self.seed)    
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            #train
            train_stats = self.train_one_epoch()
            train_stats['epoch'] = epoch
            if self.wandb_run:
                wandb_run.log(train_stats)
            
            #validate
            val_start = time.time()
            val_stats = self.validate()
            val_stats['epoch'] = epoch
            epoch_end = time.time()
            if self.wandb_run:
                wandb_run.log(val_stats)
            
            if self.lr_schedule:
                self.lr_schedule.step()
            self.verboser(f"Finished epoch {epoch} "+
                          f"\n Epoch train time was: {epoch_end-epoch_start}"+
                          f"\n Train acc : {train_stats['train_accuracy']}"+
                          f"\n Train loss : {train_stats['train_loss']}"+
                          f"\n Val acc : {val_stats['val_accuracy']}"+
                          f"\n Val loss : {val_stats['val_loss']}") 
            
            
            
        
        #save model
        self.save_model()
        
        self.verboser(f"Finished training. Train time was: {time.time() - train_start}") 
        return