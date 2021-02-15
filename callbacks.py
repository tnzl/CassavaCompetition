import torch_xla.core.xla_model as xm
import torch

class CallbackManager:

    def __init__(self, train_dict):
        #train_dict contains all the callbacks.
        self.train_dict = train_dict

    def remove_callback(self):
        pass

    #fit 
    def on_fit_begin(self, state_dict=None):
        for cb in self.train_dict['cbs']:
            cb.on_fit_begin(self.train_dict, state_dict)

    def on_fit_end(self, state_dict=None):
        for cb in self.train_dict['cbs']:
            cb.on_fit_end(self.train_dict, state_dict)

    #epoch
    def on_epoch_begin(self, epoch, state_dict=None):
        for cb in self.train_dict['cbs']:
            cb.on_epoch_begin(self.train_dict, epoch, state_dict)

    def on_epoch_end(self, epoch, state_dict=None):
        for cb in self.train_dict['cbs']:
            cb.on_epoch_end(self.train_dict, epoch, state_dict)

    #batch
    def on_batch_begin(self, batch, state_dict=None):
        for cb in self.train_dict['cbs']:
            cb.on_batch_begin(self.train_dict, batch, state_dict)

    def on_batch_end(self, batch, state_dict=None):
        for cb in self.train_dict['cbs']:
            cb.on_batch_end(self.train_dict, batch, state_dict)

class Callback:

    def __init__(self):
        pass
    
    #fit 
    def on_fit_begin(self, train_dict, state_dict=None):
        pass

    def on_fit_end(self, train_dict, state_dict=None):
        pass

    #epoch
    def on_epoch_begin(self, train_dict, epoch, state_dict=None):
        pass

    def on_epoch_end(self, train_dict, epoch, state_dict=None):
        pass
        
    #batch
    def on_batch_begin(self, train_dict, batch, state_dict=None):
        pass

    def on_batch_end(self, train_dict, batch, state_dict=None):
        pass

class PrintCallback(Callback):
    def __init__(self, logger=print):
        super().__init__()
        self.logger = logger

    #fit 
    def on_fit_begin(self, train_dict, state_dict=None):
        self.logger(f'Starting to fit...')

    def on_fit_end(self, train_dict, state_dict=None):
        self.logger(f'...fit complete!')

    #epoch
    def on_epoch_begin(self, train_dict, epoch, state_dict=None):
        self.logger(f'Starting epoch: {epoch}')

    def on_epoch_end(self, train_dict, epoch, state_dict=None):
        self.logger(f'Completed epoch : {epoch}, \n{state_dict}')
        
class CheckPrintCallback(Callback):
    def __init__(self, logger=print):
        super().__init__()
        self.logger = logger

    #fit 
    def on_fit_begin(self, train_dict, state_dict=None):
        self.logger(f'In on_fit_begin {state_dict}')

    def on_fit_end(self, train_dict, state_dict=None):
        self.logger(f'In on_fit_end {state_dict}')

    #epoch
    def on_epoch_begin(self, train_dict, epoch, state_dict=None):
        self.logger(f'In on_epoch_begin, {epoch} , {state_dict}')

    def on_epoch_end(self, train_dict, epoch, state_dict=None):
        self.logger(f'In on_epoch_end, {epoch}, {state_dict}')
        
    #batch
    def on_batch_begin(self, train_dict, batch, state_dict=None):
        if batch<=1:
            self.logger(f'In on_batch_begin, {batch}, {state_dict}')

    def on_batch_end(self, train_dict, batch, state_dict=None):
        if batch<=1:
            self.logger(f'In on_batch_end, {batch}, {state_dict}')

class UnfreezePattern(Callback):
    def __init__(self, unfreze_pattern):
        super().__init__()
        self.unfreze_pattern = unfreze_pattern
        self.param_len = None

    def unfreeze(self,train_dict, n):
        c = 0
        for p in train_dict['model'].parameters(): 
            if c > self.param_len-n-1:
                p.requires_grad = True
            c += 1

    def on_fit_begin(self, train_dict, state_dict=None):
        c = 0 
        for p in train_dict['model'].parameters():
            c += 1
        self.param_len = c

    def on_epoch_begin(self, train_dict, epoch, state_dict=None):
        self.unfreeze(train_dict, self.unfreze_pattern[epoch])

class ModelSaver(Callback):
    def __init__(self, epoch_freq=2):
        super().__init__()    
        self.epoch_freq = epoch_freq

    def on_epoch_end(self, train_dict, epoch, state_dict=None):
        if epoch%self.epoch_freq==0:
            mn=train_dict['flags']['model']
            ff=train_dict['flags']['fold']
            name = f'{mn}_epochs={epoch}_fold={ff}.pth'
            xm.rendezvous('save_model')
            xm.save(train_dict['model'].state_dict(), name)
            xm.master_print(f'{name} model saved.')
    
class RLPSchduler(Callback):
    def __init__(self):
        self.scheduler = None

    def on_fit_begin(self, train_dict, state_dict=None):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(train_dict['optimizer'],mode='min',factor=0.8
                                ,patience=1,threshold=0.0001,threshold_mode='abs',min_lr=1e-8,eps=1e-08)
        xm.master_print('Scheduler initialized.')

    def on_epoch_end(self, train_dict, epoch, state_dict=None):
        self.scheduler.step(state_dict['valid_loss'])
    