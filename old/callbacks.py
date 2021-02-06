class CallbackManager:

    def __init__(self, learner):
        #learner contains all the callbacks.
        self.learner = learner

    def remove_callback(self):
        pass

    #fit 
    def on_fit_begin(self, state_dict=None):
        for cb in self.learner.cbs:
            cb.on_fit_begin(self.learner, state_dict)

    def on_fit_end(self, state_dict=None):
        for cb in self.learner.cbs:
            cb.on_fit_end(self.learner, state_dict)

    #epoch
    def on_epoch_begin(self, epoch, state_dict=None):
        for cb in self.learner.cbs:
            cb.on_epoch_begin(self.learner, epoch, state_dict)

    def on_epoch_end(self, epoch, state_dict=None):
        for cb in self.learner.cbs:
            cb.on_epoch_end(self.learner, epoch, state_dict)

    #batch
    def on_batch_begin(self, batch, state_dict=None):
        for cb in self.learner.cbs:
            cb.on_batch_begin(self.learner, batch, state_dict)

    def on_batch_end(self, batch, state_dict=None):
        for cb in self.learner.cbs:
            cb.on_batch_end(self.learner, batch, state_dict)

class Callback:

    def __init__(self):
        pass
    
    #fit 
    def on_fit_begin(self, learner, state_dict=None):
        pass

    def on_fit_end(self, learner, state_dict=None):
        pass

    #epoch
    def on_epoch_begin(self, learner, epoch, state_dict=None):
        pass

    def on_epoch_end(self, learner, epoch, state_dict=None):
        pass
        
    #batch
    def on_batch_begin(self, learner, batch, state_dict=None):
        pass

    def on_batch_end(self, learner, batch, state_dict=None):
        pass

class PrintCallback(Callback):

    #fit 
    def on_fit_begin(self, learner, state_dict=None):
        print('In on_fit_begin', state_dict)

    def on_fit_end(self, learner, state_dict=None):
        print('In on_fit_end', state_dict)

    #epoch
    def on_epoch_begin(self, learner, epoch, state_dict=None):
        print('In on_epoch_begin', epoch , state_dict)

    def on_epoch_end(self, learner, epoch, state_dict=None):
        print('In on_epoch_end', epoch, state_dict)
        
    #batch
    def on_batch_begin(self, learner, batch, state_dict=None):
        if batch<=1:
            print('In on_batch_begin', batch, state_dict)

    def on_batch_end(self, learner, batch, state_dict=None):
        if batch<=1:
            print('In on_batch_end', batch, state_dict)

class WandbCallback(Callback):

    def __init__(self, wandb_run):
        super().__init__()
        self.wandb_run = wandb_run

    def on_fit_begin(self, learner, state_dict=None):
        self.wandb_run.watch(learner.net)
    
    def on_batch_end(self, learner, batch, state_dict=None):
        s = {}
        for key in list(state_dict.keys()):
            s[key+'_epoch='+str(learner.epoch)] = state_dict[key]
        s['batch'] = batch
        self.wandb_run.log(s)

    def on_epoch_end(self, learner, epoch, state_dict=None):
        state_dict['epoch'] = epoch
        self.wandb_run.log(state_dict)



    