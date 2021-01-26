

class CallbackManager:

    def __init__(self, learner):
        #learner contains all the callbacks.
        self.learner = learner

    def remove_callback(self):
        pass

    #fit 
    def on_fit_begin(self, state_dict=None):
        for cb in self.learner.callbacks:
            cb.on_fit_begin(self.learner, state_dict)

    def on_fit_end(self, state_dict=None):
        for cb in self.learner.callbacks:
            cb.on_fit_end(self.learner, state_dict)

    #epoch
    def on_epoch_begin(self, epoch, state_dict=None):
        for cb in self.learner.callbacks:
            cb.on_epoch_begin(self.learner, epoch, state_dict)

    def on_epoch_end(self, epoch, state_dict=None):
        for cb in self.learner.callbacks:
            cb.on_epoch_end(self.learner, epoch, state_dict)

    #batch
    def on_batch_begin(self, batch, state_dict=None):
        for cb in self.learner.callbacks:
            cb.on_batch_begin(self.learner, batch, state_dict)

    def on_batch_end(self, batch, state_dict=None):
        for cb in self.learner.callbacks:
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
        
    #batch
    def on_batch_begin(self, learner, batch, state_dict=None):
        pass

    def on_batch_end(self, learner, batch, state_dict=None):
        pass




    