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
