# core manager for training steps and tying model to optimizer and loss

class Reactor:
    def __init__( model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.isTraining = True

    @property
    def model(self):
        return self.model
    
    @property
    def optimizer(self):
        return self.optimizer
    
    @property
    def criterion(self):
        return self.criterion
    
    def step(self):
        return self.optimizer.step()

    def backward(self, loss):
        result = self.optimizer.backward(loss)
        return result
    
    def calc_loss(self, *args, **kwargs):
        return self.criterion(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self):
        self.isTraining = True
        self.model.train()
    
    def eval(self):
        self.isTraining = False
        self.model.eval()
    
    

