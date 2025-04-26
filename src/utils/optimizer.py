import torch.optim as optim
from torch import nn


class Optimizer:
    def __init__(self, model):
        self.model = model
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.create_loss_function()
        self.create_optimizer()
        self.create_scheduler()
        self.report()

    def create_loss_function(self):
        self.loss_function = nn.CrossEntropyLoss()

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def create_scheduler(self):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def report(self):
        print("Loss function, Optimizer and Scheduler are ready!")