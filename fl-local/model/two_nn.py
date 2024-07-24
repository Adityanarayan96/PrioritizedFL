import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn

class Two_NN(torch.nn.Module):
    def __init__(self, input_dim=784, num_hiddens=200, only_digits=False):
        super(Two_NN, self).__init__()
        self.input_dim = input_dim
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.input_dim, num_hiddens)
        self.linear2 = torch.nn.Linear(num_hiddens, num_hiddens)
        self.out = torch.nn.Linear(num_hiddens, 10 if only_digits else 47)

    def forward(self, x):
        x = x.reshape([-1, self.input_dim])
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.out(x)
        return x
