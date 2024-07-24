import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn

class LogisticRegression_S(nn.Module):
    def __init__(self, input_dim = 60, output_dim = 10):
        super(LogisticRegression_S, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x) #Multiclass classification
        return out
    
class LogisticRegression_F(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(LogisticRegression_F, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten the input tensor
        out = self.linear(x) #Multiclass classification
        return out
