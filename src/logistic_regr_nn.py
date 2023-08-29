import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def get_weights(self):
        return self.linear.weight

    def get_bias(self):
        return self.linear.bias

    def __init__(self, input_features=1):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class LogisticRegressionUniform(nn.Module):

    def get_weights(self):
        return self.linear.weight

    def get_bias(self):
        return self.linear.bias

    def __init__(self, input_features=1):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)
        torch.nn.init.uniform_(self.linear.weight, a=0, b=1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))