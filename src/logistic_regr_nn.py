import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def get_weights(self):
        return self.linear.weight

    def __init__(self, input_features=1):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
