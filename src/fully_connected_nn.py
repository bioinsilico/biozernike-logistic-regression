import torch
import torch.nn as nn
from collections import OrderedDict


class FullyConnected(nn.Module):

    def get_params(self):
        decay = []
        no_decay = []
        decay.append(self.model.lin1.weight)
        decay.append(self.model.lin2.weight)
        no_decay.append(self.model.lin1.bias)
        no_decay.append(self.model.lin2.bias)
        return {'params': no_decay, 'weight_decay': 0}, {'params': decay}

    def get_weights(self):
        return [(name, param) for name, param in self.model.named_parameters()]

    def __init__(self, input_features=1, hidden_layer=512):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(input_features, hidden_layer)),
            ('relu', nn.ReLU()),
            ('lin2', nn.Linear(hidden_layer, 2)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, x):
        return self.model(x)


class FullyConnectedSigmoid(nn.Module):

    def get_params(self):
        decay = []
        no_decay = []
        decay.append(self.model.lin1.weight)
        decay.append(self.model.lin2.weight)
        no_decay.append(self.model.lin1.bias)
        no_decay.append(self.model.lin2.bias)
        return {'params': no_decay, 'weight_decay': 0}, {'params': decay}

    def get_weights(self):
        return [(name, param) for name, param in self.model.named_parameters()]

    def __init__(self, input_features=1, hidden_layer=512):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(input_features, hidden_layer)),
            ('relu', nn.ReLU()),
            ('lin2', nn.Linear(hidden_layer, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        return self.model(x)


class FullySigmoid(nn.Module):

    def get_params(self):
        decay = []
        no_decay = []
        decay.append(self.model.lin1.weight)
        decay.append(self.model.lin2.weight)
        no_decay.append(self.model.lin1.bias)
        no_decay.append(self.model.lin2.bias)
        return {'params': no_decay, 'weight_decay': 0}, {'params': decay}

    def get_weights(self):
        return [(name, param) for name, param in self.model.named_parameters()]

    def __init__(self, input_features=1, hidden_layer=512):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(input_features, hidden_layer)),
            ('relu', nn.Sigmoid()),
            ('lin2', nn.Linear(hidden_layer, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        return self.model(x)


class LogisticRegression(nn.Module):

    def get_params(self):
        decay = []
        no_decay = []
        decay.append(self.linear.weight)
        no_decay.append(self.linear.bias)
        return {'params': no_decay, 'weight_decay': 0}, {'params': decay}

    def get_weights(self):
        return [("linear", self.linear.weight), ("bias", self.linear.bias)]

    def __init__(self, input_features=1):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))