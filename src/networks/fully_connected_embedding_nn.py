import torch.nn as nn
from collections import OrderedDict


class EmbeddingCosine(nn.Module):

    def get_params(self):
        decay = []
        no_decay = []
        decay.append(self.embedding.lin.weight)
        decay.append(self.embedding.norm.weight)
        no_decay.append(self.embedding.lin.bias)
        no_decay.append(self.embedding.norm.bias)
        return {'params': no_decay, 'weight_decay': 0}, {'params': decay}

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]

    def __init__(self, input_features=1, hidden_layer=512):
        super().__init__()
        self.embedding = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(input_features)),
            ('lin', nn.Linear(input_features, hidden_layer)),
            ('relu', nn.ReLU())
        ]))

    def forward(self, x, y):
        return nn.functional.cosine_similarity(
            self.embedding(x),
            self.embedding(y)
        )


class RawEmbeddingCosine(nn.Module):

    def get_params(self):
        decay = []
        no_decay = []
        decay.append(self.embedding.lin.weight)
        no_decay.append(self.embedding.lin.bias)
        return {'params': no_decay, 'weight_decay': 0}, {'params': decay}

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]

    def __init__(self, input_features=1, hidden_layer=512):
        super().__init__()
        self.embedding = nn.Sequential(OrderedDict([
            ('lin', nn.Linear(input_features, hidden_layer)),
            ('relu', nn.Sigmoid())
        ]))

    def forward(self, x, y):
        return nn.functional.cosine_similarity(
            self.embedding(x),
            self.embedding(y)
        )
