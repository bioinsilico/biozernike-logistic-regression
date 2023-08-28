from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

d_type = np.float32


class BiozernikeDataset(Dataset):

    def __init__(self, coefficients_file_name):
        self.descriptor_classes = list()
        self.descriptors = list()
        self.descriptor_pairs = list()

        df = pd.read_csv(coefficients_file_name, sep="\t", header=None)
        for i, row_i in df.iterrows():
            self.descriptor_classes.append(row_i[0])
            self.descriptors.append(np.array(row_i[1:3923], dtype=d_type))
        for i in range(len(self.descriptor_classes)):
            for j in range(i + 1, len(self.descriptor_classes)):
                self.descriptor_pairs.append([
                    1. if self.descriptor_classes[i] == self.descriptor_classes[j] else 0.,
                    i,
                    j
                ])

    def __len__(self):
        return len(self.descriptor_pairs)

    def __getitem__(self, idx):
        descriptor_i = self.descriptors[self.descriptor_pairs[idx][1]]
        geom_i = descriptor_i[0:17]
        cn_i = descriptor_i[17:]

        descriptor_j = self.descriptors[self.descriptor_pairs[idx][2]]
        geom_j = descriptor_j[0:17]
        cn_j = descriptor_j[17:]

        geom = 2 * np.absolute(geom_i - geom_j) / (1 + np.absolute(geom_i) + np.absolute(geom_j))
        cn = np.absolute(cn_i - cn_j)

        label = np.array([self.descriptor_pairs[idx][0]], dtype=d_type)

        return torch.from_numpy(np.concatenate((geom, cn))), torch.from_numpy(label)

    def weights(self):
        p = 0.5 / sum([dp[0] for dp in self.descriptor_pairs])
        n = 0.5 / sum([1 - dp[0] for dp in self.descriptor_pairs])
        return torch.tensor([p if dp[0] == 1. else n for dp in self.descriptor_pairs])
