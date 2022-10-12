import wget
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


BASE_URL = 'https://github.com/ustunb/risk-slim/raw/master/examples/data/{}_data.csv'


class RiskDataset(Dataset):
    """
    Datasets used by Ustun and Rudin in
    'Learning Optimized Risk Scores' (https://arxiv.org/abs/1610.00168)
    """
    def __init__(self, root, dataset, transform=None):
        url = BASE_URL.format(dataset)
        fname = wget.download(url, out=root)
        self._transform = transform
        self._dataset = pd.read_csv(fname, sep=',', header=0, dtype=np.float)

    def __getitem__(self, idx):
        row = np.array(self._dataset.iloc[idx])
        y = int(row[0])
        x = torch.FloatTensor(row[1:])
        if self._transform is not None:
            x = self._transform(x)
        return x, y

    def __len__(self):
        return len(self._dataset)
