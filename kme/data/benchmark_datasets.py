from torch.utils.data import Dataset
import wget, os
import pandas as pd
import torch, numpy as np


class HeartDataset(Dataset):

    def __init__(self, root):
        fname = os.path.join(root, 'HeartDisease.csv')
        if not os.path.exists(fname):
            raise('Please download the data from https://www.kaggle.com/sonumj/heart-disease-dataset-from-uci !')

        self.df = pd.read_csv(fname, header=0)
        train_cols = self.df.columns[0:-2]
        self.labels = self.df.columns[-2]
        self.labels = self.df[self.labels]
        self.df = self.df[train_cols]
        for col_name in self.df.columns:
            self.df[col_name].fillna(self.df[col_name].mode()[0], inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return torch.FloatTensor(self.df.iloc[idx].values.astype(float)), self.labels[idx].astype(np.long)


class CreditCardFraudDataset(Dataset):

    def __init__(self, root):
        fname = os.path.join(root, 'creditcard.csv')
        if not os.path.exists(fname):
            raise('Please download the data from https://www.kaggle.com/mlg-ulb/creditcardfraud !')

        self.df = pd.read_csv(fname, header=0)
        self.df = self.df.dropna()
        self.labels = self.df[self.df.columns[-1]]
        self.df = self.df[self.df.columns[0:-1]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return torch.FloatTensor(self.df.iloc[idx].values.astype(float)), self.labels[idx].astype(np.long)
