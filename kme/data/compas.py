from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os, wget


DATA_URL = 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'
FNAME = 'propublica_data_for_fairml.csv'


class CompasDataset(Dataset):
    def __init__(self, root, verbose=True):
        """ProPublica Compas dataset.

        Dataset is read in from preprocessed compas data: `propublica_data_for_fairml.csv`
        from fairml github repo.
        Source url: 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'

        Following approach of Alvariz-Melis et al (SENN).

        Parameters
        ----------
        data_path : str
            Location of Compas data.
        """
        # check if already downloaded
        fpath = os.path.join(root, FNAME)
        if not os.path.exists(fpath):
            wget.download(DATA_URL, out=fpath)
        self.df = pd.read_csv(fpath)

        # don't know why square root
        self.df['Number_of_Priors'] = (self.df['Number_of_Priors'] / self.df['Number_of_Priors'].max()) ** (1 / 2)
        # get target
        compas_rating = self.df.score_factor.values  # This is the target?? (-_-)
        self.df = self.df.drop('score_factor', axis=1)

        pruned_df, pruned_rating = find_conflicting(self.df, compas_rating)
        if verbose:
            print('Finish preprocessing data..')

        self.X = pruned_df
        self.y = pruned_rating.astype(float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return torch.FloatTensor(self.X.iloc[idx].values.astype(float)), self.y[idx].astype(np.long)


def find_conflicting(df, labels, consensus_delta=0.2):
    """
    Find examples with same exact feature vector but different label.

    Finds pairs of examples in dataframe that differ only in a few feature values.

    From SENN authors' code.

    Parameters
    ----------
    df : pd.Dataframe
        Containing compas data.
    labels : iterable
        Containing ground truth labels
    consensus_delta : float
        Decision rule parameter.

    Return
    ------
    pruned_df:
        dataframe with `inconsistent samples` removed.
    pruned_lab:
        pruned labels
    """

    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in (range(len(df))):
        if full_dups[i] and (i not in all_seen):
            i_dups = finder(df, df.iloc[i])
            groups.append(i_dups.index)
            all_seen.update(i_dups.index)

    pruned_df = []
    pruned_lab = []
    for group in groups:
        scores = np.array([labels[i] for i in group])
        consensus = round(scores.mean())
        for i in group:
            if (abs(scores.mean() - 0.5) < consensus_delta) or labels[i] == consensus:
                # First condition: consensus is close to 50/50, can't consider this "outliers", so keep them all
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)
