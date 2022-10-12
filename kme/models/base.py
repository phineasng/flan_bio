import math
import torch
from torch import nn
from abc import abstractmethod


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)


class FeatureNetBase(nn.Module):
    """
    Base class for the feature network processing subset of features separately
    """

    def __init__(self):
        super(FeatureNetBase, self).__init__()

        self.aggregate_features = self.sum_features

    def use_mean(self, flag):
        if flag:
            self.aggregate_features = self.mean_features
        else:
            self.aggregate_features = self.sum_features

    def use_global_avg_pool(self, flag):
        if flag:
            self.aggregate_features = self.pool_features
        else:
            self.aggregate_features = self.sum_features

    @abstractmethod
    def process_samples(self, x):
        """
        This function should take care of processing the samples so to:
        - extract the subgroup of features
        - add any structural/contextual parameter
        - apply the feature networks to all the subgroups
        In general it should return a list of length B (batch size)
        Each element i of the list should contain a tensor of size
        Nf_i x F
        where F is the dim of the representation, and Nf_i is the number of feat subsets. Note that Nf_i may differ
        among the samples i, e.g. for missing data.
        If Nf_i = Nf for all i, then this function may return a 3D tensor of shape: B x Nf x F
        """
        pass

    def sum_features(self, processed):
        """
        This function should sum up all the feature representations.
        Note that if the number of feature subsets is not the same for all samples, then this method should be
        overridden.
        """
        return torch.sum(processed, dim=1)

    def mean_features(self, processed):
        """
        This function should sum up all the feature representations.
        Note that if the number of feature subsets is not the same for all samples, then this method should be
        overridden.
        """
        return torch.mean(processed, dim=1)

    def pool_features(self, processed):
        """
        This function should perform global average pooling to the feature representations.
        Note that if the number of feature subsets is not the same for all samples, then this method should be
        overridden.
        """
        m = nn.AvgPool2d((1, processed.shape[1]))
        return m(processed.transpose(2, 1)).squeeze()

    def forward(self, x, return_norm=False):
        processed = self.process_samples(x)
        if return_norm:
            return self.aggregate_features(processed), torch.mean(torch.norm(processed, dim=[1, 2], p='nuc'))
            # nuc norm is a tesor of length B and then you take the mean over the samples
        return self.aggregate_features(processed)


class KMEBase(nn.Module):
    """
    Base Model providing the interface for KME networks
    """

    def __init__(self, feat_net, classifier):
        super(KMEBase, self).__init__()

        if not isinstance(feat_net, FeatureNetBase):
            raise ValueError('Expected a child class of FeatureNetBase!')
        self._feat_net = feat_net
        self._classifier = classifier

    def sample_representation(self, x):
        return self._feat_net.process_samples(x)

    def forward(self, x, return_norm=False):
        x = self._feat_net(x, return_norm)
        if return_norm:
            return self._classifier(x[0]), x[1]
        else:
            return self._classifier(x)


def create_sinusoidal_positional_embeddings(n_positions, emb_dim):
    pos_embedding = torch.zeros(n_positions, emb_dim)
    position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float()
                         * (-math.log(10000.0) / emb_dim))
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding


def create_learnable_positional_embeddings(n_positions, emb_dim):
    pos_embedding = torch.rand(n_positions, emb_dim, requires_grad=True)
    return pos_embedding
