import torch
from torch import nn
from kme.models.base import FeatureNetBase


class RiskFeatNet1(FeatureNetBase):
    """
    Simple network for risk score datasets
    """
    def __init__(self, n_feats, latent_dim, device='cpu'):
        super(RiskFeatNet1, self).__init__()
        self._hidden_sz = 64
        self._device = device
        self._n_feats = n_feats
        self._latent_dim = latent_dim
        self._feat_encoder = nn.Sequential(
            nn.Conv1d(in_channels=self._n_feats, out_channels=self._hidden_sz*self._n_feats, kernel_size=1,
                      groups=self._n_feats),
            nn.LeakyReLU(),
            nn.LayerNorm([self._hidden_sz*self._n_feats, 1]),
            nn.Conv1d(in_channels=self._hidden_sz*self._n_feats,
                      out_channels=self._hidden_sz*self._n_feats, kernel_size=1,
                      groups=self._n_feats),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=self._hidden_sz*self._n_feats,
                      out_channels=self._hidden_sz*self._n_feats, kernel_size=1,
                      groups=self._n_feats),
            nn.LeakyReLU(),
            nn.LayerNorm([self._hidden_sz*self._n_feats, 1]),
            nn.Conv1d(in_channels=self._hidden_sz*self._n_feats,
                      out_channels=self._latent_dim*self._n_feats, kernel_size=1,
                      groups=self._n_feats),
            nn.LeakyReLU(),
        )

    def process_samples(self, x):
        x = torch.unsqueeze(x, 2)
        x = self._feat_encoder(x)
        x = torch.reshape(x, (-1, self._n_feats, self._latent_dim))
        return x


class SmallTabFeatNet1(FeatureNetBase):
    """
    Simple network for risk score datasets
    """
    def __init__(self, n_feats, latent_dim, device='cpu'):
        super(SmallTabFeatNet1, self).__init__()
        self._hidden_sz = 5
        self._device = device
        self._n_feats = n_feats
        self._latent_dim = latent_dim
        self._feat_encoder = nn.Sequential(
            nn.Conv1d(in_channels=self._n_feats, out_channels=self._hidden_sz*self._n_feats, kernel_size=1,
                      groups=self._n_feats),
            nn.LeakyReLU(),
            nn.LayerNorm([self._hidden_sz*self._n_feats, 1]),
            nn.Conv1d(in_channels=self._hidden_sz*self._n_feats,
                      out_channels=self._latent_dim*self._n_feats, kernel_size=1,
                      groups=self._n_feats),
            nn.LeakyReLU(),
        )

    def process_samples(self, x):
        x = torch.unsqueeze(x, 2)
        x = self._feat_encoder(x)
        x = torch.reshape(x, (-1, self._n_feats, self._latent_dim))
        return x


class SmallTabFeatNet2(FeatureNetBase):
    """
    Simple network for risk score datasets
    """
    def __init__(self, n_feats, latent_dim, device='cpu'):
        super(SmallTabFeatNet2, self).__init__()
        self._hidden_sz = 128
        self._device = device
        self._n_feats = n_feats
        self._latent_dim = latent_dim
        self._feat_encoder = nn.Sequential(
            nn.Conv1d(in_channels=self._n_feats, out_channels=self._hidden_sz*self._n_feats, kernel_size=1,
                      groups=self._n_feats),
            nn.LeakyReLU(),
            nn.LayerNorm([self._hidden_sz*self._n_feats, 1]),
            nn.Conv1d(in_channels=self._hidden_sz*self._n_feats,
                      out_channels=self._latent_dim*self._n_feats, kernel_size=1,
                      groups=self._n_feats),
            nn.LeakyReLU(),
        )

    def process_samples(self, x):
        x = torch.unsqueeze(x, 2)
        x = self._feat_encoder(x)
        x = torch.reshape(x, (-1, self._n_feats, self._latent_dim))
        return x


class BaselineMLP1(FeatureNetBase):
    """
    Implementation of a classic MLP as a feat net, for benchmarking purposes
    """
    def __init__(self, n_feats, latent_dim, device='cpu'):
        super(BaselineMLP1, self).__init__()
        self._hidden_sz = 5
        self._device = device
        self._n_feats = n_feats
        self._latent_dim = latent_dim
        self._feat_encoder = nn.Sequential(
            nn.Linear(n_feats, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),
            nn.LeakyReLU(),
        )

    def process_samples(self, x):
        x = self._feat_encoder(x).reshape(-1, 1, self._latent_dim)
        return x
