
import torch
from torch import nn
from kme.models.base import FeatureNetBase
from kme.models.base import create_sinusoidal_positional_embeddings


class SC_FeatureNet1(FeatureNetBase):

    def __init__(self, latent_dim, device):
        super(SC_FeatureNet1, self).__init__()
        self._latent_dim = latent_dim
        self._device = device

        self.feature_encoder = nn.Sequential(
            nn.Linear(1, self._latent_dim),
            nn.LeakyReLU()).double()

        self.pos_embed = create_sinusoidal_positional_embeddings(
            117, self._latent_dim).to(device)

    def process_samples(self, x):

        x = x.unsqueeze(-1)
        features = self.feature_encoder(
            x.double()).float()*(self.pos_embed.to(x.device))
        return features


class SC_FeatureNet2(FeatureNetBase):

    def __init__(self, latent_dim, device):
        super(SC_FeatureNet2, self).__init__()
        self._latent_dim = latent_dim
        self._device = device

        self._feature_encoders = nn.ModuleList()
        for i in range(117):
            self._feature_encoders.append(nn.Sequential(
                nn.Linear(1, self._latent_dim),
                nn.LeakyReLU()
            )).double()

    def process_samples(self, x):

        x = x.unsqueeze(-1)
        features = []
        for i in range(x.shape[1]):
            features.append(self._feature_encoders[i](
                x[:, i, :].double()).float().unsqueeze(1))

        return torch.cat(features, dim=1)
