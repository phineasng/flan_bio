import torch
from torch import nn
from kme.models.base import create_sinusoidal_positional_embeddings
from kme.models.base import FeatureNetBase
import json


class TE_FeatureNet1(FeatureNetBase):

    def __init__(self, emb_dim, latent_dim, dataset, device='cpu'):
        super(TE_FeatureNet1, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim

        self._interaction_dataset = dataset
        self._languages = [ds.protein_language for ds in dataset.datasets]

        self._embeddings = nn.ModuleList()
        self._feature_encoders = nn.ModuleList()
        self._pos_embeddings = []

        for i, lang in enumerate(self._languages):
            self._embeddings.append(
                nn.Embedding(
                    lang.number_of_tokens, embedding_dim=self._emb_dim, padding_idx=lang.padding_index)
            )
            self._pos_embeddings.append(
                create_sinusoidal_positional_embeddings(dataset.datasets[i].padding_length, self._latent_dim).to(device))
            self._feature_encoders.append(nn.Sequential(
                nn.Linear(self._emb_dim, self._latent_dim),
                nn.LeakyReLU()
            ))

    def process_samples(self, x):
        features = []
        for i in range(len(self._languages)):
            s = x[i]
            s = self._embeddings[i](s.long())
            s = self._feature_encoders[i](
                s)*(self._pos_embeddings[i].to(x[i].device))
            features.append(s)

        return torch.cat(features, dim=1)


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        return torch.squeeze(data)


class TE_FeatureNet_conv(FeatureNetBase):

    def __init__(self, params_filepath, dataset, device='cpu'):
        super(TE_FeatureNet_conv, self).__init__()

        params = {}
        with open(params_filepath) as fp:
            params.update(json.load(fp))

        self._device = device
        self._tcr_emb_dim = params.get('receptor_embedding_size')
        self._epit_emb_dim = params.get('ligand_embedding_size')
        self._kernel_size = params.get('kernel_size')
        self._out_dim = params.get('latent_dim')
        self._dropout = params.get('dropout')
        self._padding_width = params.get('padding_width')

        self._tcr_emb = nn.Embedding(
            32, embedding_dim=self._tcr_emb_dim, padding_idx=0).to(device)

        self._epit_emb = nn.Embedding(
            32, embedding_dim=self._epit_emb_dim, padding_idx=0).to(device)

        self._tcr_convolution = nn.Sequential(nn.Conv2d(1, self._out_dim, [self._tcr_emb_dim, self._kernel_size], stride=self._kernel_size, padding=(0, self._padding_width)),
                                              Squeeze(),
                                              nn.ReLU(),
                                              )

        self._epit_convolution = nn.Sequential(nn.Conv2d(1, self._out_dim, [self._epit_emb_dim, self._kernel_size], stride=self._kernel_size, padding=(0, self._padding_width)),
                                               Squeeze(),
                                               nn.ReLU(),
                                               )

    def process_samples(self, s):
        tcr_seq = s[1]
        epit_seq = s[0]

        tcr_seq = self._tcr_emb(
            tcr_seq.long()).transpose(2, 1)
        epit_seq = self._epit_emb(
            epit_seq.long()).transpose(2, 1)

        tcr_features = self._tcr_convolution(
            tcr_seq.unsqueeze(1)).transpose(2, 1)

        epit_features = self._epit_convolution(
            epit_seq.unsqueeze(1)).transpose(2, 1)

        return torch.column_stack((tcr_features, epit_features))
