import torch
from torch import nn
import math
import warnings
from torch.nn import functional as F
from kme.models.base import FeatureNetBase, init_weights, create_sinusoidal_positional_embeddings
from collections.abc import Iterable
from torch.nn import Unfold


class MNISTPatcher(nn.Module):
    """
    Separate a 32x32 CIFAR image in (overlapping) patches and assign them a positional integer
    """

    def __init__(self, patch_size, stride, n_filters, padding=0, activation: str = 'ReLU', device='cpu', batch_norm=False):
        super(MNISTPatcher, self).__init__()

        if batch_norm:
            self._patcher = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=n_filters, kernel_size=patch_size,
                          stride=stride, padding=padding),
                getattr(nn, activation)(),
                nn.BatchNorm2d(num_features=n_filters)
            )
        else:
            self._patcher = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=n_filters, kernel_size=patch_size,
                          stride=stride, padding=padding),
                getattr(nn, activation)()
            )
        self._device = device
        torch.nn.init.xavier_uniform_(self._patcher[0].weight)
        self._n_patches_1d = (28 - patch_size + 2*padding) // stride + 1
        self._n_patches = self._n_patches_1d*self._n_patches_1d
        self._n_filters = n_filters

    def forward(self, imgs):
        # imgs: B x 1 x 32 x 32
        patches = self._patcher(imgs)
        # patches: B x n_filters x n_patches_1d x n_patches_1d
        patches = torch.transpose(patches, 1, 3)
        # patches: (B*n_patches) x n_filters
        patches = torch.reshape(patches, (-1, self._n_filters))
        idx = torch.arange(0, self._n_patches).repeat(
            imgs.shape[0]).to(self._device)
        return patches, idx


class MNISTSimpleFeatNet(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, patcher_kwargs, pos_emb_dim, latent_dim, device='cpu'):
        super(MNISTSimpleFeatNet, self).__init__()

        self._device = device
        self._patch_extractor = MNISTPatcher(**patcher_kwargs, device=device)
        self._pos_embedding = nn.Embedding(
            self._patch_extractor._n_patches, embedding_dim=pos_emb_dim)
        torch.nn.init.uniform_(self._pos_embedding.weight, -1.0, 1.0)
        self._feature_encoder = nn.Sequential(
            nn.Linear(pos_emb_dim + self._patch_extractor._n_filters, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim),
        )
        self._feature_encoder.apply(init_weights)
        self._latent_dim = latent_dim

    def process_samples(self, imgs):
        patches, idx = self._patch_extractor(imgs)
        pos_emb = self._pos_embedding(idx)

        # patches = (B*n_patches) x pos_emb_dim
        # pos_emb = (B*n_patches) x pos_emb_dim
        features = torch.cat([patches, pos_emb], dim=1)
        features = self._feature_encoder(features)
        return torch.reshape(features, (-1, self._patch_extractor._n_patches, self._latent_dim))


class MNISTAlternativePatching2(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, n_filters, patch_size, padding=0, device='cpu'):
        super(MNISTAlternativePatching2, self).__init__()

        self._device = device
        self._n_filters = n_filters
        if not isinstance(patch_size, Iterable):
            self._kernel_sz = (patch_size, patch_size)
        else:
            self._kernel_sz = patch_size
        self._n_patches = (int(
            (2*padding + 28)/self._kernel_sz[0]) * int((2*padding + 28)/self._kernel_sz[1]))
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv3d(self._n_patches, self._n_filters *
                      self._n_patches, (1, 1, 1), groups=self._n_patches),
            nn.LeakyReLU(),
            nn.Conv3d(self._n_filters*self._n_patches,
                      self._n_patches*32, (1, 1, 1), groups=self._n_patches),
            nn.LeakyReLU(),
            nn.Conv3d(32*self._n_patches, self._n_patches,
                      (1, 1, 1), groups=self._n_patches),
            nn.LeakyReLU()
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._kernel_sz[0]*self._kernel_sz[1], 128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim, bias=False),
        )
        self._projection = nn.Linear(
            self._kernel_sz[0]*self._kernel_sz[1], latent_dim, bias=False)
        self._latent_dim = latent_dim
        self._unfold = Unfold(
            self._kernel_sz, stride=self._kernel_sz, padding=padding)

    def process_samples(self, imgs):
        # patching
        # imgs: B x C x H x W
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, self._n_patches,
                                                             1, self._kernel_sz[0], self._kernel_sz[1])
        # patches: B x n_patches x 3 x K_h x K_w
        orig_patches = patches
        patches = self._feature_cnn_encoder(patches)
        # patches: B x n_patches x 3 x K_h x K_w
        patches = patches.reshape(-1, self._n_patches,
                                  self._kernel_sz[0]*self._kernel_sz[1])
        # patches: B x n_patches x (3xK_hxK_w)
        patches = self._feat_mlp(patches) + \
            self._projection(orig_patches.reshape(-1, self._n_patches,
                             self._kernel_sz[0]*self._kernel_sz[1]))
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching3(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching3, self).__init__()

        self._device = device
        self._n_filters = 128
        self._n_patches_1d = 7
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(
                4, 4), padding=(0, 0), stride=(4, 4)),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters, self._n_filters*2,
                      kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters*2, self._n_filters*4,
                      kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
        )
        self._pos_dim = latent_dim
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters*4, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self._n_patches),
            nn.Linear(1024, latent_dim, bias=False),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        #self._projection = nn.Linear(self._kernel_sz[0]*self._kernel_sz[1], latent_dim, bias=False)
        self._latent_dim = latent_dim

    def process_samples(self, imgs):
        # patching
        # imgs: B x C x H x W
        patches_encoding = self._feature_cnn_encoder(imgs).reshape(
            -1, self._n_filters*4, self._n_patches).transpose(1, 2)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)*self._pos_embedding
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching5(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching5, self).__init__()

        self._device = device
        self._n_filters = 4
        self._patch_sz = 7
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(3, 3), padding=(
                0, 0), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters, self._n_filters*16,
                      kernel_size=(2, 2), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters*16, self._n_filters*64,
                      kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)),
        )
        self._pos_dim = latent_dim
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters*64*2*2, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, latent_dim),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        #self._projection = nn.Linear(self._n_filters*64, latent_dim, bias=False)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = patches_encoding.reshape(
            -1, self._n_patches, self._n_filters*64*2*2)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)*self._pos_embedding
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching6(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching6, self).__init__()

        self._device = device
        self._n_filters = 32
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(2, 2), padding=(
                0, 0), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters, self._n_filters*8,
                      kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters*8, self._n_filters*4,
                      kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)),
        )
        self._pos_dim = latent_dim
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters*4, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, latent_dim),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        #self._projection = nn.Linear(self._n_filters*64, latent_dim, bias=False)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = patches_encoding.reshape(
            -1, self._n_patches, self._n_filters*4)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)*self._pos_embedding
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching7(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching7, self).__init__()

        self._device = device
        self._n_filters = 8
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(2, 2), padding=(
                1, 1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters*8,
                      kernel_size=(1, 1), padding=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters*8, self._n_filters*4,
                      kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
        )
        self._pos_dim = 32
        self._preprocess_features = nn.Sequential(
            nn.Linear(self._n_filters*4, self._pos_dim),
            nn.ReLU()
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim*5, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim, bias=False),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._pos_embedding = torch.cat([
            self._pos_embedding,
            torch.roll(self._pos_embedding, 1, 0),
            torch.roll(self._pos_embedding, -1, 0),
            torch.roll(self._pos_embedding, 7, 0),
            torch.roll(self._pos_embedding, -7, 0),
        ], dim=1)
        #self._projection = nn.Linear(self._n_filters*64, latent_dim, bias=False)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = patches_encoding.reshape(
            -1, self._n_patches, self._n_filters*4)
        patches_encoding = self._preprocess_features(
            patches_encoding).repeat(1, 1, 5)*self._pos_embedding
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching8(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching8, self).__init__()

        self._device = device
        self._n_filters = 8
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(2, 2), padding=(
                1, 1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters*8, kernel_size=(1, 1),
                      padding=(1, 1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters*8, self._n_filters*4,
                      kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
        )
        self._pos_dim = self._n_filters*4
        self._feat_mlp = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self._pos_dim, latent_dim),
            nn.BatchNorm1d(self._n_patches)
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        #self._projection = nn.Linear(self._n_filters*64, latent_dim, bias=False)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = patches_encoding.reshape(
            -1, self._n_patches, self._n_filters*4)*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching9(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching9, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 8
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(2, 2), padding=(
                1, 1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters*8, kernel_size=(1, 1),
                      padding=(1, 1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters*8, self._n_filters*4,
                      kernel_size=(3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters*4, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters*4))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding+self._pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching10(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching10, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 128
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding+self._pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching11(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching11, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding+self._pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching12(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching12, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching13(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching13, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1))
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._feat_mlp_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        patches = self._feat_mlp_2(patches)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching14(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching14, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )
        self._pos_dim = latent_dim
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        # patches: B x n_patches x latent_dim

        return patches_encoding


class MNISTAlternativePatching15(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching15, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1))
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching16(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching16, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._pos_embedding = torch.cat(
            [self._pos_embedding, self._relative_pos_embedding], dim=1)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = patches_encoding*self._pos_embedding.to(imgs.device)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching17(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching17, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 4
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = F.leaky_relu(torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2))
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching18(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching18, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 6
        self._n_patches_1d = 30 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=1)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching19(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching19, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 6
        self._n_patches_1d = 30 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.2)
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=1)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching20(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching20, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 6
        self._n_patches_1d = 30 // self._patch_sz
        self._n_patches = 3*self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(self._n_patches, self._n_filters*self._n_patches, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1),
                      groups=self._n_patches),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters*self._n_patches, self._n_filters*self._n_patches, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1),
                      groups=self._n_patches),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.2)
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim),
            nn.LeakyReLU()
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=1)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, self._n_patches, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching21(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching21, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 6
        self._n_patches_1d = 30 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.2)
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._adaptive_weights = torch.rand(
            self._n_patches, 1, requires_grad=True, device=device)
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=1)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)*(self._adaptive_weights.to(imgs.device))
        # patches: B x n_patches x latent_dim

        return patches


class MNISTAlternativePatching22(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """

    def __init__(self, latent_dim, device='cpu'):
        super(MNISTAlternativePatching22, self).__init__()
        warnings.warn(
            'Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._n_filters = 512
        self._patch_sz = 7
        self._n_patches_1d = 28 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(1, self._n_filters, kernel_size=(
                4, 4), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters, kernel_size=(
                3, 3), padding=(0, 0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.2)
        )
        self._pos_dim = latent_dim // 2
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters, latent_dim // 2),
            nn.LeakyReLU()
        )
        self._adaptive_weights = torch.rand(
            self._n_patches, 1, requires_grad=True, device=device)
        self._pos_embedding = create_sinusoidal_positional_embeddings(
            self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(
            self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(
            self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(
            0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=1)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(
            1, 2).reshape(-1, 1, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_mlp(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters))
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)*(self._adaptive_weights.to(imgs.device))
        # patches: B x n_patches x latent_dim

        return patches
