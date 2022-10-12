import torch
import warnings
from torch import nn
from kme.models.base import FeatureNetBase, init_weights, create_sinusoidal_positional_embeddings
from torchvision.models import wide_resnet50_2
from kme.models.image_utils import image2d_patcher
from collections.abc import Iterable
from torch.nn import Unfold
from torch.nn import functional as F


class CIFARCNNPatcher(nn.Module):
    """
    Separate a 32x32 CIFAR image in (overlapping) patches and assign them a positional integer
    """
    def __init__(self, patch_size, stride, n_filters, padding=0, activation: str='ReLU', device='cpu', batch_norm=False):
        super(CIFARCNNPatcher, self).__init__()

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
        self._n_patches_1d = (32 - patch_size + 2*padding) // stride + 1
        self._n_patches = self._n_patches_1d*self._n_patches_1d
        self._n_filters = n_filters

    def forward(self, imgs):
        # imgs: B x 3 x 32 x 32
        patches = self._patcher(imgs)
        # patches: B x n_filters x n_patches_1d x n_patches_1d
        patches = torch.transpose(patches, 1, 3)
        # patches: (B*n_patches) x n_filters
        patches = torch.reshape(patches, (-1, self._n_filters))
        idx = torch.arange(0, self._n_patches).repeat(imgs.shape[0]).to(self._device)
        return patches, idx


class CIFAR3CNNPatcherFixed(nn.Module):
    """
    Separate a 32x32 CIFAR image in (overlapping) patches and assign them a positional integer
    """
    def __init__(self, n_filters, padding=0, activation: str='ReLU', device='cpu'):
        super(CIFAR3CNNPatcherFixed, self).__init__()

        self._patcher = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_filters, kernel_size=3,
                      stride=1, padding=padding),
            getattr(nn, activation)(),
            nn.BatchNorm2d(num_features=n_filters),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2,
                      stride=1),
            getattr(nn, activation)(),
            nn.BatchNorm2d(num_features=n_filters),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=2,
                      stride=1),
            getattr(nn, activation)(),
        )
        self._device = device
        self._n_patches_1d = (29 + 2*padding) + 1
        self._n_patches_1d = (self._n_patches_1d - 2) + 1 # max pooling
        self._n_patches_1d = (self._n_patches_1d - 2) + 1 # 2nd conv
        self._n_patches_1d = (self._n_patches_1d - 2) + 1 # max pooling
        self._n_patches_1d = (self._n_patches_1d - 2) + 1 # 3rdd conv
        self._n_patches = self._n_patches_1d*self._n_patches_1d
        self._n_filters = n_filters

    def forward(self, imgs):
        # imgs: B x 3 x 32 x 32
        patches = self._patcher(imgs)
        # patches: B x n_filters x n_patches_1d x n_patches_1d
        patches = torch.transpose(patches, 1, 3)
        # patches: (B*n_patches) x n_filters
        patches = torch.reshape(patches, (-1, self._n_filters))
        idx = torch.arange(0, self._n_patches).repeat(imgs.shape[0]).to(self._device)
        return patches, idx


class CIFARFeatureNet(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, patcher_kwargs, pos_emb_dim, latent_dim, device='cpu'):
        super(CIFARFeatureNet, self).__init__()

        self._device = device
        self._patch_extractor = CIFARCNNPatcher(**patcher_kwargs, device=device)
        self._patch_encoder = nn.Sequential(
            nn.Linear(self._patch_extractor._n_filters, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, pos_emb_dim),
        )
        self._patch_encoder.apply(init_weights)
        self._pos_embedding = nn.Embedding(self._patch_extractor._n_patches, embedding_dim=pos_emb_dim)
        torch.nn.init.uniform_(self._pos_embedding.weight, -1.0, 1.0)
        self._feature_encoder = nn.Sequential(
            nn.Linear(2*pos_emb_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim),
        )
        self._feature_encoder.apply(init_weights)
        self._latent_dim = latent_dim

    def process_samples(self, imgs):
        patches, idx = self._patch_extractor(imgs)
        patches = self._patch_encoder(patches)
        pos_emb = self._pos_embedding(idx)

        # patches = (B*n_patches) x pos_emb_dim
        # pos_emb = (B*n_patches) x pos_emb_dim
        features = torch.cat([patches, pos_emb], dim=1)
        features = self._feature_encoder(features)
        return torch.reshape(features, (-1, self._patch_extractor._n_patches, self._latent_dim))


class CIFARPositionGating(FeatureNetBase):
    """
    Using the position embedding to gate
    """
    def __init__(self, patcher, patcher_kwargs, latent_dim, device='cpu'):
        super(CIFARPositionGating, self).__init__()

        self._device = device
        self._patch_extractor = globals()[patcher](**patcher_kwargs, device=device)
        self._pos_embedding = nn.Embedding(self._patch_extractor._n_patches, embedding_dim=latent_dim)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._patch_extractor._n_filters, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self._feature_encoder.apply(init_weights)
        self._latent_dim = latent_dim
        self._gate = nn.Sigmoid()

    def process_samples(self, imgs):
        patches, idx = self._patch_extractor(imgs)
        pos_emb = self._gate(self._pos_embedding(idx))

        # patches = (B*n_patches) x pos_emb_dim
        # pos_emb = (B*n_patches) x pos_emb_dim
        features = self._feature_encoder(patches)
        features = features*pos_emb
        return torch.reshape(features, (-1, self._patch_extractor._n_patches, self._latent_dim))


class CIFARPositionGating3x(FeatureNetBase):
    """
    Using the position embedding to gate
    """
    def __init__(self, patcher, patcher_kwargs, latent_dim, device='cpu'):
        super(CIFARPositionGating3x, self).__init__()

        self._device = device
        self._patch_extractor = globals()[patcher](**patcher_kwargs, device=device)
        self._pos_embedding_1 = nn.Embedding(self._patch_extractor._n_patches, embedding_dim=256)
        self._pos_embedding_2 = nn.Embedding(self._patch_extractor._n_patches, embedding_dim=256)
        self._pos_embedding_3 = nn.Embedding(self._patch_extractor._n_patches, embedding_dim=latent_dim)
        self._feature_encoder_1 = nn.Sequential(
            nn.Linear(self._patch_extractor._n_filters, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self._feature_encoder_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self._feature_encoder_3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self._feature_encoder_1.apply(init_weights)
        self._feature_encoder_2.apply(init_weights)
        self._feature_encoder_3.apply(init_weights)
        self._latent_dim = latent_dim
        self._gate = nn.Sigmoid()

    def process_samples(self, imgs):
        patches, idx = self._patch_extractor(imgs)
        pos_emb_1 = self._gate(self._pos_embedding_1(idx))
        pos_emb_2 = self._gate(self._pos_embedding_2(idx))
        pos_emb_3 = self._gate(self._pos_embedding_3(idx))

        # patches = (B*n_patches) x pos_emb_dim
        # pos_emb = (B*n_patches) x pos_emb_dim
        features = self._feature_encoder_1(patches)
        features = features*pos_emb_1
        features = self._feature_encoder_2(features)
        features = features*pos_emb_2
        features = self._feature_encoder_3(features)
        features = features*pos_emb_3
        return torch.reshape(features, (-1, self._patch_extractor._n_patches, self._latent_dim))


class CIFARNoPosition(FeatureNetBase):
    """
    Using the position embedding to gate
    """
    def __init__(self, patcher, patcher_kwargs, latent_dim, device='cpu'):
        super(CIFARNoPosition, self).__init__()

        self._device = device
        self._patch_extractor = globals()[patcher](**patcher_kwargs, device=device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._patch_extractor._n_filters, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self._feature_encoder.apply(init_weights)
        self._latent_dim = latent_dim
        self._gate = nn.Sigmoid()

    def process_samples(self, imgs):
        patches, idx = self._patch_extractor(imgs)

        # patches = (B*n_patches) x pos_emb_dim
        # pos_emb = (B*n_patches) x pos_emb_dim
        features = self._feature_encoder(patches)
        return torch.reshape(features, (-1, self._patch_extractor._n_patches, self._latent_dim))


class CIFARSimplerFeatureNet1(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, patcher_kwargs, pos_emb_dim, latent_dim, device='cpu'):
        super(CIFARSimplerFeatureNet1, self).__init__()

        self._device = device
        self._patch_extractor = CIFARCNNPatcher(**patcher_kwargs, device=device)
        self._pos_embedding = nn.Embedding(self._patch_extractor._n_patches, embedding_dim=pos_emb_dim)
        torch.nn.init.uniform_(self._pos_embedding.weight, -1.0, 1.0)
        self._feature_encoder = nn.Sequential(
            nn.Linear(pos_emb_dim + self._patch_extractor._n_filters, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
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


class CIFARAlternativePatching1(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, n_filters, patch_size, padding=0, device='cpu'):
        super(CIFARAlternativePatching1, self).__init__()

        self._device = device
        if not isinstance(patch_size, Iterable):
            self._kernel_sz = (patch_size, patch_size)
        else:
            self._kernel_sz = patch_size
        self._n_patches = ( int((2*padding + 32)/self._kernel_sz[0]) * int((2*padding + 32)/self._kernel_sz[1]) )
        filter_size = self._kernel_sz[0]*self._kernel_sz[1]*3
        self._net_list = nn.ModuleList()
        self._padding = padding
        for _ in range(self._n_patches):
            self._net_list.append(nn.Sequential(
                nn.Linear(filter_size, n_filters),
                nn.ReLU(),
                nn.Linear(n_filters, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, latent_dim),
            ))
            self._net_list[-1].apply(init_weights)
        self._latent_dim = latent_dim

    def process_samples(self, imgs):
        # patching
        # imgs: B x C x H x W
        patches = image2d_patcher(imgs, kernel_sz=self._kernel_sz, padding=self._padding)
        # patches: B x (CxK_SZ[0]xK_SZ[1]) x self._n_patches
        patches = patches.transpose(0, 2).transpose(1, 2)
        # patches: self._n_patches x B x (CxK_SZ[0]xK_SZ[1])
        patch_reprs = []
        for i in range(self._n_patches):
            patch_reprs.append(self._net_list[i](patches[i:i+1]))
        patch_reprs = torch.cat(patch_reprs, dim=0).transpose(0, 1)

        return patch_reprs


class CIFARAlternativePatching2(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, n_filters, patch_size, padding=0, device='cpu'):
        super(CIFARAlternativePatching2, self).__init__()

        self._device = device
        self._n_filters = n_filters
        if not isinstance(patch_size, Iterable):
            self._kernel_sz = (patch_size, patch_size)
        else:
            self._kernel_sz = patch_size
        self._n_patches = ( int((2*padding + 32)/self._kernel_sz[0]) * int((2*padding + 32)/self._kernel_sz[1]) )
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv3d(self._n_patches, self._n_filters*self._n_patches, (1, 1, 1), groups=self._n_patches),
            nn.ReLU(),
            nn.LayerNorm([3, self._kernel_sz[0], self._kernel_sz[1]]),
            nn.Conv3d(self._n_filters*self._n_patches, self._n_patches*256, (1, 1, 1), groups=self._n_patches),
            nn.ReLU(),
            nn.Conv3d(256*self._n_patches, self._n_patches*128, (1, 1, 1), groups=self._n_patches),
            nn.ReLU(),
            nn.LayerNorm([3, self._kernel_sz[0], self._kernel_sz[1]]),
            nn.Conv3d(128*self._n_patches, self._n_patches*32, (1, 1, 1), groups=self._n_patches),
            nn.ReLU(),
            nn.Conv3d(32*self._n_patches, self._n_patches, (1, 1, 1), groups=self._n_patches),
            nn.ReLU(),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(3*self._kernel_sz[0]*self._kernel_sz[1], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._kernel_sz, stride=self._kernel_sz, padding=padding)

    def process_samples(self, imgs):
        # patching
        # imgs: B x C x H x W
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, self._n_patches,
                                                             3, self._kernel_sz[0], self._kernel_sz[1])
        # patches: B x n_patches x 3 x K_h x K_w
        patches = self._feature_cnn_encoder(patches)
        # patches: B x n_patches x 3 x K_h x K_w
        patches = patches.reshape(-1, self._n_patches, 3*self._kernel_sz[0]*self._kernel_sz[1])
        # patches: B x n_patches x (3xK_hxK_w)
        patches = self._feat_mlp(patches)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching3(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, n_filters, patch_size, padding=0, device='cpu'):
        super(CIFARAlternativePatching3, self).__init__()

        self._device = device
        self._n_filters = n_filters
        if not isinstance(patch_size, Iterable):
            self._kernel_sz = (patch_size, patch_size)
        else:
            self._kernel_sz = patch_size
        self._n_patches = ( int((2*padding + 32)/self._kernel_sz[0]) * int((2*padding + 32)/self._kernel_sz[1]) )
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv3d(self._n_patches, self._n_filters*self._n_patches, (1, 1, 1), groups=self._n_patches),
            nn.LeakyReLU(),
            nn.LayerNorm([3, self._kernel_sz[0], self._kernel_sz[1]]),
            nn.Conv3d(self._n_filters*self._n_patches, self._n_patches*256, (1, 1, 1), groups=self._n_patches),
            nn.LeakyReLU(),
            nn.Conv3d(256*self._n_patches, self._n_patches*128, (1, 1, 1), groups=self._n_patches),
            nn.LeakyReLU(),
            nn.LayerNorm([3, self._kernel_sz[0], self._kernel_sz[1]]),
            nn.Conv3d(128*self._n_patches, self._n_patches*32, (1, 1, 1), groups=self._n_patches),
            nn.LeakyReLU(),
            nn.Conv3d(32*self._n_patches, self._n_patches, (1, 1, 1), groups=self._n_patches),
            nn.ReLU(),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(3*self._kernel_sz[0]*self._kernel_sz[1], 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim),
        )
        self._projection = nn.Linear(3*self._kernel_sz[0]*self._kernel_sz[1], latent_dim, bias=False)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._kernel_sz, stride=self._kernel_sz, padding=padding)

    def process_samples(self, imgs):
        # patching
        # imgs: B x C x H x W
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, self._n_patches,
                                                             3, self._kernel_sz[0], self._kernel_sz[1])
        orig_patches = patches
        # patches: B x n_patches x 3 x K_h x K_w
        patches = self._feature_cnn_encoder(patches)
        # patches: B x n_patches x 3 x K_h x K_w
        patches = patches.reshape(-1, self._n_patches, 3*self._kernel_sz[0]*self._kernel_sz[1])
        # patches: B x n_patches x (3xK_hxK_w)
        patches = self._feat_mlp(patches) + self._projection(orig_patches.reshape(
            -1, self._n_patches, 3*self._kernel_sz[0]*self._kernel_sz[1]
        ))
        # patches: B x n_patches x latent_dim

        return patches


class CIFARSimplerFeatureNet2_3Patcher(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, patcher_kwargs, pos_emb_dim, latent_dim, device='cpu'):
        super(CIFARSimplerFeatureNet2_3Patcher, self).__init__()

        self._device = device
        self._patch_extractor = CIFAR3CNNPatcherFixed(**patcher_kwargs, device=device)
        self._pos_embedding = nn.Embedding(self._patch_extractor._n_patches, embedding_dim=pos_emb_dim)
        torch.nn.init.uniform_(self._pos_embedding.weight, -1.0, 1.0)
        self._feature_encoder = nn.Sequential(
            nn.Linear(pos_emb_dim + self._patch_extractor._n_filters, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
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


class CIFARFeatureNoBNNet(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, patcher_kwargs, pos_emb_dim, latent_dim, device='cpu'):
        super(CIFARFeatureNoBNNet, self).__init__()

        self._device = device
        self._patch_extractor = CIFARCNNPatcher(**patcher_kwargs, device=device)
        self._patch_encoder = nn.Sequential(
            nn.Linear(self._patch_extractor._n_filters, 128),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, pos_emb_dim),
        )
        self._patch_encoder.apply(init_weights)
        self._pos_embedding = nn.Embedding(self._patch_extractor._n_patches, embedding_dim=pos_emb_dim)
        torch.nn.init.uniform_(self._pos_embedding.weight, -1.0, 1.0)
        self._feature_encoder = nn.Sequential(
            nn.Linear(2*pos_emb_dim, 128),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),
        )
        self._feature_encoder.apply(init_weights)
        self._latent_dim = latent_dim

    def process_samples(self, imgs):
        patches, idx = self._patch_extractor(imgs)
        patches = self._patch_encoder(patches)
        pos_emb = self._pos_embedding(idx)

        # patches = (B*n_patches) x pos_emb_dim
        # pos_emb = (B*n_patches) x pos_emb_dim
        features = torch.cat([patches, pos_emb], dim=1)
        features = self._feature_encoder(features)
        return torch.reshape(features, (-1, self._patch_extractor._n_patches, self._latent_dim))


class CIFARLatentClassifier(nn.Module):
    """
    Classifier based on a latent representation of a CIFAR image
    """
    def __init__(self, latent_dim, n_classes):
        super(CIFARLatentClassifier, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, n_classes)
        )
        self._classifier.apply(init_weights)

    def forward(self, x):
        return self._classifier(x)


class CIFARSmallLeakyLatentClassifier(nn.Module):
    """
    Classifier based on a latent representation of a CIFAR image
    """
    def __init__(self, latent_dim, n_classes):
        super(CIFARSmallLeakyLatentClassifier, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes)
        )
        self._classifier.apply(init_weights)

    def forward(self, x):
        return self._classifier(x)


class CIFARLeakyLatentClassifier(nn.Module):
    """
    Classifier based on a latent representation of a CIFAR image
    """

    def __init__(self, latent_dim, n_classes):
        super(CIFARLeakyLatentClassifier, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes)
        )
        self._classifier.apply(init_weights)

    def forward(self, x):
        return self._classifier(x)


class CIFARLatentLinearClassifier(nn.Module):
    """
    Classifier based on a latent representation of a CIFAR image
    """
    def __init__(self, latent_dim, n_classes):
        super(CIFARLatentLinearClassifier, self).__init__()
        self._classifier = nn.Linear(latent_dim, n_classes)
        init_weights(self._classifier)

    def forward(self, x):
        return self._classifier(x)


class CIFARBigLatentClassifier(nn.Module):
    """
    Classifier based on a latent representation of a CIFAR image
    """
    def __init__(self, latent_dim, n_classes):
        super(CIFARBigLatentClassifier, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_classes),
        )
        init_weights(self._classifier)

    def forward(self, x):
        return self._classifier(x)


class CIFARHomo3LatentClassifier(nn.Module):
    """
    Classifier based on a latent representation of a CIFAR image
    """
    def __init__(self, latent_dim, n_classes):
        super(CIFARHomo3LatentClassifier, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, n_classes),
        )
        init_weights(self._classifier)

    def forward(self, x):
        return self._classifier(torch.pow(x, 3))


class CIFARAlternativePatching4(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching4, self).__init__()

        self._device = device
        self._n_filters = 16
        self._kernel_sz = 1
        self._n_patches_1d = 32 // self._kernel_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=self._kernel_sz, padding=(0,0), stride=self._kernel_sz),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters, self._n_filters*16, kernel_size=(1, 1), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.LayerNorm([32, 32]),
            nn.Conv2d(self._n_filters*16, self._n_filters*64, kernel_size=(1, 1), padding=(0,0), stride=(1, 1)),
        )
        self._pos_dim = latent_dim
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters*64, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        #self._projection = nn.Linear(self._n_filters*64, latent_dim, bias=False)
        self._latent_dim = latent_dim

    def process_samples(self, imgs):
        # patching
        # imgs: B x C x H x W
        patches_encoding = self._feature_cnn_encoder(imgs).reshape(
            -1, self._n_filters*64, self._n_patches).transpose(1, 2)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)*self._pos_embedding
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching5(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching5, self).__init__()

        self._device = device
        self._n_filters = 16
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters, self._n_filters*16, kernel_size=(2, 2), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(self._n_filters*16, self._n_filters*64, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
        )
        self._pos_dim = latent_dim
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._n_filters*64*7*7, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        #self._projection = nn.Linear(self._n_filters*64, latent_dim, bias=False)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 3 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._n_filters*64*7*7)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)*self._pos_embedding
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching6(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching6, self).__init__()

        self._device = device
        self._n_filters = 2048
        self._patch_sz = 4
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(2, 2), padding=(1,1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
        )
        self._pos_dim = 1024
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 2, self._pos_dim),
            nn.LeakyReLU(),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim, latent_dim),
            nn.LeakyReLU(),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2))*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching7(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching7, self).__init__()

        self._device = device
        self._n_filters = 256
        self._patch_sz = 4
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(2, 2), padding=(1,1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
        )
        self._pos_dim = 1024
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 2, self._pos_dim),
            nn.LeakyReLU(),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim, latent_dim),
            nn.LeakyReLU(),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2))*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching8(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching8, self).__init__()

        self._device = device
        self._n_filters = 128
        self._patch_sz = 4
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout(p=0.2),
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(2, 2), padding=(1,1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout(p=0.2),
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            nn.Dropout(p=0.2),
        )
        self._pos_dim = 256
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 2, self._pos_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2))*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching9(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching9, self).__init__()

        self._device = device
        self._n_filters = 512
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(7, 7), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(4, 4), padding=(2,2), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))
        )
        self._pos_dim = 1024
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 2, self._pos_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim, latent_dim),
            nn.LeakyReLU()
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2))*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching10(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching10, self).__init__()

        self._device = device
        self._n_filters = 512
        self._patch_sz = 4
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(2, 2), padding=(1,1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1)),
        )
        self._pos_dim = 256
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 2, self._pos_dim),
            nn.LeakyReLU(),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim, latent_dim),
            nn.LeakyReLU(),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(
            patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2))*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching11(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching11, self).__init__()

        self._device = device
        self._n_filters = 512
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(2, 2), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1)),
        )
        self._pos_dim = latent_dim
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim, latent_dim),
            nn.LeakyReLU(),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding) + (self._pos_embedding.to(imgs.device))
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching12(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching12, self).__init__()

        self._device = device
        self._n_filters = 2048
        self._patch_sz = 4
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters, self._n_filters // 8, kernel_size=(2, 2), padding=(1,1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Conv2d(self._n_filters // 8, self._n_filters // 16, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1)),
        )
        self._pos_dim = latent_dim // 2
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 16, self._pos_dim),
            nn.LeakyReLU(),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(
            self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(patches_encoding.reshape(-1, self._n_patches, self._n_filters // 16))
        patches = F.leaky_relu(torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2))
        # patches_encoding: B x n_patches x 4*n_filters
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching13(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching13, self).__init__()

        self._device = device
        self._n_filters = 128
        self._patch_sz = 4
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout(p=0.2),
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(2, 2), padding=(1,1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout(p=0.2),
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            nn.Dropout(p=0.2),
        )
        self._pos_dim = 256
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 2, self._pos_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim*2, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(
            self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2))
        patches_encoding = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching14(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching14, self).__init__()

        self._device = device
        self._n_filters = 256
        self._patch_sz = 4
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(3, 3), padding=(2,2), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(2, 2), padding=(1,1), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(3, 3), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            nn.Dropout2d(p=0.1),
        )
        self._pos_dim = 1024
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 2, self._pos_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim*2, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(
            self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2))
        patches_encoding = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARAlternativePatching15(FeatureNetBase):
    """
    Network to process CIFAR images as a bag of features with positional embeddings
    """
    def __init__(self, latent_dim, device='cpu'):
        super(CIFARAlternativePatching15, self).__init__()

        self._device = device
        self._n_filters = 256
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        self._feature_cnn_encoder = nn.Sequential(
            nn.Conv2d(3, self._n_filters, kernel_size=(4, 4), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(), # nf x 5 x 5
            nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout2d(p=0.1), # nf x 4 x 4
            nn.Conv2d(self._n_filters, self._n_filters // 4, kernel_size=(2, 2), padding=(0,0), stride=(1, 1), padding_mode='replicate'),
            nn.LeakyReLU(), # nf//4 x 3 x 3
            nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1)),
            nn.Dropout2d(p=0.1), # nf//4 x 2 x 2
            nn.Conv2d(self._n_filters // 4, self._n_filters // 2, kernel_size=(2, 2), padding=(0,0), stride=(1, 1)),
            nn.LeakyReLU(), # nf//2 x 1 x 1
        )
        self._pos_dim = 1024
        self._feat_processor = nn.Sequential(
            nn.Linear(self._n_filters // 2, self._pos_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim*2, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(
            self._n_patches, self._pos_dim)
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B*n_patches x 1 x H x W
        patches_encoding = self._feature_cnn_encoder(patches)
        # patches_encoding: B*n_patches x self._n_filters*64 x 7 x 7
        patches_encoding = self._feat_processor(patches_encoding.reshape(-1, self._n_patches, self._n_filters // 2))
        patches_encoding = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        # patches_encoding: B x n_patches x 4*n_filters
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARPreTrainedWideResNet50_2(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(CIFARPreTrainedWideResNet50_2, self).__init__()

        self._device = device
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = 1024
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._pos_dim)*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARPreTrainedWideResNet50_2_2(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(CIFARPreTrainedWideResNet50_2_2, self).__init__()

        self._device = device
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = 1024
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self._pos_dim, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._pos_dim)*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARPreTrainedWideResNet50_2_3(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(CIFARPreTrainedWideResNet50_2_3, self).__init__()

        self._device = device
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1024, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, 1024)
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding) + (self._pos_embedding.to(imgs.device))
        # patches: B x n_patches x latent_dim

        return patches


class CIFARPreTrainedWideResNet50_2_4(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(CIFARPreTrainedWideResNet50_2_4, self).__init__()

        self._device = device
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1024, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, 1024)
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding) + (self._pos_embedding.to(imgs.device))
        # patches: B x n_patches x latent_dim

        return patches


class CIFARPreTrainedWideResNet50_2_5(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(CIFARPreTrainedWideResNet50_2_5, self).__init__()
        warnings.warn('Warning: for this model we expect the latent dim to be multiple of 4!')

        self._device = device
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim // 2
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Linear(1024, latent_dim // 2),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = self._feat_mlp(patches_encoding.reshape(-1, self._n_patches, 1024))
        # patches_encoding: B x n_patches x 4096
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding+self._pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARPreTrainedWideResNet50_2_6(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(CIFARPreTrainedWideResNet50_2_6, self).__init__()
        warnings.warn('Warning: for this model we expect the latent dim to be multiple of 4!')

        self._device = device
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim // 2
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(
            self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Linear(1024, latent_dim // 2),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = self._feat_mlp(patches_encoding.reshape(-1, self._n_patches, 1024))
        # patches_encoding: B x n_patches x 4096
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)
        # patches: B x n_patches x latent_dim

        return patches


class CIFARPreTrainedWideResNet50_2_7(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(CIFARPreTrainedWideResNet50_2_7, self).__init__()

        self._device = device
        self._patch_sz = 8
        self._n_patches_1d = 32 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = 1024
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self._pos_dim, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._pos_dim)*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches
