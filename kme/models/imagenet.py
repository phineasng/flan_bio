import torch
from torchvision.models import resnext50_32x4d, wide_resnet50_2, densenet161, resnet18
from torch import nn
from kme.models.base import FeatureNetBase, create_sinusoidal_positional_embeddings, create_learnable_positional_embeddings
from torch.nn import Unfold
import warnings
import torch.nn.functional as F
"""
Feature networks for ImageNet dataset
"""


class ImageNetPretrainedResNext50_32x4d(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPretrainedResNext50_32x4d, self).__init__()

        self._device = device
        self._patch_sz = 28
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = resnext50_32x4d(pretrained=True).to(device)
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

        self._pos_dim = 4096
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
        patches_encoding = self._pretrained_modules(patches).reshape(-1, self._n_patches, self._pos_dim)*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class ImageNetPreTrainedWideResNet50_2(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPreTrainedWideResNet50_2, self).__init__()

        self._device = device
        self._patch_sz = 28
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            #resnet.layer2
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(12544, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        print(patches.shape)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        print(patches_encoding.shape)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, 12544)
        print(patches_encoding.shape)
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding) + (self._pos_embedding.to(imgs.device))
        print(patches.shape)
        # patches: B x n_patches x latent_dim

        return patches


class ImageNetPreTrainedWideResNet50_2_2(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPreTrainedWideResNet50_2_2, self).__init__()

        self._device = device
        self._patch_sz = 28
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(8192, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, 8192)
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding) + (self._pos_embedding.to(imgs.device))
        # patches: B x n_patches x latent_dim

        return patches


class ImageNetPreTrainedWideResNet50_2_3(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPreTrainedWideResNet50_2_3, self).__init__()

        self._device = device
        self._patch_sz = 14
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2048, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, 2048)
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding) + (self._pos_embedding.to(imgs.device))
        # patches: B x n_patches x latent_dim

        return patches


class ImageNetPreTrainedWideResNet50_2_4(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPreTrainedWideResNet50_2_4, self).__init__()

        self._device = device
        self._patch_sz = 32
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = wide_resnet50_2(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Linear(2048, latent_dim, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, 2048)
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding) + (self._pos_embedding.to(imgs.device))
        # patches: B x n_patches x latent_dim

        return patches


class ImageNetPretrainedDenseNet161(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPretrainedDenseNet161, self).__init__()
        warnings.warn('Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._patch_sz = 28
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        densenet = densenet161(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            densenet.features.conv0,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
            densenet.features.denseblock1,
            densenet.features.transition1
        )
        self._out_dim_encoder = 192*3*3

        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim // 2
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._out_dim_encoder, latent_dim // 2, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._out_dim_encoder)
        patches_encoding = self._feat_mlp(patches_encoding)
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)

        return patches


class ImageNetPretrainedResNet18(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPretrainedResNet18, self).__init__()
        warnings.warn('Warning: for this model we expect the latent dim to be even!')

        self._device = device
        self._patch_sz = 32
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = resnet18(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self._out_dim_encoder = 4096

        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = latent_dim // 2
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim).to(device)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d, self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self._out_dim_encoder, latent_dim // 2),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._out_dim_encoder)
        patches_encoding = self._feat_mlp(patches_encoding)
        patches = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device)
        ], dim=2)

        return patches


class ImageNetPretrainedResNext50_32x4d_2(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPretrainedResNext50_32x4d_2, self).__init__()

        self._device = device
        self._patch_sz = 28
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = resnext50_32x4d(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            nn.Dropout2d(p=0.2),
            resnet.layer2,
            resnet.layer3,
            nn.Dropout2d(p=0.2)
        )

        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._out_dim_encoder = 4096
        self._pos_dim = latent_dim // 2
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(
            self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self._out_dim_encoder, latent_dim // 2, bias=False),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._feat_mlp(self._pretrained_modules(patches).reshape(-1, self._n_patches, self._out_dim_encoder))
        patches_encoding = torch.cat([
            patches_encoding*self._pos_embedding.to(imgs.device),
            patches_encoding*self._relative_pos_embedding.to(imgs.device),
        ], dim=2)
        # patches_encoding: B x n_patches x 4096
        patches = patches_encoding
        # patches: B x n_patches x latent_dim

        return patches


class ImageNetPretrainedResNext50_32x4d_3(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, dropout=0.9, freeze=True, device='cpu'):
        super(ImageNetPretrainedResNext50_32x4d_3, self).__init__()

        self._device = device
        self._patch_sz = 28
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = resnext50_32x4d(pretrained=True).to(device)
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

        self._pos_dim = 4096
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._dropout = nn.Dropout2d(p=dropout)
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
        patches_encoding = self._dropout(self._pretrained_modules(patches).reshape(-1, self._n_patches, self._pos_dim, 1))
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._pos_dim)*(self._pos_embedding.to(imgs.device))
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class ImageNetPretrainedResNext50_32x4d_4(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPretrainedResNext50_32x4d_4, self).__init__()

        self._device = device
        self._patch_sz = 28
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = resnext50_32x4d(pretrained=True).to(device)
        self._preconv = nn.Conv2d(6, 3, kernel_size=1)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            nn.Dropout2d(p=0.2),
            resnet.layer2,
            resnet.layer3,
            nn.Dropout2d(p=0.2)
        )

        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._out_dim_encoder = 4096
        self._pos_dim = 3*self._patch_sz*self._patch_sz
        self._pos_embedding = create_learnable_positional_embeddings(self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self._out_dim_encoder, latent_dim),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        relative_pos_embedding = F.avg_pool2d(relative_pos_embedding, 3, 1, padding=1)
        relative_pos_embedding = relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(self._n_patches, self._pos_dim)
        pos_emb = self._pos_embedding.reshape(self._n_patches, 3, self._patch_sz, self._patch_sz)
        rel_pos = relative_pos_embedding.reshape(self._n_patches, 3, self._patch_sz, self._patch_sz)
        pos_info = torch.cat([pos_emb, rel_pos], dim=1)
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, self._n_patches, 3, self._patch_sz, self._patch_sz)
        patches = patches.repeat([1, 1, 2, 1, 1])
        patches = patches*(pos_info.to(imgs.device))
        patches = self._preconv(patches.reshape(-1, 6, self._patch_sz, self._patch_sz))
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._feat_mlp(self._pretrained_modules(patches).reshape(-1, self._n_patches, self._out_dim_encoder))
        # patches_encoding: B x n_patches x 4096
        # patches: B x n_patches x latent_dim

        return patches_encoding


class ImageNetPretrainedResNext50_32x4d_5(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPretrainedResNext50_32x4d_5, self).__init__()

        self._device = device
        self._patch_sz = 28
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = resnext50_32x4d(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.relu,
            resnet.maxpool,
            nn.Dropout2d(p=0.3),
            resnet.layer1,
            nn.Dropout2d(p=0.8),
            resnet.layer2,
            nn.Dropout2d(p=0.8),
            resnet.layer3,
            nn.Dropout2d(p=0.8),
        )

        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = 4096
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


class ImageNetPretrainedResNext50_32x4d_6(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPretrainedResNext50_32x4d_6, self).__init__()

        self._device = device
        self._patch_sz = 32
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = resnext50_32x4d(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.LeakyReLU(),
            resnet.maxpool,
            resnet.layer1,
            nn.Dropout2d(p=0.3),
            resnet.layer2,
            resnet.layer3,
            nn.Dropout2d(p=0.1),
        )

        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = 4096
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(
            self._n_patches, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim*2, latent_dim)
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = patches_encoding.reshape(-1, self._n_patches, self._pos_dim)
        patches_encoding = torch.cat([
            patches_encoding*(self._pos_embedding.to(imgs.device)),
            patches_encoding*(self._relative_pos_embedding.to(imgs.device)),
        ], dim=2)
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches


class ImageNetPretrainedResNext50_32x4d_7(FeatureNetBase):
    """
    Feature net for ImageNet that uses pretrained pytorch models
    """
    def __init__(self, latent_dim, freeze=True, device='cpu'):
        super(ImageNetPretrainedResNext50_32x4d_7, self).__init__()

        self._device = device
        self._patch_sz = 32
        self._n_patches_1d = 224 // self._patch_sz
        self._n_patches = self._n_patches_1d**2
        resnet = resnext50_32x4d(pretrained=True).to(device)
        self._pretrained_modules = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.LeakyReLU(),
            resnet.maxpool,
            resnet.layer1,
            nn.Dropout2d(p=0.3),
            resnet.layer2,
            resnet.layer3,
            nn.Dropout2d(p=0.1),
        )

        if freeze:
            for p in self._pretrained_modules.parameters():
                p.requires_grad = False

        self._pos_dim = 1024
        self._pos_embedding = create_sinusoidal_positional_embeddings(self._n_patches, self._pos_dim)
        self._relative_pos_embedding = self._pos_embedding.reshape(self._n_patches_1d, self._n_patches_1d,
                                                                   self._pos_dim).transpose(1, 2).transpose(0, 1)
        self._relative_pos_embedding = F.avg_pool2d(self._relative_pos_embedding, 3, 1, padding=1)
        self._relative_pos_embedding = self._relative_pos_embedding.transpose(0, 1).transpose(1, 2).reshape(
            self._n_patches, self._pos_dim)
        self._projection = nn.Linear(4096, self._pos_dim)
        self._feat_mlp = nn.Sequential(
            nn.Linear(self._pos_dim*2, self._pos_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self._pos_dim, self._pos_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self._pos_dim // 2, self._pos_dim // 4),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self._pos_dim // 4, latent_dim),
            nn.LeakyReLU(),
        )
        self._latent_dim = latent_dim
        self._unfold = Unfold(self._patch_sz, stride=self._patch_sz, padding=0)

    def process_samples(self, imgs):
        # patching
        patches = self._unfold(imgs).transpose(1, 2).reshape(-1, 3, self._patch_sz, self._patch_sz)
        # patches: B x n_patches x 3 x 28 x 28
        patches_encoding = self._pretrained_modules(patches)
        patches_encoding = self._projection(patches_encoding.reshape(-1, self._n_patches, 4096))
        patches_encoding = torch.cat([
            patches_encoding*(self._pos_embedding.to(imgs.device)),
            patches_encoding*(self._relative_pos_embedding.to(imgs.device)),
        ], dim=2)
        # patches_encoding: B x n_patches x 4096
        patches = self._feat_mlp(patches_encoding)
        # patches: B x n_patches x latent_dim

        return patches
