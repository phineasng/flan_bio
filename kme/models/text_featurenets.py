import torch
from torch import nn
from kme.models.base import create_sinusoidal_positional_embeddings
from kme.data.text_classification import PAD_TOKEN
from kme.models.base import FeatureNetBase


class TextFeatNet1(FeatureNetBase):
    """
    Basic text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet1, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len, self._latent_dim).to(device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._emb_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self._latent_dim),
            nn.LeakyReLU(),
        )

    def process_samples(self, s):
        s = self._embeddings(s)
        s = self._feature_encoder(s)*(self._pos_embeddings.to(s.device))
        return s


class TextFeatNet2(FeatureNetBase):
    """
    Basic text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet2, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len, self._latent_dim).to(device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._emb_dim, 2048),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self._latent_dim),
            nn.LeakyReLU(),
        )

    def process_samples(self, s):
        s = self._embeddings(s)
        s = self._feature_encoder(s) + (self._pos_embeddings.to(s.device))
        return s


class TextFeatNet3(FeatureNetBase):
    """
    Basic text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet3, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len, 2048).to(device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._emb_dim, 2048),
            nn.LeakyReLU()
        )
        self._feature_encoder_2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, self._latent_dim),
            nn.LeakyReLU(),
        )

    def process_samples(self, s):
        s = self._embeddings(s)
        s = self._feature_encoder(s)*(self._pos_embeddings.to(s.device))
        s = self._feature_encoder_2(s)
        return s


class TextFeatNet4(FeatureNetBase):
    """
    Basic text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet4, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len, 2048).to(device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._emb_dim, 2048),
            nn.Dropout(p=0.2),
            nn.LeakyReLU()
        )
        self._feature_encoder_2 = nn.Sequential(
            nn.Linear(2048, self._latent_dim),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
        )

    def process_samples(self, s):
        mask = (s != self._vocab.stoi[PAD_TOKEN])
        s = self._embeddings(s)
        s = (s.transpose(1, 2).transpose(0, 1) *
             mask).transpose(0, 1).transpose(1, 2)
        s = self._feature_encoder(s)*(self._pos_embeddings.to(s.device))
        s = self._feature_encoder_2(s)
        return s


class TextFeatNet5(FeatureNetBase):
    """
    Basic text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet5, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len, latent_dim).to(device)
        self._feature_encoder = nn.Linear(
            self._emb_dim, latent_dim, bias=False)

    def process_samples(self, s):
        mask = (s != self._vocab.stoi[PAD_TOKEN])
        s = self._embeddings(s)
        s = (s.transpose(1, 2).transpose(0, 1) *
             mask).transpose(0, 1).transpose(1, 2)
        s = self._feature_encoder(s)*(self._pos_embeddings.to(s.device))
        return s


class TextFeatNet6(FeatureNetBase):
    """
    Basic text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet6, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len, 2048).to(device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._emb_dim, 2048, bias=False),
            nn.Dropout(p=0.2)
        )
        self._feature_encoder_2 = nn.Sequential(
            nn.Linear(2048, self._latent_dim),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
        )

    def process_samples(self, s):
        mask = (s != self._vocab.stoi[PAD_TOKEN])
        s = self._embeddings(s)
        s = (s.transpose(1, 2).transpose(0, 1) *
             mask).transpose(0, 1).transpose(1, 2)
        s = self._feature_encoder(s)*(self._pos_embeddings.to(s.device))
        s = self._feature_encoder_2(s)
        return s


class TextFeatNet7(FeatureNetBase):
    """
    CNN text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet7, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._stride = 5
        self._pos_dim = 1024

        self._conv = nn.Sequential(
            nn.Conv1d(self._emb_dim, self._emb_dim,
                      kernel_size=self._stride, stride=self._stride),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(self._emb_dim, self._pos_dim, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len // self._stride, self._pos_dim).to(device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._pos_dim, self._latent_dim),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
        )

    def process_samples(self, s):
        s = self._embeddings(s)
        # s : B x MAX_LEN x emb_dim
        s = s.transpose(1, 2)
        # S : B x emb_dim x MAX_LEN
        s = self._conv(s)
        # s : B x pos_dim x (MAX_LEN//5)
        s = s.transpose(1, 2)
        # s : B x (MAX_LEN//5) x pos_dim
        s = s*(self._pos_embeddings.to(s.device))
        s = self._feature_encoder(s)
        # s : B x (MAX_LEN//5) x latent_dim
        return s


class TextFeatNet8(FeatureNetBase):
    """
    CNN text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet8, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._stride = 5
        self._pos_dim = 32

        self._conv = nn.Sequential(
            nn.Conv1d(self._emb_dim, self._emb_dim,
                      kernel_size=self._stride, stride=self._stride),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(self._emb_dim, self._pos_dim, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len // self._stride, self._pos_dim).to(device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._pos_dim, self._latent_dim),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
        )

    def process_samples(self, s):
        s = self._embeddings(s)
        # s : B x MAX_LEN x emb_dim
        s = s.transpose(1, 2)
        # S : B x emb_dim x MAX_LEN
        s = self._conv(s)
        # s : B x pos_dim x (MAX_LEN//5)
        s = s.transpose(1, 2)
        # s : B x (MAX_LEN//5) x pos_dim
        s = s*(self._pos_embeddings.to(s.device))
        s = self._feature_encoder(s)
        # s : B x (MAX_LEN//5) x latent_dim
        return s


class TextFeatNet9(FeatureNetBase):
    """
    CNN text feature network
    """

    def __init__(self, emb_dim, latent_dim, vocabulary, max_sentence_len, device='cpu'):
        super(TextFeatNet9, self).__init__()

        self._emb_dim = emb_dim
        self._latent_dim = latent_dim
        self._vocab = vocabulary
        self._vocab.stoi = self._vocab.get_stoi()
        self._max_len = max_sentence_len

        self._stride = 5
        self._pos_dim = 32

        self._conv = nn.Sequential(
            nn.Conv1d(self._emb_dim, self._emb_dim,
                      kernel_size=self._stride, stride=self._stride),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(self._emb_dim, self._emb_dim, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(self._emb_dim, self._emb_dim, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(self._emb_dim, self._emb_dim, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(self._emb_dim, self._pos_dim, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )

        self._embeddings = nn.Embedding(len(self._vocab), embedding_dim=self._emb_dim,
                                        padding_idx=self._vocab.stoi[PAD_TOKEN])
        self._pos_embeddings = create_sinusoidal_positional_embeddings(
            self._max_len // self._stride, self._pos_dim).to(device)
        self._feature_encoder = nn.Sequential(
            nn.Linear(self._pos_dim, self._latent_dim),
            nn.LeakyReLU(),
        )

    def process_samples(self, s):
        s = self._embeddings(s)
        # s : B x MAX_LEN x emb_dim
        s = s.transpose(1, 2)
        # S : B x emb_dim x MAX_LEN
        s = self._conv(s)
        # s : B x pos_dim x (MAX_LEN//5)
        s = s.transpose(1, 2)
        # s : B x (MAX_LEN//5) x pos_dim
        s = s*(self._pos_embeddings.to(s.device))
        # s : B x (MAX_LEN//5) x latent_dim
        s = self._feature_encoder(s)

        return s
