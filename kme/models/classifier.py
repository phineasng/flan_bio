import torch
import os
from torch import nn
from kme.models.base import init_weights
from kme.tools.checkpoint import CHECKPOINT_FNAME, MODEL_STATE_DICT_KEY
from kme.tools.config import load_config, DATASET_KEY, MODEL_KEY
from kme.data.text_classification import DATASETS as TEXT_DATASETS, MAX_SENTENCE_LEN


class HomoLatentClassifier(nn.Module):
    """
    Classifier based on a latent representation of a CIFAR image
    """

    def __init__(self, latent_dim, n_classes):
        super(HomoLatentClassifier, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 256, bias=False),
            nn.LeakyReLU(),
            nn.Linear(256, 256, bias=False),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, 256, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256, bias=False),
            nn.LeakyReLU(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256, bias=False),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, n_classes, bias=False),
        )

    def forward(self, x):
        return self._classifier(x)


class SmallHomoClassifier(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(SmallHomoClassifier, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 64, bias=False),
            nn.LeakyReLU(),
            nn.Linear(64, 16, bias=False),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(16, n_classes, bias=False),
        )

    def forward(self, x):
        return self._classifier(x)


class SmallHomoClassifier2(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(SmallHomoClassifier2, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024, bias=False),
            nn.LeakyReLU(),
            nn.Linear(1024, n_classes, bias=False),
        )

    def forward(self, x):
        return self._classifier(x)


class SmallHomoClassifier3(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(SmallHomoClassifier3, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, n_classes, bias=False),
        )

    def forward(self, x):
        return self._classifier(x)


class SmallClassifier(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(SmallClassifier, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class SmallClassifier2(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(SmallClassifier2, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class SmallClassifier3(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(SmallClassifier3, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class WideClassifier1(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(WideClassifier1, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class WideClassifierDrop(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(WideClassifierDrop, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class WideClassifierDrop1024(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(WideClassifierDrop1024, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class WideClassifier2(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(WideClassifier2, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class WideClassifierDrop2(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(WideClassifierDrop2, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class HugeClassifier1(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(HugeClassifier1, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class Classifier1(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(Classifier1, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class Classifier2(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(Classifier2, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class Classifier3(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(Classifier3, self).__init__()
        self._classifier = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self._classifier(x)


class Identity(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Linear(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(Linear, self).__init__()
        self._classifier = nn.Linear(latent_dim, n_classes, bias=False)

    def forward(self, x):
        return self._classifier(x)


class PreTrainedClassifier(nn.Module):
    def __init__(self, pretrained_root, config_name, freeze=True):
        super(PreTrainedClassifier, self).__init__()
        config_path = os.path.join(pretrained_root, config_name)
        config = load_config(config_path)
        model_params = config[MODEL_KEY]
        self._config_path = config_path

        classifier_cls = model_params['classifier']
        classifier_args = model_params['classifier_args']
        self._classifier = globals()[classifier_cls](**classifier_args)
        ckpt_fpath = os.path.join(pretrained_root, CHECKPOINT_FNAME)

        def adapt_key(k):
            kk = k.split('.')
            return '.'.join(kk[2:])
        if os.path.exists(ckpt_fpath):
            state = torch.load(ckpt_fpath)
            cls_state = {adapt_key(
                k): v for k, v in state[MODEL_STATE_DICT_KEY].items() if '_feat_net' not in k}
            self._classifier.load_state_dict(cls_state)
        if freeze:
            for p in self._classifier.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self._classifier(x)


class TCRReconstructor1(nn.Module):
    def __init__(self, dataset, classifier_class: str = "WideClassifier", classifier_args: dict = None):
        super(TCRReconstructor1, self).__init__()
        self._classifier = globals()[classifier_class](**classifier_args)
        self._latent_dim = classifier_args['latent_dim']
        self._dataset = dataset
        self._languages = [
            ds.protein_language for ds in self._dataset.datasets]

        self._standard_hidden_dim = 256
        self._projector = nn.Linear(
            self._latent_dim, self._standard_hidden_dim)
        self._decoders = nn.ModuleList()
        self._final_projections = nn.ModuleList()

        for i, lang in enumerate(self._languages):
            max_seq_len = dataset.datasets[i].padding_length
            vocab_dim = lang.number_of_tokens

            self._decoders.append(nn.Sequential(
                nn.ConvTranspose1d(1, 4, kernel_size=4),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(4, 64, kernel_size=4),
                nn.LeakyReLU(),
                nn.LayerNorm([64, self._standard_hidden_dim + 6]),
                nn.ConvTranspose1d(64, 128, kernel_size=4),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.ConvTranspose1d(128, vocab_dim, kernel_size=4),
                nn.LeakyReLU(),
            ))

            self._final_projections.append(
                nn.Linear(self._standard_hidden_dim + 12, max_seq_len)
            )

    def forward(self, z):
        y = self._classifier(z)
        outs = [y]
        z = self._projector(z)
        z = z.reshape(-1, 1, self._standard_hidden_dim)
        for i in range(len(self._languages)):
            x_hat = self._decoders[i](z)
            x_hat = self._final_projections[i](x_hat).transpose(1, 2)

            outs.append(x_hat)

        return tuple(outs)
