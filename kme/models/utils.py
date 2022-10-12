from kme.models.base import *
from kme.models.cifar import *
from kme.models.mnist import *
from kme.models.tabular_nets import *
from kme.models.classifier import *
from kme.models.text_featurenets import *
from kme.models.te_featurenets import *
from kme.models.single_cell_featurenets import *
from kme.models.imagenet import *
import torch.nn as nn


def build_kme_net(build_args: dict, device='cpu'):
    feature_net_cls = build_args['feature_net']
    classifier_cls = build_args['classifier']

    feature_net_args = build_args['feature_net_args']
    classifier_args = build_args['classifier_args']

    feature_net = globals()[feature_net_cls](**feature_net_args, device=device)
    classifier = globals()[classifier_cls](**classifier_args)

    kme_net = KMEBase(feature_net, classifier)
    return kme_net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
