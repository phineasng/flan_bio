
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import medmnist
from medmnist import INFO, Evaluator


def get_medmnist_dataset(device, dataroot, data_flag='dermamnist', download=False):

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(
        root=dataroot + "/Train", split='train', transform=data_transform, download=download)
    test_dataset = DataClass(
        root=dataroot + "/Test", split='test', transform=data_transform, download=download)

    # Count class samples
    classes = [0]*7
    for i in range(len(train_dataset)):
        cl = train_dataset[i][1].tolist()[0]
        classes[cl] += 1

    return train_dataset, test_dataset


def medmnist_roc_auc(y_true, y_score):
    auc = 0
    for i in range(y_score.shape[1]):
        y_true_binary = (y_true == i).astype(float)
        y_score_binary = y_score[:, i]
        auc += roc_auc_score(y_true_binary, y_score_binary)
    ret = auc / y_score.shape[1]

    return ret
