import torch
import os
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, ImageFolder
from torchvision.transforms import transforms as T, InterpolationMode
from torch.utils.data import random_split, DataLoader
from kme.data.cub import Cub2011
from kme.data.risk_datasets import RiskDataset
from kme.data.benchmark_datasets import CreditCardFraudDataset, HeartDataset
from kme.data.text_classification import DATASETS as TEXT_DATASETS, get_text_dataset, CollateProcessor
from kme.data.compas import CompasDataset
from kme.data.tcr_datasets import get_tcr_dataset
from kme.data.single_cell_datasets import get_single_cell_dataset
from torch.utils.data import Subset
from kme.tools.config import load_config
from kme.data.MedMnist import get_medmnist_dataset


def get_loader_dataset(loader):
    dataset = loader.dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


TRANSFORMS = {
    'cifar10': {
        'train': T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4, padding_mode='reflect'),
            T.RandomAffine(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
    },
    'svhn': {
        'train': T.Compose([
            T.RandomCrop(32, padding=4, padding_mode='reflect'),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomGrayscale(p=0.5),
            T.ToTensor(),
            T.Normalize((0.4376821, 0.4437697, 0.47280442),
                        (0.19803012, 0.20101562, 0.19703614))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.4376821, 0.4437697, 0.47280442),
                        (0.19803012, 0.20101562, 0.19703614))
        ]),
    },
    'svhn_no_augment': {
        'train': T.Compose([
            T.ToTensor(),
            T.Normalize((0.4376821, 0.4437697, 0.47280442),
                        (0.19803012, 0.20101562, 0.19703614))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.4376821, 0.4437697, 0.47280442),
                        (0.19803012, 0.20101562, 0.19703614))
        ]),
    },
    'mnist': {
        'train': T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ]),
    },
    'fashionmnist': {
        'train': T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ]),
    },
    'imagenet': {
        'train': T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': T.Compose([
            T.RandomResizedCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    },
    'cub': {
        'train': T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(224, padding=14),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomGrayscale(p=0.2),
        ]),
        'test': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    },
    'cub_no_augment': {
        'train': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    },
}
TRANSFORMS['cifar100'] = TRANSFORMS['cifar10']
REVERSE_TRANSFORM = {
    'cub': T.Normalize(mean=-np.array([0.485, 0.456, 0.406])/np.array([0.229, 0.224, 0.225]), std=1./np.array([0.229, 0.224, 0.225])),
    'svhn': T.Normalize(mean=-np.array([0.4376821, 0.4437697, 0.47280442])/np.array([0.19803012, 0.20101562, 0.19703614]), std=1./np.array([0.19803012, 0.20101562, 0.19703614])),
    'mnist': T.Normalize(mean=-np.array([0.1307])/np.array([0.3081]), std=1./np.array([0.3081])),
    'dermamnist': T.Normalize(mean=-np.array([0.5])/np.array([0.5]), std=1./np.array([0.5]))
}


DATASETS = {
    'cifar10': {
        'train': lambda root, args: CIFAR10(root=root, train=True, download=True, transform=TRANSFORMS['cifar10']['train']),
        'test': lambda root, args: CIFAR10(root=root, train=False, download=True, transform=TRANSFORMS['cifar10']['test'])
    },
    'cifar100': {
        'train': lambda root, args: CIFAR100(root=root, train=True, download=True, transform=TRANSFORMS['cifar100']['train']),
        'test': lambda root, args: CIFAR100(root=root, train=False, download=True, transform=TRANSFORMS['cifar100']['test'])
    },
    'svhn': {
        'train': lambda root, args: SVHN(root=root, split='train', download=True, transform=TRANSFORMS['svhn']['train']),
        'test': lambda root, args: SVHN(root=root, split='test', download=True, transform=TRANSFORMS['svhn']['test'])
    },
    'svhn_no_augment': {
        'train': lambda root, args: SVHN(root=root, split='train', download=True, transform=TRANSFORMS['svhn_no_augment']['train']),
        'test': lambda root, args: SVHN(root=root, split='test', download=True, transform=TRANSFORMS['svhn_no_augment']['test'])
    },
    'mnist': {
        'train': lambda root, args: MNIST(root=root, train=True, download=True, transform=TRANSFORMS['mnist']['train']),
        'test': lambda root, args: MNIST(root=root, train=False, download=True, transform=TRANSFORMS['mnist']['test'])
    },
    'fashionmnist': {
        'train': lambda root, args: FashionMNIST(root=root, train=True, download=True, transform=TRANSFORMS['fashionmnist']['train']),
        'test': lambda root, args: FashionMNIST(root=root, train=False, download=True, transform=TRANSFORMS['fashionmnist']['test'])
    },
    'adult': {
        'train': lambda root, args: RiskDataset(root=root, dataset='adult', transform=None)
    },
    'mushroom': {
        'train': lambda root, args: RiskDataset(root=root, dataset='mushroom', transform=None)
    },
    'mammo': {
        'train': lambda root, args: RiskDataset(root=root, dataset='mammo', transform=None)
    },
    'spambase': {
        'train': lambda root, args: RiskDataset(root=root, dataset='mushroom', transform=None)
    },
    'bank': {
        'train': lambda root, args: RiskDataset(root=root, dataset='mushroom', transform=None)
    },
    'compas': {
        'train': lambda root, args:  CompasDataset(root=root)
    },
    'credit': {
        'train': lambda root, args:  CreditCardFraudDataset(root=root)
    },
    'heart': {
        'train': lambda root, args:  HeartDataset(root=root)
    },
    'imagenet': {
        'train': lambda root, args: ImageFolder(root=os.path.join(root, 'train'), transform=TRANSFORMS['imagenet']['train']),
        'test': lambda root, args: ImageFolder(root=os.path.join(root, 'val'), transform=TRANSFORMS['imagenet']['test'])
    },
    'cub': {
        'train': lambda root, args: Cub2011(root=root, transform=TRANSFORMS['cub']['train'], train=True),
        'test': lambda root, args: Cub2011(root=root, transform=TRANSFORMS['cub']['test'], train=False)
    },
    'cub_no_augment': {
        'train': lambda root, args: Cub2011(root=root, transform=TRANSFORMS['cub_no_augment']['train'], train=True),
        'test': lambda root, args: Cub2011(root=root, transform=TRANSFORMS['cub_no_augment']['test'], train=False)
    },
}


def get_loaders(dataset, dataroot, valid_split=0.2, batch_size=100, random_seed=None, test_split=0.2, dataset_args={},
                shuffle=True, device="cpu"):

    if dataset in TEXT_DATASETS.keys():
        train_set, test_set = get_text_dataset(
            dataset, root=dataroot, **dataset_args)
        collate_fn = CollateProcessor(dataset, train_set.get_vocab())
    elif dataset == 'tcr':
        train_set, test_set = get_tcr_dataset(device, **dataset_args)
        collate_fn = None
    elif dataset == 'single_cell':
        train_set, test_set = get_single_cell_dataset(device, **dataset_args)
        collate_fn = None
    elif dataset == 'medmnist':
        train_set, test_set = get_medmnist_dataset(device, dataroot)
        collate_fn = None
    else:
        collate_fn = None
        train_set = DATASETS[dataset]['train'](dataroot, dataset_args)
        if 'test' in DATASETS[dataset].keys():
            test_set = DATASETS[dataset]['test'](dataroot, dataset_args)
        else:
            train_size = int(len(train_set)*(1.-test_split))
            test_size = int(len(train_set) - train_size)

            if random_seed is not None:
                torch.manual_seed(random_seed)
            train_set, test_set = random_split(
                train_set, [train_size, test_size])

    train_size = int(len(train_set)*(1.-valid_split))
    valid_size = int(len(train_set) - train_size)

    if random_seed is not None:
        torch.manual_seed(random_seed)
    train_set, valid_set = random_split(train_set, [train_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=shuffle, collate_fn=collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn, drop_last=True)

    return train_loader, valid_loader, test_loader
