from kme.data.text_classification import DATASETS as TEXT_DATASETS, MAX_SENTENCE_LEN
from kme.tools.config import load_config, DATASET_KEY, TRAINING_KEY, MODEL_KEY, CKPT_KEY
from kme.tools.checkpoint import load_checkpoint
from kme.models.utils import build_kme_net
from kme.data.utils import get_loaders, get_loader_dataset
from torch.optim import *
import torch.optim as optim


def load_kme_model(config_path, device):

    config = load_config(config_path)

    dataset_params = config[DATASET_KEY]
    training_params = config[TRAINING_KEY]
    model_params = config[MODEL_KEY]
    ckpt_root = config[CKPT_KEY]

    # training params
    BATCH_SIZE = training_params['batch_size']
    OPTIMIZER = training_params['optimizer']
    OPTIMIZER_PARAMS = training_params['optimizer_params']
    SCHEDULER = training_params['scheduler']

    # dataset
    train_loader, valid_loader, test_loader = get_loaders(
        **dataset_params, batch_size=BATCH_SIZE, device=device)

    # create model
    if 'use_mean' not in model_params:
        model_params['use_mean'] = False

    if dataset_params['dataset'] in TEXT_DATASETS:
        model_params['feature_net_args']['vocabulary'] = train_loader.dataset.dataset.get_vocab()
        model_params['feature_net_args']['max_sentence_len'] = MAX_SENTENCE_LEN[dataset_params['dataset']]
    elif dataset_params['dataset'] == 'tcr':
        dataset_instance = get_loader_dataset(train_loader)
        model_params['feature_net_args']['dataset'] = dataset_instance

    kme_net = build_kme_net(model_params, device=device)
    kme_net._feat_net.use_mean(model_params['use_mean'])

    # optimization
    if OPTIMIZER == "SAM":
        OPTIMIZER_PARAMS["base_optimizer"] = globals(
        )[OPTIMIZER_PARAMS["base_optimizer"]]
    optimizer = globals()[OPTIMIZER](kme_net.parameters(), **OPTIMIZER_PARAMS)
    scheduler = None
    if SCHEDULER is not None:
        SCHEDULER_PARAMS = training_params['scheduler_params']
        scheduler = getattr(optim.lr_scheduler, SCHEDULER)(
            optimizer, **SCHEDULER_PARAMS)

    kme_net, optimizer, scheduler, epoch = load_checkpoint(
        ckpt_root, kme_net, optimizer, scheduler, device=device)

    return kme_net, train_loader, test_loader
