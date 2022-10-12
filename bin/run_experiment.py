from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.nn import DataParallel
from kme.data.text_classification import DATASETS as TEXT_DATASETS, MAX_SENTENCE_LEN
from kme.tools.config import load_config, save_config, DATASET_KEY, TRAINING_KEY, MODEL_KEY, CKPT_KEY
from kme.tools.training import train_routine, test_routine
from kme.tools.checkpoint import load_checkpoint, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from kme.models.utils import build_kme_net, count_parameters
from kme.data.utils import get_loaders, get_loader_dataset
from torch.optim import *
import numpy as np
import torch.optim as optim
from pandas import json_normalize
import argparse
import torch


def run(config_path: str, interactive_flag=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)

    dataset_params = config[DATASET_KEY]
    training_params = config[TRAINING_KEY]
    model_params = config[MODEL_KEY]
    ckpt_root = config[CKPT_KEY]
    dataset = dataset_params["dataset"]

    # training params
    N_EPOCHS = training_params['n_epochs']
    BATCH_SIZE = training_params['batch_size']
    OPTIMIZER = training_params['optimizer']
    OPTIMIZER_PARAMS = training_params['optimizer_params']
    SCHEDULER = training_params['scheduler']
    if 'norm_reg' in training_params:
        NORM_REG = training_params['norm_reg']
    else:
        NORM_REG = 0.

    # dataset
    train_loader, valid_loader, test_loader = get_loaders(
        **dataset_params, batch_size=BATCH_SIZE, device=device)

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

    if torch.cuda.device_count() > 1:
        kme_net = DataParallel(kme_net)
    kme_net = kme_net.to(device)
    print(count_parameters(kme_net))

    writer = SummaryWriter(log_dir=ckpt_root)

    # training routine
    for e in range(epoch + 1, N_EPOCHS + 1):
        kme_net.train()
        train_loss, train_accuracy, train_running_norm = \
            train_routine(e, kme_net, optimizer, train_loader, device, dataset=dataset, if_tqdm=interactive_flag, if_print=False,
                          norm_reg=NORM_REG)
        kme_net.eval()
        valid_loss, valid_accuracy = test_routine(
            e, kme_net, valid_loader, device, dataset=dataset)
        save_checkpoint(ckpt_root, kme_net, optimizer, scheduler, e - 1)
        writer.add_scalar('accuracy/train', train_accuracy, global_step=e)
        writer.add_scalar('accuracy/valid', valid_accuracy, global_step=e)
        writer.add_scalar('loss/train', train_loss, global_step=e)
        writer.add_scalar('loss/valid', valid_loss, global_step=e)
        if scheduler is not None:
            scheduler.step()

    # test metrics
    if dataset_params['dataset'] == 'tcr':

        test_loss, test_accuracy, true_labels, predicted_labels = test_routine(
            N_EPOCHS, kme_net, test_loader, device, dataset, return_labels_and_preds=True)
        predicted_labels_norm = torch.softmax(
            torch.tensor(predicted_labels), 1)
        ras = roc_auc_score(
            true_labels, predicted_labels_norm[:, 1])
        metrics_dict = {
            'test_auc': ras,
            'test_balanced_acc': balanced_accuracy_score(true_labels, np.argmax(predicted_labels, axis=1))
        }
        print('AUC: {}. Balanced Accuracy: {}'.format(
            metrics_dict['test_auc'], metrics_dict['test_balanced_acc']))

    elif dataset_params['dataset'] == 'single_cell':
        test_loss, test_accuracy, true_labels, predicted_labels = test_routine(
            N_EPOCHS, kme_net, test_loader, device, dataset,  return_labels_and_preds=True)

        predicted_labels_norm = torch.softmax(
            torch.tensor(predicted_labels), 1)
        ras = roc_auc_score(
            true_labels, predicted_labels_norm, multi_class='ovr')

        metrics_dict = {
            'test_auc': ras,
            'test_balanced_acc': balanced_accuracy_score(true_labels, np.argmax(predicted_labels, axis=1))
        }
        print('AUC: {}. Balanced Accuracy: {}'.format(
            metrics_dict['test_auc'], metrics_dict['test_balanced_acc']))
    else:
        test_loss, test_accuracy = test_routine(N_EPOCHS, kme_net, test_loader, device, dataset,
                                                return_labels_and_preds=False)
        metrics_dict = dict()

    metrics_dict.update({
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    })

    # hyperparameters
    if scheduler is None:
        config[TRAINING_KEY]['scheduler'] = ''
    hparams_dict = json_normalize(config, sep='_').to_dict(orient='records')[0]
    for k in list(hparams_dict.keys()):
        if hasattr(hparams_dict[k], '__class__'):
            del hparams_dict[k]
    hparams_dict['parameters_count'] = count_parameters(kme_net)

    # write to summary
    writer.add_hparams(
        hparams_dict,
        metrics_dict
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ExperimentRunner',
                                     description='Routine that run an experiment according to a config file')
    parser.add_argument('--config_file', type=str,
                        required=True, help='Path to the config file to use')
    parser.add_argument('--interactive', action='store_true',
                        help='If defined, show training bar.')

    args = parser.parse_args()
    run(args.config_file, args.interactive)
