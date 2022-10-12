import torch, argparse
from pandas import json_normalize
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.optim import *
from kme.data.utils import get_loaders, get_loader_dataset
from kme.models.utils import build_kme_net, count_parameters
from torch.utils.tensorboard import SummaryWriter
from kme.tools.training import compute_accuracy
from kme.tools.checkpoint import load_checkpoint, save_checkpoint
from kme.tools.config import load_config, save_config, DATASET_KEY, TRAINING_KEY, MODEL_KEY, CKPT_KEY
from kme.data.text_classification import DATASETS as TEXT_DATASETS, MAX_SENTENCE_LEN
from torch.nn import DataParallel
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from kme.extern.radam import RAdam
from kme.extern.sam import SAM
import tqdm
from torch.distributions import Categorical


def train_routine(epoch, model, optimizer, loader, device, if_tqdm=False, if_print=False, norm_reg=0.):
    if if_tqdm:
        bar = tqdm.tqdm(enumerate(loader))
    else:
        bar = enumerate(loader)

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.
    running_acc = 0.
    running_mean_norm = 0.
    running_ligand_err = 0.
    running_receptor_err = 0.

    dataset_class = get_loader_dataset(loader).__class__

    for i, data in bar:
        if "Protein" in dataset_class.__name__:
            imgs = [d.to(device) for d in data[:-1]]
            labels = torch.squeeze(data[-1].long()).to(device)
        else:
            imgs = data[0].to(device)
            labels = data[1].to(device)

        if norm_reg > 1e-10:
            out, mean_norm = model(imgs, return_norm=True)
            mean_norm = torch.mean(mean_norm)
        else:
            out = model(imgs, return_norm=False)
            mean_norm = torch.zeros([1]).to(labels.device)

        pred_y = out[0]
        ligand_hat = out[1]
        receptor_hat = out[2]

        dist_ligand = Categorical(logits=ligand_hat)
        dist_receptor = Categorical(logits=receptor_hat)

        rec_err_ligand = dist_ligand.log_prob(imgs[0]).sum(-1).mean()
        rec_err_receptor = dist_receptor.log_prob(imgs[1]).sum(-1).mean()

        w_lig = 0.
        w_recep = 0.

        loss = criterion(pred_y, labels) + norm_reg*mean_norm - w_lig*rec_err_ligand - w_recep*rec_err_receptor

        loss.backward()
        if isinstance(optimizer, SAM):
            optimizer.first_step(zero_grad=True)

            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                    module.eval()

            if norm_reg > 1e-10:
                out, mean_norm = model(imgs, return_norm=True)
                mean_norm = torch.mean(mean_norm)
            else:
                out = model(imgs, return_norm=False)
                mean_norm = torch.zeros([1]).to(labels.device)

            pred_y = out[0]
            ligand_hat = out[1]
            receptor_hat = out[2]

            dist_ligand = Categorical(logits=ligand_hat)
            dist_receptor = Categorical(logits=receptor_hat)

            rec_err_ligand = dist_ligand.log_prob(imgs[0]).sum(-1).mean()
            rec_err_receptor = dist_receptor.log_prob(imgs[1]).sum(-1).mean()

            loss_second_pass = criterion(pred_y, labels) + norm_reg * mean_norm - w_lig * rec_err_ligand - w_recep * rec_err_receptor

            loss_second_pass.backward()
            optimizer.second_step(zero_grad=True)
            model.train()
        else:
            optimizer.step()
        optimizer.zero_grad()


        running_loss = (running_loss*i + loss.item())/(i+1)
        curr_accuracy = compute_accuracy(pred_y, labels, mean_reduce=True)
        running_acc = (running_acc*i + curr_accuracy)/(i+1)
        running_mean_norm = (running_mean_norm*i + mean_norm.item())/(i+1)
        running_ligand_err = (running_ligand_err*i + rec_err_ligand.item())/(i+1)
        running_receptor_err = (running_receptor_err*i + rec_err_receptor.item())/(i+1)
        if if_tqdm:
            bar.set_description('%d] %d/%d Loss: %f, Accuracy: %f, Mean Nuc Norm %f, Ligand Rec: %f, Receptor Rec: %f' % (
                epoch, i+1, len(loader), running_loss, running_acc, running_mean_norm, running_ligand_err, running_receptor_err))
        elif if_print:
            print('%d] %d/%d Loss: %f, Accuracy: %f, Mean Nuc Norm %f' % (
                epoch, i+1, len(loader), running_loss, running_acc, running_mean_norm))

    return running_loss, running_acc, running_mean_norm


def test_routine(epoch, model, loader, device, return_labels_and_preds=False):
    total_loss = 0.
    total_correct = 0
    n_samples = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    dataset_class = get_loader_dataset(loader).__class__

    true_labels = []
    prediction_scores = []

    for data in loader:
        if "Protein" in dataset_class.__name__:
            imgs = [d.to(device) for d in data[:-1]]
            labels = torch.squeeze(data[-1].long()).to(device)
        else:
            imgs = data[0].to(device)
            labels = data[1].to(device)

        true_labels.append(labels.detach().cpu().numpy())

        out = model(imgs)
        pred_y = out[0]

        prediction_scores.append(pred_y.detach().cpu().numpy())

        loss = criterion(pred_y, labels)
        correct = compute_accuracy(pred_y, labels, mean_reduce=False)

        total_loss += loss.item()
        total_correct += correct
        n_samples += len(labels)

    print('%d] Loss: %f, Accuracy %f' % (epoch, total_loss/float(n_samples), total_correct/float(n_samples)))

    model.train()

    if return_labels_and_preds:
        return total_loss / float(n_samples), total_correct / float(n_samples), \
               np.concatenate(true_labels), np.concatenate(prediction_scores, axis=0)
    return total_loss/float(n_samples), total_correct/float(n_samples)


def run(config_path: str, interactive_flag=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)

    dataset_params = config[DATASET_KEY]
    training_params = config[TRAINING_KEY]
    model_params = config[MODEL_KEY]
    ckpt_root = config[CKPT_KEY]

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
    train_loader, valid_loader, test_loader = get_loaders(**dataset_params, batch_size=BATCH_SIZE)

    # create model
    if 'use_mean' not in model_params:
        model_params['use_mean'] = False
    if dataset_params['dataset'] in TEXT_DATASETS:
        model_params['feature_net_args']['vocabulary'] = train_loader.dataset.dataset.get_vocab()
        model_params['feature_net_args']['max_sentence_len'] = MAX_SENTENCE_LEN[dataset_params['dataset']]
    elif dataset_params['dataset'] == 'tcr':
        dataset_instance = get_loader_dataset(train_loader)
        model_params['feature_net_args']['dataset'] = dataset_instance
        model_params['classifier_args']['dataset'] = dataset_instance
    kme_net = build_kme_net(model_params, device=device)
    kme_net._feat_net.use_mean(model_params['use_mean'])
    kme_net = kme_net.to(device)

    # optimization
    if OPTIMIZER == "SAM":
        OPTIMIZER_PARAMS["base_optimizer"] = globals()[OPTIMIZER_PARAMS["base_optimizer"]]
    optimizer = globals()[OPTIMIZER](kme_net.parameters(), **OPTIMIZER_PARAMS)
    scheduler = None
    if SCHEDULER is not None:
        SCHEDULER_PARAMS = training_params['scheduler_params']
        scheduler = getattr(optim.lr_scheduler, SCHEDULER)(optimizer, **SCHEDULER_PARAMS)

    # checkpoint and summary writer
    kme_net, optimizer, scheduler, epoch = load_checkpoint(ckpt_root, kme_net, optimizer, scheduler, device=device)

    if  torch.cuda.device_count() > 1:
        kme_net = DataParallel(kme_net)
    kme_net = kme_net.to(device)
    print(count_parameters(kme_net))

    writer = SummaryWriter(log_dir=ckpt_root)

    # training routine
    for e in range(epoch + 1, N_EPOCHS + 1):
        kme_net.train()
        train_loss, train_accuracy, train_running_norm = \
            train_routine(e, kme_net, optimizer, train_loader, device, if_tqdm=interactive_flag, if_print=False,
                          norm_reg=NORM_REG)
        kme_net.eval()
        valid_loss, valid_accuracy = test_routine(e, kme_net, valid_loader, device)
        save_checkpoint(ckpt_root, kme_net, optimizer, scheduler, e - 1)
        writer.add_scalar('accuracy/train', train_accuracy, global_step=e)
        writer.add_scalar('accuracy/valid', valid_accuracy, global_step=e)
        writer.add_scalar('loss/train', train_loss, global_step=e)
        writer.add_scalar('loss/valid', valid_loss, global_step=e)
        if scheduler is not None:
            scheduler.step()

    # test metrics
    if dataset_params['dataset'] == 'tcr':
        test_loss, test_accuracy, true_labels, predicted_labels = test_routine(N_EPOCHS, kme_net, test_loader, device, return_labels_and_preds=True)
        metrics_dict = {
            'test_auc': roc_auc_score(true_labels, predicted_labels[:, 1]),
            'test_balanced_acc': balanced_accuracy_score(true_labels, np.argmax(predicted_labels, axis=1))
        }
        print('AUC: {}. Balanced Accuracy: {}'.format(metrics_dict['test_auc'], metrics_dict['test_balanced_acc']))
    else:
        test_loss, test_accuracy, true_labels, predicted_labels = test_routine(N_EPOCHS, kme_net, test_loader, device,
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
    parser.add_argument('--config_file', type=str, required=True, help='Path to the config file to use')
    parser.add_argument('--interactive', action='store_true', help='If defined, show training bar.')

    args = parser.parse_args()
    run(args.config_file, args.interactive)
