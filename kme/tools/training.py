import torch
from torch import nn
import tqdm
import numpy as np
from kme.extern.sam import SAM


def compute_accuracy(pred_y, y, mean_reduce=False):
    pred = torch.argmax(pred_y, dim=1)
    correct = pred.eq(y).sum().float()
    if mean_reduce:
        return correct.item() / float(len(y))
    return correct.item()


def train_routine(epoch, model, optimizer, loader, device, dataset, if_tqdm=False, if_print=False, norm_reg=0.):
    if if_tqdm:
        bar = tqdm.tqdm(enumerate(loader))
    else:
        bar = enumerate(loader)

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.
    running_acc = 0.
    running_mean_norm = 0.

    for i, data in bar:
        if dataset == "tcr":
            imgs = [d.to(device) for d in data[:-1]]
            labels = torch.squeeze(data[-1].long()).to(device)
        elif dataset == "single_cell":
            imgs = data[:, :-1].to(device)
            labels = torch.squeeze(data[:, -1].long()).to(device)
        else:
            imgs = data[0].to(device)
            labels = data[1].to(device).squeeze()

        if norm_reg > 1e-10:
            pred_y, mean_norm = model(imgs, return_norm=True)
            mean_norm = torch.mean(mean_norm)
            loss = criterion(pred_y, labels) + norm_reg*mean_norm

        pred_y = model(imgs, return_norm=False)
        mean_norm = torch.zeros([1]).to(labels.device)
        loss = criterion(pred_y, labels) + norm_reg*mean_norm

        loss.backward()
        if isinstance(optimizer, SAM):
            optimizer.first_step(zero_grad=True)

            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
                    module.eval()

            if norm_reg > 1e-10:
                pred_y, mean_norm = model(imgs, return_norm=True)
                mean_norm = torch.mean(mean_norm)
            else:
                pred_y = model(imgs, return_norm=False)
                mean_norm = torch.zeros([1]).to(labels.device)

            loss_second_pass = criterion(pred_y, labels) + norm_reg*mean_norm
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
        if if_tqdm:
            bar.set_description('%d] %d/%d Loss: %f, Accuracy: %f, Mean Nuc Norm %f' % (
                epoch, i+1, len(loader), running_loss, running_acc, running_mean_norm))
        elif if_print:
            print('%d] %d/%d Loss: %f, Accuracy: %f, Mean Nuc Norm %f' % (
                epoch, i+1, len(loader), running_loss, running_acc, running_mean_norm))
    return running_loss, running_acc, running_mean_norm


def test_routine(epoch, model, loader, device, dataset, return_labels_and_preds=False):
    total_loss = 0.
    total_correct = 0
    n_samples = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    true_labels = []
    prediction_scores = []

    for data in loader:
        if dataset == "tcr":
            imgs = [d.to(device) for d in data[:-1]]
            labels = torch.squeeze(data[-1].long()).to(device)
        elif dataset == "single_cell":
            imgs = data[:, :-1].to(device)
            labels = torch.squeeze(data[:, -1].long()).to(device)
        else:
            imgs = data[0].to(device)
            labels = data[1].to(device).squeeze()

        true_labels.append(labels.detach().cpu().numpy())

        pred_y = model(imgs)

        prediction_scores.append(pred_y.detach().cpu().numpy())

        loss = criterion(pred_y, labels)
        correct = compute_accuracy(pred_y, labels, mean_reduce=False)

        total_loss += loss.item()
        total_correct += correct
        n_samples += len(labels)

    print('%d] Loss: %f, Accuracy %f' %
          (epoch, total_loss/float(n_samples), total_correct/float(n_samples)))

    model.train()

    if return_labels_and_preds:
        return total_loss / float(n_samples), total_correct / float(n_samples), \
            np.concatenate(true_labels), np.concatenate(
                prediction_scores, axis=0)
    return total_loss/float(n_samples), total_correct/float(n_samples)
