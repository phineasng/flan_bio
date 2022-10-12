import torch
import argparse
from pandas import json_normalize
import torch.optim as optim
from torch.optim import *
from kme.data.utils import get_loaders, DATASETS, REVERSE_TRANSFORM, TRANSFORMS
from kme.models.utils import build_kme_net, count_parameters
from torch.utils.tensorboard import SummaryWriter
from kme.tools.checkpoint import load_checkpoint, save_checkpoint
from kme.tools.training import train_routine, test_routine
from kme.tools.config import load_config, save_config, DATASET_KEY, TRAINING_KEY, MODEL_KEY, CKPT_KEY
from kme.data.text_classification import DATASETS as TXT_DS, MAX_SENTENCE_LEN
from torch.nn import DataParallel
from kme.extern.radam import RAdam
import tqdm
import os
import numpy as np
from sklearn.neighbors import KDTree
import pickle
from captum.attr import IntegratedGradients, Saliency, InputXGradient
from captum.attr._utils.visualization import visualize_image_attr
from torchvision.transforms import Resize
from matplotlib import pyplot as plt
from kme.tools.interpreting import match_single_patches
from matplotlib.cm import get_cmap
import pandas as pd


IMAGE_DATASETS = {'cifar10', 'cifar100', 'cub', 'svhn', 'svhn_no_augment',
                  'mnist', 'fashionmnist', 'imagenet', 'cub_no_augment'}
N_CHANNELS = {
    'mnist': 1,
    'fashionmnist': 1,
    'cifar10': 3,
    'cifar100': 3,
    'cub': 3,
    'cub_no_augment': 3,
    'svhn': 3,
    'imagenet': 3,
    'svhn_no_augment': 3
}
IMG_SZ = {
    'mnist': 28,
    'fashionmnist': 28,
    'cifar10': 32,
    'cifar100': 32,
    'cub': 224,
    'cub_no_augment': 224,
    'svhn': 32,
    'imagenet': 224,
    'svhn_no_augment': 32
}
TEXT_DATASETS = set(TXT_DS.keys())
TABULAR_DATASETS = set(DATASETS.keys()).difference(IMAGE_DATASETS)


X_fname = 'X.npy'
Y_fname = 'Y.npy'
probs_fname = 'probs.npy'
latent_fname = 'latent.npy'
latent_norms_fname = 'latent_norms.npy'
latent_probs_fname = 'latent_probs.npy'
sample_tree_fname = 'sample_tree.pkl'
per_feature_tree_fname = 'feat_{}_tree.pkl'
joint_tree_fname = 'joint_tree.pkl'
feature_attribution_fname = 'feat_attr_{}.pkl'


def check_latent_files(root):
    for fname in [X_fname, Y_fname, probs_fname, latent_fname]:
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            return False
    return True


def load(config_path, device):
    config = load_config(config_path)

    dataset_params = config[DATASET_KEY]
    if dataset_params['dataset'] == 'cub':
        dataset_params['dataset'] = 'cub_no_augment'
    training_params = config[TRAINING_KEY]
    model_params = config[MODEL_KEY]
    ckpt_root = config[CKPT_KEY]
    dataset_params['shuffle'] = False

    # training params
    N_EPOCHS = training_params['n_epochs']
    BATCH_SIZE = 10
    OPTIMIZER = training_params['optimizer']
    OPTIMIZER_PARAMS = training_params['optimizer_params']
    SCHEDULER = training_params['scheduler']
    if 'norm_reg' in training_params:
        NORM_REG = training_params['norm_reg']
    else:
        NORM_REG = 0.

    # dataset
    train_loader, valid_loader, test_loader = get_loaders(
        **dataset_params, batch_size=BATCH_SIZE)

    # create model
    if 'use_mean' not in model_params:
        model_params['use_mean'] = False
    if dataset_params['dataset'] in TEXT_DATASETS:
        model_params['feature_net_args']['vocabulary'] = train_loader.dataset.dataset.get_vocab()
        model_params['feature_net_args']['max_sentence_len'] = MAX_SENTENCE_LEN[dataset_params['dataset']]
    kme_net = build_kme_net(model_params, device=device)
    kme_net._feat_net.use_mean(model_params['use_mean'])
    # optimization
    optimizer = globals()[OPTIMIZER](kme_net.parameters(), **OPTIMIZER_PARAMS)
    scheduler = None
    if SCHEDULER is not None:
        SCHEDULER_PARAMS = training_params['scheduler_params']
        scheduler = getattr(optim.lr_scheduler, SCHEDULER)(
            optimizer, **SCHEDULER_PARAMS)
    # checkpoint and summary writer
    kme_net, optimizer, scheduler, epoch = load_checkpoint(
        ckpt_root, kme_net, optimizer, scheduler, device=device)
    print(count_parameters(kme_net))

    if torch.cuda.device_count() > 1:
        kme_net = DataParallel(kme_net)
    kme_net = kme_net.to(device)

    return kme_net, train_loader, test_loader, ckpt_root, dataset_params['dataset']


def compute_latent(model, loader, root, device):
    X = []
    Y = []
    probs = []
    latent_repr_pre_agg = []

    for x, y in tqdm.tqdm(loader):
        x = x.to(device)
        p = model(x)
        latent = model._feat_net.process_samples(x)

        X.append(x.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())
        Y.append(y.detach().cpu().numpy())
        latent_repr_pre_agg.append(latent.detach().cpu().numpy())

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    probs = np.concatenate(probs, axis=0)
    latent_repr_pre_agg = np.concatenate(latent_repr_pre_agg, axis=0)

    np.save(os.path.join(root, X_fname), X)
    np.save(os.path.join(root, Y_fname), Y)
    np.save(os.path.join(root, probs_fname), probs)
    np.save(os.path.join(root, latent_fname), latent_repr_pre_agg)
    return


def create_trees(root):
    fpath = os.path.join(root, latent_fname)
    if not os.path.exists(fpath):
        raise RuntimeError(
            "File '{}' not detected. Please create it using this program with flag '--compute_latent'".format(fpath))
    latent = np.load(fpath)

    # create per feature tree
    for i in tqdm.tqdm(range(latent.shape[1])):
        feat_repr = latent[:, i, :]
        tree = KDTree(feat_repr, leaf_size=40)
        with open(os.path.join(root, per_feature_tree_fname.format(i)), 'wb') as out_file:
            pickle.dump(tree, out_file)

    # create sample tree
    sample_repr = np.sum(latent, axis=1)
    tree = KDTree(sample_repr, leaf_size=40)
    with open(os.path.join(root, sample_tree_fname), 'wb') as out_file:
        pickle.dump(tree, out_file)

    # create joint tree
    joint_repr = np.reshape(latent, (-1, latent.shape[2]))
    tree = KDTree(joint_repr, leaf_size=40)
    with open(os.path.join(root, joint_tree_fname), 'wb') as out_file:
        pickle.dump(tree, out_file)

    return


def compute_native_attributions(model, loader, dataset, device):
    attributions = []
    if dataset in IMAGE_DATASETS:
        fold = torch.nn.Fold(
            output_size=IMG_SZ[dataset], kernel_size=model._feat_net._patch_sz, stride=model._feat_net._patch_sz)
    for x, _ in tqdm.tqdm(loader):
        x = x.to(device)
        latent_repr = model._feat_net.process_samples(x)
        curr_attr = torch.norm(latent_repr, dim=2, p=2).unsqueeze(dim=1)

        if dataset in IMAGE_DATASETS:
            curr_attr = curr_attr.repeat(
                1, N_CHANNELS[dataset]*(model._feat_net._patch_sz**2), 1)
            curr_attr = fold(curr_attr)

        curr_attr = curr_attr.transpose(1, 2).transpose(2, 3)
        attributions.append(curr_attr.detach().cpu().numpy())
    return np.concatenate(attributions, axis=0)


def compute_captum_attributions(model, loader, dataset, method, device):
    attributions = []

    attr_algo = globals()[method](model)

    for x, _ in tqdm.tqdm(loader):
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1)

        if method == 'IntegratedGradients':
            curr_attr = attr_algo.attribute(x, target=preds, baselines=0.)
        else:
            curr_attr = attr_algo.attribute(x, target=preds)

        if dataset in IMAGE_DATASETS:
            curr_attr = curr_attr.transpose(1, 2).transpose(2, 3)

        attributions.append(curr_attr.detach().cpu().numpy())

    return np.concatenate(attributions, axis=0)


def compute_feature_attributions(model, loader, results_root, methods, dataset, device):
    for m in methods:
        if m == 'native':
            attributions = compute_native_attributions(
                model, loader, dataset, device)
        else:
            attributions = compute_captum_attributions(
                model, loader, dataset, m, device)
        np.save(os.path.join(results_root,
                feature_attribution_fname.format(m)), attributions)
    return


def attribute_image(model, loader, id, device, root):
    folder = os.path.join(root, 'attributions')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, 'flan_{}.png'.format(id))
    img = loader.dataset[id][0].to(device).unsqueeze(0)
    resize = Resize(loader.dataset.get_size(id))
    unnormalize = REVERSE_TRANSFORM['cub']

    fold = torch.nn.Fold(
        output_size=IMG_SZ[dataset], kernel_size=model._feat_net._patch_sz, stride=model._feat_net._patch_sz)
    latent_repr = model._feat_net.process_samples(img)

    curr_attr = torch.norm(latent_repr, dim=2, p=2).unsqueeze(dim=1)
    curr_attr = curr_attr.repeat(
        1, N_CHANNELS[dataset]*(model._feat_net._patch_sz**2), 1)
    curr_attr = fold(curr_attr)
    curr_attr = resize(unnormalize(curr_attr)).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    img = resize(unnormalize(img)).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    fig, ax = visualize_image_attr(
        curr_attr, img, cmap='Reds', use_pyplot=False, method='blended_heat_map')
    fig.savefig(path, dpi=300)
    print('Saved to {}'.format(path))


def attribute_image_comparison(model, loader, id, device, root):
    folder = os.path.join(root, 'attributions')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, 'comparison_{}.png'.format(id))
    img = loader.dataset[id][0].to(device).unsqueeze(0)
    size = loader.dataset.get_size(id)
    resize = Resize(size)
    unnormalize = REVERSE_TRANSFORM['cub']
    target = torch.argmax(model(img), dim=1)

    fold = torch.nn.Fold(
        output_size=IMG_SZ[dataset], kernel_size=model._feat_net._patch_sz, stride=model._feat_net._patch_sz)
    latent_repr = model._feat_net.process_samples(img)

    # Compute native attributions
    attr_native = torch.norm(latent_repr, dim=2, p=2).unsqueeze(dim=1)
    attr_native = attr_native.repeat(
        1, N_CHANNELS[dataset]*(model._feat_net._patch_sz**2), 1)
    attr_native = fold(attr_native)
    attr_native = resize(attr_native).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    # Compute IG attributions
    algo = IntegratedGradients(model)
    attr_ig = algo.attribute(img, target=target)
    attr_ig = resize(attr_ig).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    # Compute Saliency
    algo = Saliency(model)
    attr_saliency = algo.attribute(img, target=target)
    attr_saliency = resize(attr_saliency).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    # Compute Saliency
    algo = InputXGradient(model)
    attr_ixg = algo.attribute(img, target=target)
    attr_ixg = resize(attr_ixg).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    img_numpy = resize(unnormalize(img)).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 4)

    visualize_image_attr(attr_native, img_numpy, cmap='Reds', use_pyplot=False,
                         method='blended_heat_map', plt_fig_axis=(fig, ax[0]))
    ax[0].set_title('Latent Norms')
    ax[0].title.set_size(8)
    visualize_image_attr(attr_ig, img_numpy, cmap='Reds', use_pyplot=False,
                         method='blended_heat_map', plt_fig_axis=(fig, ax[1]))
    ax[1].set_title('Integrated Gradients')
    ax[1].title.set_size(8)
    visualize_image_attr(attr_saliency, img_numpy, cmap='Reds', use_pyplot=False,
                         method='blended_heat_map', plt_fig_axis=(fig, ax[2]))
    ax[2].set_title('Saliency')
    ax[2].title.set_size(8)
    visualize_image_attr(attr_ixg, img_numpy, cmap='Reds', use_pyplot=False,
                         method='blended_heat_map', plt_fig_axis=(fig, ax[3]))
    ax[3].set_title('InputXGradient')
    ax[3].title.set_size(8)

    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved to {}'.format(path))


def analyze_top_patches(model, loader, id, device, root, top_k, top_k_vis=3):
    folder = os.path.join(root, 'masking')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, 'top{}_{}.png'.format(top_k, id))
    img = loader.dataset[id][0].to(device).unsqueeze(0)
    true_label = loader.dataset[id][1]
    size = loader.dataset.get_size(id)
    resize = Resize(size)
    unnormalize = REVERSE_TRANSFORM['cub']

    probs = torch.softmax(model(img), dim=1)
    target = torch.argmax(probs, dim=1).detach(
    ).cpu().numpy().astype(np.int)[0]
    probs = probs.detach().cpu().numpy()[0]

    fold = torch.nn.Fold(
        output_size=IMG_SZ[dataset], kernel_size=model._feat_net._patch_sz, stride=model._feat_net._patch_sz)
    unfold = model._feat_net._unfold
    img_orig_color = unnormalize(img)
    patches = unfold(img_orig_color).transpose(
        1, 2).reshape(-1, 3, model._feat_net._patch_sz, model._feat_net._patch_sz)
    latent_repr = model._feat_net.process_samples(img)

    # Compute native attributions and retrieve top_k
    attr_native = torch.norm(latent_repr, dim=2, p=2).squeeze()
    top_k_norms = torch.topk(attr_native, k=top_k)[1]

    fig, ax = plt.subplots(1, top_k_vis + 1)

    for i in range(top_k_vis):
        idx = top_k_norms[i]
        patch_vis = patches[idx:idx+1]
        patch_vis = resize(patch_vis).transpose(
            1, 2).transpose(2, 3)[0].detach().cpu().numpy()

        patch_pred = torch.softmax(model._classifier(
            latent_repr[:, idx, :]), dim=1).squeeze()
        top_3_patch_pred = torch.topk(patch_pred, k=3)
        patch_pred_idx = top_3_patch_pred[1].detach(
        ).cpu().numpy().astype(np.int)
        patch_pred_probs = top_3_patch_pred[0].detach().cpu().numpy()

        curr_title = '{}: {:.2f}\n{}: {:.2f}\n{}: {:.2f}'.format(
            loader.dataset.get_label(patch_pred_idx[0]), patch_pred_probs[0],
            loader.dataset.get_label(patch_pred_idx[1]), patch_pred_probs[1],
            loader.dataset.get_label(patch_pred_idx[2]), patch_pred_probs[2],
        )

        ax[i+1].imshow(patch_vis)
        ax[i+1].xaxis.set_ticks_position("none")
        ax[i+1].yaxis.set_ticks_position("none")
        ax[i+1].set_yticklabels([])
        ax[i+1].set_xticklabels([])
        ax[i+1].grid(b=False)
        ax[i+1].set_title(curr_title)
        ax[i+1].title.set_size(4)

    masked_img = torch.zeros(patches.shape, device=device)
    masked_img[top_k_norms] = patches[top_k_norms]
    masked_img = fold(masked_img.reshape(-1, model._feat_net._n_patches,
                      3*model._feat_net._patch_sz*model._feat_net._patch_sz).transpose(1, 2))
    masked_img = resize(masked_img).transpose(
        1, 2).transpose(2, 3)[0].detach().cpu().numpy()
    partial_repr = torch.sum(latent_repr[:, top_k_norms, :], dim=1)
    partial_probs = torch.softmax(model._classifier(partial_repr), dim=1)
    partial_target = torch.argmax(
        partial_probs, dim=1).detach().cpu().numpy().astype(np.int)[0]
    partial_probs = partial_probs.detach().cpu().numpy()[0]
    probs_ord_idx = np.argsort(partial_probs)
    last_title = '(Full) {}: {:.2f}\n(Partial-1) {}: {:.2f}\n(Partial-2) {}: {:.2f}\n(Partial-3) {}: {:.2f}\n(True) {}'.format(
        loader.dataset.get_label(target), probs[target],
        loader.dataset.get_label(
            partial_target), partial_probs[partial_target],
        loader.dataset.get_label(
            probs_ord_idx[-2]), partial_probs[probs_ord_idx[-2]],
        loader.dataset.get_label(
            probs_ord_idx[-3]), partial_probs[probs_ord_idx[-3]],
        loader.dataset.get_label(true_label),
    )
    ax[0].imshow(masked_img)
    ax[0].xaxis.set_ticks_position("none")
    ax[0].yaxis.set_ticks_position("none")
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[0].grid(b=False)
    ax[0].set_title(last_title)
    ax[0].title.set_size(4)

    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved to {}'.format(path))


def correspondence_images(model, loader, train_loader, id, device, root, top_k=10):
    if not check_latent_files(root):
        print('Pre-Computing latent representations')
        compute_latent(model, train_loader, root, device)

    folder = os.path.join(root, 'correspondences')
    os.makedirs(folder, exist_ok=True)

    # useful operations
    fold = torch.nn.Fold(
        output_size=IMG_SZ[dataset], kernel_size=model._feat_net._patch_sz, stride=model._feat_net._patch_sz)
    unfold = model._feat_net._unfold

    # get the image of interest
    img = loader.dataset[id][0].to(device).unsqueeze(0)
    true_label = loader.dataset[id][1]

    # compute its representation
    patches_rep = model._feat_net.process_samples(img).detach().cpu().numpy()
    sample_repr = np.sum(patches_rep, axis=1).squeeze()

    #
    size = loader.dataset.get_size(id)
    resize = Resize(size)
    unnormalize = REVERSE_TRANSFORM['cub']

    fig, ax = plt.subplots(1, 3)

    def plot_img_at_ax(_img, ax_id, _hl, title='', **kwargs):
        img_numpy = resize(unnormalize(_img))
        img_numpy = img_numpy.transpose(1, 2).transpose(2, 3)[
            0].detach().cpu().numpy()
        ax[ax_id].imshow(img_numpy, alpha=0.5)
        ax[ax_id].imshow(_hl, alpha=0.5, cmap='Reds')
        ax[ax_id].xaxis.set_ticks_position("none")
        ax[ax_id].yaxis.set_ticks_position("none")
        ax[ax_id].set_yticklabels([])
        ax[ax_id].set_xticklabels([])
        ax[ax_id].grid(b=False)
        ax[ax_id].set_title(title)
        ax[ax_id].title.set_size(4)

    # get training latent representation
    train_patches_rep = np.load(os.path.join(root, latent_fname))
    train_samples_rep = np.sum(train_patches_rep, axis=1)

    # retrieve closest samples and farthest sample
    distances = np.linalg.norm(train_samples_rep - sample_repr, axis=1)
    sorted_ids = np.argsort(distances)
    closest_idx = sorted_ids[0]
    farthest_idx = sorted_ids[-1]

    closest_img = train_loader.dataset[closest_idx][0].to(device).unsqueeze(0)
    farthest_img = train_loader.dataset[farthest_idx][0].to(
        device).unsqueeze(0)

    # compute top predictions
    def compute_top_3_probs(_model, _img):
        _probs = torch.softmax(_model(_img), dim=1)
        _top_3_patch_pred = torch.topk(_probs, k=3)
        _pred_idx = _top_3_patch_pred[1].squeeze(
        ).detach().cpu().numpy().astype(np.int)
        _pred_probs = _top_3_patch_pred[0].squeeze().detach().cpu().numpy()

        text = '{}: {:.2f}\n{}: {:.2f}\n{}: {:.2f}'.format(
            loader.dataset.get_label(_pred_idx[0]), _pred_probs[0],
            loader.dataset.get_label(_pred_idx[1]), _pred_probs[1],
            loader.dataset.get_label(_pred_idx[2]), _pred_probs[2],
        )
        return text

    def compute_correspondences(_model, _ref_img, _other_img, minimize=True):
        # get patches representations
        _ref_patches_repr = _model._feat_net.process_samples(_ref_img)
        attr_native = torch.norm(_ref_patches_repr, dim=2, p=2).squeeze()
        top_k_norms = torch.topk(attr_native, k=top_k)[
            1].squeeze().detach().cpu().numpy().astype(np.int)
        _ref_patches_repr = _ref_patches_repr.detach().cpu().numpy()
        _other_patches_repr = _model._feat_net.process_samples(
            _other_img).detach().cpu().numpy()

        correspondence = match_single_patches(
            _ref_patches_repr[0][top_k_norms], _other_patches_repr[0], minimize)

        # get colormap
        m = get_cmap('tab10')
        _ref_alpha = np.zeros((_ref_patches_repr.shape[1], 1, 1, 1))
        _other_alpha = np.zeros((_ref_patches_repr.shape[1], 1, 1, 1))

        _ref_hl = np.zeros((_ref_patches_repr.shape[1], 3, 1, 1))
        _other_hl = np.zeros((_ref_patches_repr.shape[1], 3, 1, 1))

        def fold_patch(_patches):
            C = _patches.shape[1]
            P_SZ = _patches.shape[2]

            # _patches = N_P x C x P_SZ x P_SZ
            _patches = torch.FloatTensor(_patches)
            _patches = _patches.reshape(1, -1, C*P_SZ*P_SZ).transpose(1, 2)
            folded_img = resize(fold(_patches)[0]).transpose(
                0, 1).transpose(1, 2).detach().cpu().numpy()
            return folded_img

        color = np.array([1.0, 0., 0.])
        for k, i in enumerate(top_k_norms):
            _other_idx = correspondence[1][k]

            _ref_alpha[i, 0, 0, 0] = 1.
            _other_alpha[_other_idx, 0, 0, 0] = 1.

            _ref_hl[i, :, 0, 0] = color
            _other_hl[_other_idx, :, 0, 0] = color

        _ref_alpha = np.repeat(_ref_alpha, _model._feat_net._patch_sz, axis=2).repeat(
            _model._feat_net._patch_sz, axis=3)
        _other_alpha = np.repeat(_other_alpha, _model._feat_net._patch_sz, axis=2).repeat(
            _model._feat_net._patch_sz, axis=3)
        _ref_hl = np.repeat(_ref_hl, _model._feat_net._patch_sz, axis=2).repeat(
            _model._feat_net._patch_sz, axis=3)
        _other_hl = np.repeat(_other_hl, _model._feat_net._patch_sz, axis=2).repeat(
            _model._feat_net._patch_sz, axis=3)

        _ref_alpha = fold_patch(_ref_alpha)
        _other_alpha = fold_patch(_other_alpha)
        _ref_hl = fold_patch(_ref_hl)
        _other_hl = fold_patch(_other_hl)

        _ref_hl = np.concatenate([_ref_hl, _ref_alpha], axis=2)
        _other_hl = np.concatenate([_other_hl, _other_alpha], axis=2)

        return _ref_hl, _other_hl

    _, closest_hl = compute_correspondences(
        model, img, closest_img, minimize=True)
    ref_hl, farthest_hl = compute_correspondences(
        model, img, farthest_img, minimize=False)
    plot_img_at_ax(img, 0, ref_hl, compute_top_3_probs(model, img))
    plot_img_at_ax(closest_img, 1, closest_hl,
                   compute_top_3_probs(model, closest_img))
    plot_img_at_ax(farthest_img, 2, farthest_hl,
                   compute_top_3_probs(model, farthest_img))

    path = os.path.join(folder, '{}_{}_{}.png'.format(
        id, closest_idx, farthest_idx))
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved to {}'.format(path))


def compute_latent_norms(root):
    latent_reprs = np.load(os.path.join(root, latent_fname))
    latent_norms = np.linalg.norm(latent_reprs, axis=2)
    np.save(os.path.join(root, latent_norms_fname), latent_norms)
    return latent_norms


def compute_latent_probs(model, root, device):
    latent_reprs = np.load(os.path.join(root, latent_fname))
    n_patches = latent_reprs.shape[1]
    latent_probs = []

    latent_reprs = torch.FloatTensor(
        latent_reprs).reshape(-1, latent_reprs.shape[-1])

    for i in tqdm.tqdm(range(0, len(latent_reprs), 30)):
        curr_repr = latent_reprs[i:i+30].to(device)

        probs = torch.softmax(model._classifier(
            curr_repr), dim=1).detach().cpu().numpy()
        latent_probs.append(probs)

    latent_probs = np.concatenate(latent_probs, axis=0).reshape(
        (-1, n_patches, probs.shape[-1]))

    np.save(os.path.join(root, latent_probs_fname), latent_probs)
    return latent_probs


def create_counterfactual(model, loader, train_loader, id, device, root, top_k, target_class, consistent):
    if not check_latent_files(root):
        print('Pre-Computing latent representations')
        compute_latent(model, train_loader, root, device)

    latent_norms_path = os.path.join(root, latent_norms_fname)
    if not os.path.exists(latent_norms_path):
        latent_norms = compute_latent_norms(root)
    else:
        latent_norms = np.load(latent_norms_path)

    latent_probs_path = os.path.join(root, latent_probs_fname)
    if not os.path.exists(latent_probs_path):
        latent_probs = compute_latent_probs(model, root, device)
    else:
        latent_probs = np.load(latent_probs_path)

    folder = os.path.join(root, 'counterfactuals')
    os.makedirs(folder, exist_ok=True)

    # useful operations
    fold = torch.nn.Fold(
        output_size=IMG_SZ[dataset], kernel_size=model._feat_net._patch_sz, stride=model._feat_net._patch_sz)
    unfold = model._feat_net._unfold

    # get the image of interest
    img = loader.dataset[id][0].to(device).unsqueeze(0)
    true_label = loader.dataset[id][1]

    # compute its representation
    patches_rep = model._feat_net.process_samples(img).detach().cpu().numpy()
    sample_repr = np.sum(patches_rep, axis=1).squeeze()

    #
    size = loader.dataset.get_size(id)
    resize = Resize(size)
    unnormalize = REVERSE_TRANSFORM['cub']

    if size[0] > size[1]:
        fig, ax = plt.subplots(3, 1)
    else:
        fig, ax = plt.subplots(1, 3)

    def img2patches(_img):
        patches = unfold(_img).transpose(1, 2).reshape(-1, 3,
                                                       model._feat_net._patch_sz, model._feat_net._patch_sz)
        return patches

    def patches2img(_patches):
        C = _patches.shape[1]
        P_SZ = _patches.shape[2]

        # _patches = N_P x C x P_SZ x P_SZ
        _patches = _patches.reshape(1, -1, C*P_SZ*P_SZ).transpose(1, 2)
        folded_img = fold(_patches)
        return folded_img

    def compute_counterfactual(_model, _ref_img, _change_top):
        # get patches representations
        _ref_patches_repr = _model._feat_net.process_samples(_ref_img)
        attr_native = torch.norm(_ref_patches_repr, dim=2, p=2).squeeze()
        top_k_norms = torch.topk(attr_native, k=top_k, largest=_change_top)[
            1].squeeze().detach().cpu().numpy().astype(np.int)
        _ref_patches_repr = _ref_patches_repr.detach().cpu().numpy()

        def get_best_candidate(_idx):
            curr_other_norms = latent_norms[:, _idx:_idx+1]
            curr_other_probs = latent_probs[:, _idx:_idx+1, target_class]

            df = pd.DataFrame(np.concatenate(
                [curr_other_norms, curr_other_probs], axis=1), columns=['norms', 'probs'])
            df['rank'] = df.sort_values(by=['norms', 'probs'], ascending=False)[
                'probs'].index + 1
            return df.index[df['rank'] == 1].tolist()[0]

        _ref_patches = img2patches(img)

        if consistent:
            best_candidate_idx = get_best_candidate(top_k_norms[0])
            candidate_img = train_loader.dataset[best_candidate_idx][0].to(
                device).unsqueeze(0)
            candidate_patches = img2patches(candidate_img)
            for i in top_k_norms:
                _ref_patches[i] = candidate_patches[i]
        else:
            for i in top_k_norms:
                best_candidate_idx = get_best_candidate(i)
                candidate_img = train_loader.dataset[best_candidate_idx][0].to(
                    device).unsqueeze(0)
                candidate_patches = img2patches(candidate_img)
                _ref_patches[i] = candidate_patches[i]

        return patches2img(_ref_patches)

    def plot_img_at_ax(_img, ax_id, title='', **kwargs):
        img_numpy = resize(unnormalize(_img))
        img_numpy = img_numpy.transpose(1, 2).transpose(2, 3)[
            0].detach().cpu().numpy()
        ax[ax_id].imshow(img_numpy)
        ax[ax_id].xaxis.set_ticks_position("none")
        ax[ax_id].yaxis.set_ticks_position("none")
        ax[ax_id].set_yticklabels([])
        ax[ax_id].set_xticklabels([])
        ax[ax_id].grid(b=False)
        ax[ax_id].set_title(title)
        ax[ax_id].title.set_size(4)

    def compute_top_3_probs(_model, _img):
        _probs = torch.softmax(_model(_img), dim=1)
        _top_3_patch_pred = torch.topk(_probs, k=3)
        _pred_idx = _top_3_patch_pred[1].squeeze(
        ).detach().cpu().numpy().astype(np.int)
        _pred_probs = _top_3_patch_pred[0].squeeze().detach().cpu().numpy()

        text = '{}: {:.2f}\n{}: {:.2f}\n{}: {:.2f}'.format(
            loader.dataset.get_label(_pred_idx[0]), _pred_probs[0],
            loader.dataset.get_label(_pred_idx[1]), _pred_probs[1],
            loader.dataset.get_label(_pred_idx[2]), _pred_probs[2],
        )
        return text

    counter_top = compute_counterfactual(model, img, _change_top=True)
    counter_bottom = compute_counterfactual(model, img, _change_top=False)
    plot_img_at_ax(img, 0, compute_top_3_probs(model, img))
    plot_img_at_ax(counter_top, 1, compute_top_3_probs(model, counter_top))
    plot_img_at_ax(counter_bottom, 2,
                   compute_top_3_probs(model, counter_bottom))

    path = os.path.join(folder, '{}_{}_{}_{}.png'.format(
        id, target_class, loader.dataset.get_label(target_class), consistent))
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved to {}'.format(path))


def transfer_explanation(model, loader, id, device, root, top_k=10):
    mnist_config_path = model._classifier._config_path
    mnist_model, mnist_train_loader, mnist_test_loader, mnist_results_root, mnist_dataset \
        = load(mnist_config_path, device)

    if not check_latent_files(mnist_results_root):
        print('Pre-Computing latent representations')
        compute_latent(mnist_model, mnist_train_loader,
                       mnist_results_root, device)

    folder = os.path.join(root, 'correspondences')
    os.makedirs(folder, exist_ok=True)

    # useful operations
    fold = torch.nn.Fold(
        output_size=IMG_SZ[dataset], kernel_size=model._feat_net._patch_sz, stride=model._feat_net._patch_sz)
    mnist_fold = torch.nn.Fold(
        output_size=IMG_SZ[mnist_dataset], kernel_size=mnist_model._feat_net._patch_sz, stride=mnist_model._feat_net._patch_sz)

    # get the image of interest
    img = loader.dataset[id][0].to(device).unsqueeze(0)

    # compute its representation
    patches_rep = model._feat_net.process_samples(img).detach().cpu().numpy()
    sample_repr = np.sum(patches_rep, axis=1).squeeze()

    #
    size = [32, 32]
    resize = Resize(size)
    unnormalize = REVERSE_TRANSFORM['svhn']
    mnist_unnormalize = REVERSE_TRANSFORM['mnist']

    # prepare plotting
    fig, ax = plt.subplots(1, 2)

    def plot_img_at_ax(_img, ax_id, _unnormalize, _hl, title='', **kwargs):
        img_numpy = resize(_img)
        img_numpy = img_numpy.transpose(1, 2).transpose(2, 3)[
            0].detach().cpu().numpy()
        if img_numpy.shape[2] == 1:
            ax[ax_id].imshow(img_numpy, alpha=0.8, cmap='gray')
        else:
            ax[ax_id].imshow(img_numpy, alpha=0.8)
        ax[ax_id].imshow(_hl, alpha=0.2, cmap='Reds')
        ax[ax_id].xaxis.set_ticks_position("none")
        ax[ax_id].yaxis.set_ticks_position("none")
        ax[ax_id].set_yticklabels([])
        ax[ax_id].set_xticklabels([])
        ax[ax_id].grid(b=False)
        ax[ax_id].set_title(title)
        ax[ax_id].title.set_size(4)

    # get training latent representation
    train_patches_rep = np.load(os.path.join(mnist_results_root, latent_fname))
    train_samples_rep = np.sum(train_patches_rep, axis=1)

    # retrieve closest samples and farthest sample
    distances = np.linalg.norm(train_samples_rep - sample_repr, axis=1)
    sorted_ids = np.argsort(distances)
    closest_idx = sorted_ids[0]

    closest_img = mnist_train_loader.dataset[closest_idx][0].to(
        device).unsqueeze(0)

    # compute top predictions
    def compute_top_3_probs(_model, _img):
        _probs = torch.softmax(_model(_img), dim=1)
        _top_3_patch_pred = torch.topk(_probs, k=3)
        _pred_idx = _top_3_patch_pred[1].squeeze(
        ).detach().cpu().numpy().astype(np.int)
        _pred_probs = _top_3_patch_pred[0].squeeze().detach().cpu().numpy()

        text = '{}: {:.2f}\n{}: {:.2f}\n{}: {:.2f}'.format(
            _pred_idx[0], _pred_probs[0],
            _pred_idx[1], _pred_probs[1],
            _pred_idx[2], _pred_probs[2],
        )
        return text

    def compute_correspondences(_model, _ref_img, _other_model, _other_img):
        # get patches representations
        _ref_patches_repr = _model._feat_net.process_samples(_ref_img)
        attr_native = torch.norm(_ref_patches_repr, dim=2, p=2).squeeze()
        top_k_norms = torch.topk(attr_native, k=top_k)[
            1].squeeze().detach().cpu().numpy().astype(np.int)
        _ref_patches_repr = _ref_patches_repr.detach().cpu().numpy()
        _other_patches_repr = _other_model._feat_net.process_samples(
            _other_img).detach().cpu().numpy()

        correspondence = match_single_patches(
            _ref_patches_repr[0][top_k_norms], _other_patches_repr[0])

        # get colormap
        _ref_alpha = np.zeros((_ref_patches_repr.shape[1], 1, 1, 1))
        _other_alpha = np.zeros((_other_patches_repr.shape[1], 1, 1, 1))

        _ref_hl = np.zeros((_ref_patches_repr.shape[1], 3, 1, 1))
        _other_hl = np.zeros((_other_patches_repr.shape[1], 3, 1, 1))

        def fold_patch(_patches, _fold):
            C = _patches.shape[1]
            P_SZ = _patches.shape[2]

            # _patches = N_P x C x P_SZ x P_SZ
            _patches = torch.FloatTensor(_patches)
            _patches = _patches.reshape(1, -1, C*P_SZ*P_SZ).transpose(1, 2)
            folded_img = resize(_fold(_patches)[0]).transpose(
                0, 1).transpose(1, 2).detach().cpu().numpy()
            return folded_img

        color = np.array([1.0, 0., 0.])
        for k, i in enumerate(top_k_norms):
            _other_idx = correspondence[1][k]

            _ref_alpha[i, 0, 0, 0] = 1.
            _other_alpha[_other_idx, 0, 0, 0] = 1.

            _ref_hl[i, :, 0, 0] = color
            _other_hl[_other_idx, :, 0, 0] = color

        _ref_alpha = np.repeat(_ref_alpha, _model._feat_net._patch_sz, axis=2).repeat(
            _model._feat_net._patch_sz, axis=3)
        _other_alpha = np.repeat(_other_alpha, _other_model._feat_net._patch_sz, axis=2).repeat(
            _other_model._feat_net._patch_sz, axis=3)
        _ref_hl = np.repeat(_ref_hl, _model._feat_net._patch_sz, axis=2).repeat(
            _model._feat_net._patch_sz, axis=3)
        _other_hl = np.repeat(_other_hl, _other_model._feat_net._patch_sz, axis=2).repeat(
            _other_model._feat_net._patch_sz, axis=3)

        _ref_alpha = fold_patch(_ref_alpha, fold)
        _other_alpha = fold_patch(_other_alpha, mnist_fold)
        _ref_hl = fold_patch(_ref_hl, fold)
        _other_hl = fold_patch(_other_hl, mnist_fold)

        _ref_hl = np.concatenate([_ref_hl, _ref_alpha], axis=2)
        _other_hl = np.concatenate([_other_hl, _other_alpha], axis=2)

        return _ref_hl, _other_hl

    ref_hl, closest_hl = compute_correspondences(
        model, img, mnist_model, closest_img)
    plot_img_at_ax(img, 0, unnormalize, ref_hl,
                   compute_top_3_probs(model, img))
    plot_img_at_ax(closest_img, 1, mnist_unnormalize, closest_hl,
                   compute_top_3_probs(mnist_model, closest_img))

    path = os.path.join(folder, '{}_{}.png'.format(id, closest_idx))
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved to {}'.format(path))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        prog='Tool to analyze KME-net'
    )
    parser.add_argument('--config_file', type=str,
                        help='Path to the config to load', required=True)
    parser.add_argument('--print_test_accuracy', action='store_true')
    parser.add_argument('--compute_latent', action='store_true',
                        help='Preprocess the samples to get the latent representations of both samples and features.')
    parser.add_argument('--create_nn_tree', action='store_true',
                        help='If samples have been preprocessed, compute trees.')
    parser.add_argument('--top_k', type=int, default=3,
                        help='How many patches to consider')

    feat_attr_parser_group = parser.add_argument_group('feat_attr')

    feat_attr_parser_group.add_argument('--methods', nargs='*', default=['native', 'IntegratedGradients', 'Saliency',
                                                                         'InputXGradient'],
                                        help='Methods to use to compute the feature attributions.')
    feat_attr_parser_group.add_argument('--compute_on_train', action='store_true',
                                        help='If true, use the training dataset to compute the attributions.')
    feat_attr_parser_group.add_argument('--compute_feature_attribution', action='store_true',
                                        help='Whether to perform feature attribution analysis.')
    feat_attr_parser_group.add_argument('--visualize_feature_attribution', action='store_true',
                                        help='Whether to visualize feature attributions, i.e. '
                                        'generate an image showing the attribution.')
    feat_attr_parser_group.add_argument('--perturbation_analysis', action='store_true',
                                        help='Test attributions.')

    cub_parser_group = parser.add_argument_group('cub dataset')
    cub_parser_group.add_argument('--cub', action='store_true',
                                  help='Perform CUB analysis')
    cub_parser_group.add_argument('--sample_id', type=int, default=-1,
                                  help='ID of the test sample to analyze')
    cub_parser_group.add_argument('--create_attribution_image', action='store_true',
                                  help='Takes a test sample and computes attributions')
    cub_parser_group.add_argument('--create_attribution_comparison', action='store_true',
                                  help='Takes a test sample and compares attributions')
    cub_parser_group.add_argument('--top_k_analysis', action='store_true',
                                  help='Takes a test sample and analyzes top k patches')
    cub_parser_group.add_argument('--target_class', type=int, default=-1,
                                  help='Target Class for counterfactual')
    cub_parser_group.add_argument('--top_k_vis', type=int, default=3,
                                  help='How many patches to visualize')
    cub_parser_group.add_argument('--create_correspondence', action='store_true',
                                  help='Takes a test sample and computes correspondence')
    cub_parser_group.add_argument('--consistent', action='store_true',
                                  help='If create a consistent counterfactual')
    cub_parser_group.add_argument('--create_counterfactual', action='store_true',
                                  help='Takes a test sample and computes counterfactual')

    svhn_parser_group = parser.add_argument_group('svhn dataset')
    svhn_parser_group.add_argument('--svhn', action='store_true',
                                   help='Perform SVHN analysis')
    svhn_parser_group.add_argument('--transfer_explanation', action='store_true',
                                   help='Produce transfer explanation with MNIST')

    args = parser.parse_args()
    model, train_loader, test_loader, results_root, dataset = load(
        args.config_file, device)
    model.eval()

    if args.print_test_accuracy:
        test_routine(-1, model, test_loader, device)

    if args.compute_latent:
        compute_latent(model, train_loader, results_root, device)

    if args.create_nn_tree:
        create_trees(results_root)

    if args.compute_feature_attribution:
        loader = test_loader
        if args.compute_on_train:
            loader = train_loader
        compute_feature_attributions(
            model, loader, results_root, args.methods, dataset, device)

    if args.cub:
        if args.sample_id == -1:
            args.sample_id = np.random.randint(0, len(test_loader.dataset))

        if args.create_attribution_image:
            attribute_image(model, test_loader, args.sample_id,
                            device, results_root)

        if args.create_attribution_comparison:
            attribute_image_comparison(
                model, test_loader, args.sample_id, device, results_root)

        if args.top_k_analysis:
            analyze_top_patches(model, test_loader, args.sample_id, device, results_root,
                                args.top_k, args.top_k_vis)

        if args.create_correspondence:
            correspondence_images(model, test_loader, train_loader,
                                  args.sample_id, device, results_root, top_k=args.top_k)

        if args.create_counterfactual:
            if args.target_class == -1:
                args.target_class = np.random.randint(
                    0, len(test_loader.dataset.label_map))
            create_counterfactual(model, test_loader, train_loader, args.sample_id, device, results_root, top_k=args.top_k,
                                  target_class=args.target_class, consistent=args.consistent)

    if args.svhn:
        if args.sample_id == -1:
            args.sample_id = np.random.randint(0, len(test_loader.dataset))
        if args.transfer_explanation:
            transfer_explanation(
                model, test_loader, args.sample_id, device, results_root, top_k=args.top_k)
