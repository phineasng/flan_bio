import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.optim import *
from kme.data.utils import get_loaders, get_loader_dataset
from kme.models.utils import build_kme_net, count_parameters
from kme.tools.checkpoint import load_checkpoint, save_checkpoint
from kme.tools.config import load_config, save_config, DATASET_KEY, TRAINING_KEY, MODEL_KEY, CKPT_KEY
from pytoda.proteins import ProteinLanguage
import argparse
import warnings
import torch.optim as optim
import json


def interpret_sentence(config_path, ind=0, test=False):

    warnings.simplefilter(action='ignore', category=FutureWarning)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)

    dataset_params = config[DATASET_KEY]
    training_params = config[TRAINING_KEY]
    model_params = config[MODEL_KEY]
    ckpt_root = config[CKPT_KEY]

    BATCH_SIZE = training_params['batch_size']
    OPTIMIZER = training_params['optimizer']
    OPTIMIZER_PARAMS = training_params['optimizer_params']
    SCHEDULER = training_params['scheduler']

    p_filepath = dataset_params['dataset_args']['params_filepath']
    params = {}
    with open(p_filepath) as fp:
        params.update(json.load(fp))

    epit_padlen = params.get('ligand_padding_length')

    train_loader, valid_loader, test_loader = get_loaders(
        **dataset_params, batch_size=BATCH_SIZE, device=device)
    if test:
        loader = test_loader
    else:
        loader = train_loader

    if 'use_mean' not in model_params:
        model_params['use_mean'] = False

    if dataset_params['dataset'] == 'tcr':
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
        ckpt_root+'/checkpoint.pth', kme_net, optimizer, scheduler, device=device)

    # PREDICT
    epit_seq = loader.dataset[ind][0].unsqueeze(0)
    tcr_seq = loader.dataset[ind][1].unsqueeze(0)
    label = loader.dataset[ind][2]

    label = int(label.tolist()[0])
    output = kme_net([epit_seq.to(device), tcr_seq.to(device)])
    prob = torch.softmax(output, dim=1)

    # Text Sequeces
    def transform_text(sequence, language):
        ind2str = language.index_to_token
        text = [ind2str[i] for i in sequence]
        return text

    protein_language = ProteinLanguage()
    tcr_text = transform_text(tcr_seq.squeeze().detach().cpu(
    ).numpy(), language=protein_language)
    epit_text = transform_text(epit_seq.squeeze().detach().cpu(
    ).numpy(), language=protein_language)

    # Calculate Features

    # Feat1, remove the pads
    if config['model']['feature_net'] == "TE_FeatureNet1":
        tcr_pad_ind = [i for i in range(
            len(tcr_text)) if tcr_text[i] == '<PAD>'][-1]
        epit_pad_ind = [i for i in range(
            len(epit_text)) if epit_text[i] == '<PAD>'][-1]
        tcr_text = [tcr_text[i] for i in range(
            len(tcr_text)) if tcr_text[i] != '<PAD>']
        epit_text = [epit_text[i] for i in range(
            len(epit_text)) if epit_text[i] != '<PAD>']

    features = kme_net.sample_representation(
        [epit_seq.to(device), tcr_seq.to(device)])

    features_epit = features[0, :epit_padlen, :]
    features_tcr = features[0, epit_padlen:, :]

    # Compute the norms of Features
    attr_tcr = torch.norm(features_tcr, dim=1, p=2).detach().cpu().numpy()
    attr_epit = torch.norm(features_epit, dim=1, p=2).detach().cpu().numpy()

    if config['model']['feature_net'] == "TE_FeatureNet1":
        attr_tcr = attr_tcr[tcr_pad_ind:]
        attr_epit = attr_epit[epit_pad_ind:]

    # Rescaling values to [0,1]: (x - min) / (max - min)
    vmax = max(np.max(attr_tcr[1:]), np.max(attr_epit[1:]))
    vmin = min(np.min(attr_tcr[1:]), np.min(attr_epit[1:]))
    attr_tcr = (attr_tcr - vmin)/(vmax - vmin)
    attr_epit = (attr_epit - vmin)/(vmax - vmin)

    # PLOTS
    fig, ax = amino_plot(tcr_text, attr_tcr, epit_text,
                         attr_epit, prob, label)

    # fig.savefig(ckpt_root + '/sample'+str(ind)+'_features_figure.png',
    #            dpi=300, bbox_inches='tight')


def amino_plot(tcr_text, attr_tcr, epit_text, attr_epit, prob, label):

    cmap = mpl.cm.Reds

    COL_UNIT = 2
    ROW_UNIT = 0.5
    MAX_COLS = 30
    MAX_ROWS = 4 + (len(tcr_text) + len(epit_text))//MAX_COLS

    fig = plt.figure(figsize=[MAX_COLS*COL_UNIT*2, MAX_ROWS*ROW_UNIT*2])
    ax = fig.add_axes([0, 0, 1, 1])

    total_pred = torch.argmax(prob).tolist()
    total_binding = round(prob.tolist()[0][total_pred], 3)

    row_shift = 0.

    ax.text(0., 0.3, "TCR", fontsize=13, weight='bold', fontfamily='serif')
    row_shift += ROW_UNIT

    def render_sequence(seq, attr, row_shift):
        col_shift = 0

        for i in range(len(seq)):
            color = cmap.__call__(attr[i])
            ax.add_patch(mpatches.Rectangle(
                (col_shift, row_shift), COL_UNIT, 0.3, color=color))
            if '<' in seq[i]:
                FS = 5
            else:
                FS = 10

            ax.text(0.1 + col_shift, row_shift + 0.2,
                    seq[i], fontsize=FS, weight='bold', fontfamily='serif')

            col_shift += COL_UNIT

            if (i + 1) % MAX_COLS == 0:
                col_shift = 0
                row_shift += ROW_UNIT
            elif i == len(seq) - 1:
                row_shift += ROW_UNIT

        row_shift += ROW_UNIT
        return row_shift

    row_shift = render_sequence(
        tcr_text, attr_tcr, row_shift)

    ax.text(0., row_shift, "EPITOPE", fontsize=13,
            weight='bold', fontfamily='serif')
    row_shift += ROW_UNIT - 0.35
    row_shift = render_sequence(
        epit_text, attr_epit, row_shift)

    ax.text(0., row_shift + 0.25,
            f'PREDICTION: {total_pred} with probability: {total_binding} ', fontsize=13, weight='bold', fontfamily='serif')
    ax.text(0., row_shift + 0.5,
            f'REAL LABEL: {label} ', fontsize=13, weight='bold', fontfamily='serif')

    ax.set_xlim(0 - 0.5, MAX_COLS*COL_UNIT + 0.5)
    ax.set_ylim(row_shift + 1, 0.)

    ax.axis('off')
    plt.show()
    return fig, ax
