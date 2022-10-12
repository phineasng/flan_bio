import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.optim import *
from kme.tools.config import load_config, save_config, DATASET_KEY, TRAINING_KEY
import json
import pandas as pd
from pytoda.proteins import ProteinLanguage
from bin.load_model import load_kme_model


def get_text_and_attr(kme_net, tcr_seq, epit_seq, epit_padlen, device):

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
    # tensor with dim: [seq_length, 2*latent_dim]

    features = kme_net.sample_representation(
        [epit_seq.to(device), tcr_seq.to(device)])

    features_epit = features[0, :epit_padlen, :]
    features_tcr = features[0, epit_padlen:, :]

    # Compute the norms of Features
    attr_tcr = torch.norm(features_tcr, dim=1, p=2).detach().cpu().numpy()
    attr_epit = torch.norm(features_epit, dim=1, p=2).detach().cpu().numpy()

    # Rescaling values to [0,1]: (x - min) / (max - min)
    vmax = max(np.max(attr_tcr), np.max(attr_epit))
    vmin = min(np.min(attr_tcr), np.min(attr_epit))
    attr_tcr = (attr_tcr - vmin)/(vmax - vmin)
    attr_epit = (attr_epit - vmin)/(vmax - vmin)

    return tcr_text, attr_tcr, epit_text, attr_epit


def pick_features_and_classify(kme_net, epit_seq, tcr_seq, no_features, device):

    features = kme_net.sample_representation(
        [epit_seq.to(device), tcr_seq.to(device)])
    picked_features = torch.tensor([], device=device)
    ind_picked_features = torch.sort(torch.norm(
        features, dim=-1, p=2), descending=True).indices[:, :no_features].to(device)
    for sample in range(len(ind_picked_features)):
        picked_features = torch.cat(
            (picked_features, features[sample][ind_picked_features[sample]].unsqueeze(0)), dim=0)

    classifier = kme_net._classifier
    feature_sum = torch.sum(picked_features, 1)
    out = classifier(feature_sum)
    out = torch.softmax(out, dim=1)

    return ind_picked_features, picked_features, out


def total_aa_number(config_path, test=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kme_net, train_loader, test_loader = load_kme_model(config_path, device)
    if test:
        loader = test_loader
    else:
        loader = train_loader

    counter_tcr_total = [0]*len(ProteinLanguage().token_to_index.keys())
    counter_epit_total = [0]*len(ProteinLanguage().token_to_index.keys())
    aa_labels = ProteinLanguage().index_to_token.values()

    for samp in loader.dataset:
        tcr_seq = samp[1]
        epit_seq = samp[0]
        for aa in tcr_seq:
            counter_tcr_total[int(aa)] += 1
        for aa in epit_seq:
            counter_epit_total[int(aa)] += 1

    df_tcr = pd.Series(counter_tcr_total, list(aa_labels))
    df_epit = pd.Series(counter_epit_total, list(aa_labels))
    return df_tcr, df_epit


def top_picked_aminoacids(config_path, no_features=10, normalize=True, test=True):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kme_net, train_loader, test_loader = load_kme_model(config_path, device)
    if test:
        loader = test_loader
    else:
        loader = train_loader

    config = load_config(config_path)
    dataset_params = config[DATASET_KEY]

    p_filepath = dataset_params['dataset_args']['params_filepath']
    params = {}
    with open(p_filepath) as fp:
        params.update(json.load(fp))

    epit_padlen = params.get('ligand_padding_length')
    protein_language = ProteinLanguage()

    counter_tcr = pd.DataFrame(
        {key: [0] for key in (protein_language.token_to_index.keys())})
    counter_epit = pd.DataFrame(
        {key: [0] for key in (protein_language.token_to_index.keys())})

    for samp in loader.dataset:
        epit_seq = samp[0].unsqueeze(0)
        tcr_seq = samp[1].unsqueeze(0)
        label = samp[2]
        label = int(label.tolist()[0])

        tcr_text, attr_tcr, epit_text, attr_epit = get_text_and_attr(
            kme_net, tcr_seq, epit_seq, epit_padlen, device)
        ind_picked_features, picked_features, out = pick_features_and_classify(
            kme_net, epit_seq, tcr_seq, no_features, device)

        for j in ind_picked_features.squeeze().tolist():
            if j < len(epit_text):
                counter_epit[epit_text[j]] += 1
            else:
                counter_tcr[tcr_text[j-len(epit_text)]] += 1

    # NORMALIZE COUNTS WITH RESPECT TO THE TOTAL NUMBER OF AA
    if normalize:
        total_tcr, total_epit = total_aa_number(config_path, test)
        total_tcr[total_tcr == 0] = 1
        total_epit[total_epit == 0] = 1
        counter_tcr = counter_tcr/total_tcr
        counter_epit = counter_epit/total_epit

    return counter_epit, counter_tcr


def plot_aa(counter_epit, counter_tcr):
    fig, ax = plt.subplots()

    ax.bar(counter_epit.columns[5:-2],
           counter_epit.values[0][5:-2], label='Epitope')
    ax.bar(counter_tcr.columns[5:-2], counter_tcr.values[0][5:-2], label='TCR')

    ax.legend()
    plt.show()


def example_based_int(config_path, id=7, test=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, train_loader, test_loader = load_kme_model(config_path, device)
    if test:
        loader = test_loader
    else:
        loader = train_loader

    epit_seq = loader.dataset[id][0].unsqueeze(0)
    tcr_seq = loader.dataset[id][1].unsqueeze(0)

    x_star = [epit_seq, tcr_seq]
    star_features = model.sample_representation(x_star).detach()
    sum_star_features = torch.sum(star_features, dim=1)

    distances = []

    for samp in loader.dataset:
        epit_seq = samp[0].unsqueeze(0)
        tcr_seq = samp[1].unsqueeze(0)
        inp = [epit_seq, tcr_seq]

        inp_features = model.sample_representation(inp).detach()
        sum_inp_features = torch.sum(inp_features, dim=1)
        total_dist = torch.cdist(
            sum_inp_features, sum_star_features, p=2).item()
        distances.append(total_dist)

    nearest_neigh_ind = torch.sort(torch.tensor(
        distances, device=device), descending=True).indices[:3].tolist()

    return nearest_neigh_ind
