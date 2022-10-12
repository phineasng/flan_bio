
from bin.load_model import load_kme_model
import torch
from kme.tools.config import load_config
import pandas as pd
from sklearn_extra.cluster import KMedoids
import numpy as np


def load_gene_names(path_picked_genes):
    df = pd.read_csv(path_picked_genes, delimiter=' ',
                     header=None, squeeze=True)
    gene_names = df.values.tolist()
    return gene_names


def get_cell_weights(config_path, test=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kme_net, train_loader, test_loader = load_kme_model(config_path, device)

    config = load_config(config_path)
    path_picked_genes = config["dataset_params"]["dataset_args"]["path_picked_genes"]

    all_gene_names = load_gene_names(path_picked_genes)

    markers = ['CD3D', 'CD8A', 'CD8B', 'CCR10', 'TNFRSF18',
               'CD4', 'ID3', 'CD79A', 'PF4', 'NKG7',
               'S100A8', 'S100A9']
    # high_exp_genes = ['MALAT1', 'B2M', 'ACTB', 'FTL', 'FTH1', 'LTB']
    high_var_genes = ['MALAT1', 'B2M', 'FTL', 'GNLY', 'CD74', 'ACTB']
    markers = markers + high_var_genes
    # remove duplicates
    markers = list(dict.fromkeys(markers))
    markers_id = [all_gene_names.index(marker) for marker in markers]

    if test:
        loader = test_loader
    else:
        loader = train_loader

    marker_weights = []
    labels = []

    for samp in loader.dataset:
        cell = samp[:-1].unsqueeze(0)
        labels.append(int(samp[-1]))
        features = kme_net.sample_representation(cell)
        features_norms = torch.norm(features, dim=-1, p=2).squeeze()
        marker_weights.append(features_norms[markers_id].half().tolist())

    df = pd.DataFrame(marker_weights, columns=markers)

    return df, labels, markers


def algor_inter(config_path, id=0, test=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, train_loader, test_loader = load_kme_model(config_path, device)
    if test:
        loader = test_loader
    else:
        loader = train_loader

    config = load_config(config_path)
    path_picked_genes = config["dataset_params"]["dataset_args"]["path_picked_genes"]

    cell = loader.dataset[id][:-1].unsqueeze(0)

    features = model._feat_net.process_samples(cell)
    feature_norms = torch.norm(features, dim=2, p=2)
    picked_features_id = torch.sort(
        feature_norms, descending=True).indices.squeeze().tolist()[:3]

    all_gene_names = load_gene_names(path_picked_genes)
    picked_names = [all_gene_names[i] for i in picked_features_id]

    full_out = model(cell).detach().squeeze()
    full_out = torch.softmax(full_out, 0)
    prediction = [int(torch.argmax(full_out))]

    temp = torch.sort(torch.softmax(full_out, 0), descending=True)
    temp = {str(int(temp.indices[0])): round(temp.values[0].tolist(), 3), str(int(temp.indices[1])): round(
            temp.values[1].tolist(), 3), str(int(temp.indices[2])): round(temp.values[2].tolist(), 3)}

    top_3_prob = [temp]

    # Find the top 3 marker genes
    picked_features_id_all = torch.sort(
        feature_norms, descending=True).indices.squeeze().tolist()
    markers = ['CD3D', 'CD8A', 'CD8B', 'CCR10', 'TNFRSF18',
               'CD4', 'ID3', 'CD79A', 'PF4', 'NKG7',
               'S100A8', 'S100A9']
    markers_id = [all_gene_names.index(marker) for marker in markers]
    sort_markers = []
    for m in markers_id:
        sort_markers.append(picked_features_id_all.index(m))

    sort_markers = torch.tensor(sort_markers)
    top_markers_positions = torch.sort(sort_markers).values[:3]
    top_markers_id = [picked_features_id_all[k] for k in top_markers_positions]
    top_markers_names = [all_gene_names[k] for k in top_markers_id]

    picked_features_id = picked_features_id + top_markers_id
    picked_names = picked_names + top_markers_names

    for k in picked_features_id:

        picked_feature = features[:, k, :]
        classifier = model._classifier
        out = classifier(picked_feature).detach().squeeze()

        prediction.append(torch.argmax(torch.softmax(out, 0)).tolist())

        temp = torch.sort(torch.softmax(out, 0), descending=True)
        temp = {str(int(temp.indices[0])): round(temp.values[0].tolist(), 3), str(int(temp.indices[1])): round(
            temp.values[1].tolist(), 3), str(int(temp.indices[2])): round(temp.values[2].tolist(), 3)}
        top_3_prob.append(temp)

    df = pd.DataFrame(top_3_prob, columns=[str(
        i) for i in range(7)], index=["full"]+picked_names)
    df.replace(np.nan, '-', regex=True)

    return df


def example_based(config_path, no_clusters=5, test=False):
    # This might take a while to run

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kme_net, train_loader, test_loader = load_kme_model(config_path, device)
    if test:
        loader = test_loader
    else:
        loader = train_loader

    # SEPERATE THE TEST DATA IN no_groups GROUPS:
    for i in range(7):
        locals()['Group' + str(i)] = []

    medoit_data = {}

    for i, samp in enumerate(loader.dataset):
        samp = samp
        cell = samp[:-1].unsqueeze(0)
        out = kme_net(cell).detach()
        pred = torch.argmax(torch.softmax(out, dim=1)).tolist()
        group = locals()['Group' + str(pred)]
        group.append(cell)

    medoits_model = KMedoids(no_clusters, method='pam')

    def find_class_centers(group):

        group_features = torch.tensor([], device=device)
        for ex in group:
            x_star_repr = kme_net.sample_representation(ex).detach()
            group_features = torch.cat((group_features, x_star_repr), dim=0)

        group_representation = torch.sum(
            group_features, dim=1)  # aggregated latent space
        df = group_representation.detach().cpu().numpy()
        medoits_model.fit(df)
        medoits_centers = [group[i] for i in medoits_model.medoid_indices_]

        return medoits_centers

    medoit_data = {}
    for i in range(7):
        medoit_data['Group' +
                    str(i)] = find_class_centers(locals()['Group' + str(i)])

    return medoit_data
