
import csv
import gzip
import os
import scipy.io
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import random


def check_imbalance(classes):

    counts = []
    for c in set(classes):
        counts.append(classes.count(c))

    return counts


def load_matrix_data(path_matrix, path_classes, path_barcodes, path_picked_genes):

    # read in MEX format matrix as table
    mat = scipy.io.mmread(path_matrix)
    mat = mat.T
    picked_genes_names = pd.read_csv(
        path_picked_genes, delimiter=' ', header=None, squeeze=True).values.tolist()

    mclasses = pd.read_csv(path_classes, delimiter=',')
    classes = mclasses['Cluster'].tolist()
    # Make classes start from 0
    classes = [c-1 for c in classes]
    picked_barcodes = mclasses['Barcode'].tolist()

    barcodes_path = os.path.join(path_barcodes)
    barcodes = [row[0] for row in csv.reader(
        gzip.open(barcodes_path, mode="rt"), delimiter="\t")]
    ind_picked_barcodes = [i for i in range(
        len(barcodes)) if barcodes[i] in picked_barcodes]

    # transform table to pandas dataframe and label rows and columns
    matrix = pd.DataFrame.sparse.from_spmatrix(mat, index=picked_genes_names)
    matrix = matrix.iloc[:, ind_picked_barcodes]
    matrix.columns = picked_barcodes

    return matrix, classes


def get_single_cell_dataset(device, path_matrix, path_classes, path_barcodes, path_picked_genes, oversample=True):

    random.seed(1)

    matrix, classes = load_matrix_data(
        path_matrix, path_classes, path_barcodes, path_picked_genes)

    if oversample:
        X = torch.tensor(matrix.T.values)
        y = torch.tensor(classes).unsqueeze(-1)
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)

        df = pd.DataFrame(X)
        df['classes'] = y
    else:
        df = matrix.T
        df['classes'] = classes

    train_set, test_set = train_test_split(
        df, test_size=0.2, stratify=df['classes'])

    train_set = torch.tensor(
        train_set.values, dtype=torch.double, device=device)
    test_set = torch.tensor(test_set.values, dtype=torch.double, device=device)

    return train_set, test_set
