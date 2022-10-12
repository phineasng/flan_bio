import pandas as pd
import anndata as ad
from bin.interpreting_sc.analyse_features import get_cell_weights
import scanpy as sc
from kme.tools.config import load_config
from bin.interpreting_sc.analyse_features import example_based, load_gene_names


def make_andata(config_path, markers_path):
    df, labels, markers = get_cell_weights(
        config_path, markers_path, test=True)

    labels_df = pd.Series([str(label) for label in labels.int().tolist()])
    labels_df = labels_df.astype('category')

    obs_meta = pd.DataFrame({
        'n_genes': df.shape[1],
        'labels': labels_df
    })

    adata = ad.AnnData(df, obs=obs_meta)

    label_names = ['CD8+', 'Megakaryocyte', 'CD4+',
                   'Naive CD4+, Myeloid', 'Naive CD8+', 'B-cell', 'NK, Regulatory, Act. CD8+']
    adata.rename_categories('labels', label_names)

    mp = sc.pl.MatrixPlot(adata, markers, groupby='labels', categories_order=['B-cell', 'Megakaryocyte', 'CD4+', 'Naive CD4+, Myeloid', 'CD8+', 'Naive CD8+', 'NK, Regulatory, Act. CD8+'],
                          standard_scale='var', cmap='Reds')
    mp.show()


def plot_heatmap(config_path, train_loader, group_ind):
    # This might take a while to run

    medoit_data = example_based(config_path, no_clusters=5, test=False)
    data = medoit_data['Group' + str(group_ind)]

    config = load_config(config_path)
    path_picked_genes = config["dataset_params"]["dataset_args"]["path_picked_genes"]
    all_gene_names = load_gene_names(path_picked_genes)

    # Keep only the "important" genes
    markers = ['CD3D', 'CD8A', 'CD8B', 'CCR10', 'TNFRSF18',
               'CD4', 'ID3', 'CD79A', 'PF4', 'NKG7',
               'S100A8', 'S100A9']
    high_var_genes = ['MALAT1', 'B2M', 'FTL', 'GNLY', 'CD74', 'ACTB', 'CCL5']
    markers = markers + high_var_genes
    markers_id = [all_gene_names.index(marker) for marker in markers]

    cells_small = []

    for cell in data:
        cell_small = [cell.squeeze().tolist()[i] for i in markers_id]
        cells_small.append(cell_small)

    label_names = ['CD8+', 'Megakaryocyte', 'CD4+',
                   'Naive CD4+, Myeloid', 'Naive CD8+', 'B-cell', 'NK, Regulatory, Act. CD8+']

    # For each gene subtract the minimum and divide each by its maximum value on the training dataset
    # Find vmin and vmax for each gene
    matrix = pd.DataFrame(0, index=range(
        len(train_loader.dataset)), columns=all_gene_names)
    for i, sample in enumerate(train_loader.dataset):
        matrix.loc[i, all_gene_names] = sample[:-1].tolist()
    vmax = matrix.max().iloc[markers_id]
    vmin = matrix.min().iloc[markers_id]

    df = pd.DataFrame(cells_small, columns=markers)
    df_norm = df.sub(vmin)
    df_norm = df.div(vmax)

    labels_df = pd.Series([str(label) for label in range(5)])
    labels_df = labels_df.astype('category')

    obs_meta = pd.DataFrame({
        'n_genes': df.shape[1],
        'labels': labels_df
    })
    adata = ad.AnnData(df_norm, obs=obs_meta)
    adata.var_names_make_unique()

    mp = sc.pl.MatrixPlot(adata, markers, groupby='labels', cmap='Reds')
    mp.show()
