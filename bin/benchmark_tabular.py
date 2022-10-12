import numpy as np
import torch, os, shutil, pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from kme.data.utils import get_loaders
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier
from kme.models.utils import build_kme_net
from kme.tools.training import train_routine, test_routine
from copy import deepcopy
from kme.extern.senn.trainer import init_trainer
from sklearn.metrics import accuracy_score, roc_auc_score


N_EPOCHS = 10


def empty_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


BENCHMARK_DATASETS = ['heart',  'compas', 'adult', 'mammo']
# BENCHMARK_DATASETS = ['credit', ]
# BENCHMARK_DATASETS = ['bank', 'spambase', 'mammo'] #
# BENCHMARK_DATASETS = ['mushroom',]

SKLEARN_MODELS = {
    'logistic': lambda : LogisticRegression(),
    'tree_small': lambda : DecisionTreeClassifier(max_depth=10),
    'tree_big': lambda : DecisionTreeClassifier(),
    'ebm': lambda : ExplainableBoostingClassifier(),
    'random_forest': lambda : RandomForestClassifier(n_estimators=100),
    'xgboost': lambda : XGBClassifier(n_estimators=100, use_label_encoder=False)
}
DEEP_MODELS_ARGS = {
    'mlp': {
        "feature_net": "BaselineMLP1",
        "classifier": "Linear",
        "feature_net_args": {
            "latent_dim": 16,
            "n_feats": 11
        },
        "classifier_args": {
            "latent_dim": 16,
            "n_classes": 2
        }
    },
    'flan': {
        "feature_net": "SmallTabFeatNet2",
        "classifier": "SmallClassifier2",
        "feature_net_args": {
            "latent_dim": 16,
            "n_feats": 11
        },
        "classifier_args": {
            "latent_dim": 16,
            "n_classes": 2
        }
    }
}
SENN_CONFIG_BASE = {
    "train": True,
    "conceptizer": "IdentityConceptizer",
    "parameterizer": "LinearParameterizer",
    "hidden_sizes": [64, 128, 64, 32],
    "num_concepts": 0,
    "concept_names": [],
    "num_classes": 2,
    "dropout": 0.1,
    "aggregator": "SumAggregator",
    "lr": 5e-4,
    "epochs": N_EPOCHS,
    "robustness_loss": "compas_robustness_loss",
    "robust_reg": 0.0,
    "concept_reg": 0,
    "print_freq": 100,
    "exp_name": "benchmark",
    "batch_size" : 200,
    "sparsity_reg": 2e-5,
    "eval_freq" : 30,
    "manual_seed": 111
}


def update_model_params(n_feats, model):
    args = deepcopy(DEEP_MODELS_ARGS[model])
    args['feature_net_args']['n_feats'] = n_feats
    return args


DEEP_MODELS = {
    'mlp': lambda n_feats, device: build_kme_net(update_model_params(n_feats, 'mlp'), device=device),
    'flan': lambda n_feats, device: build_kme_net(update_model_params(n_feats, 'flan'), device=device)
}


def loader2array(loader):
    X = []
    Y = []
    for x, y in loader:
        X.append(x.detach().numpy())
        Y.append(y.detach().numpy())
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y


def train_sklearn(x, y, x_test, y_test):
    results = dict()
    for k, constructor in SKLEARN_MODELS.items():
        print('... ... Benchmarking {}'.format(k))
        if k == 'logistic':
            scaler = StandardScaler()
            xx = scaler.fit_transform(x)
            xx_test = scaler.transform(x_test)
        else:
            xx = x
            xx_test = x_test

        model = constructor()
        model.fit(xx, y)
        y_hat = model.predict_proba(xx_test)
        results[k] = roc_auc_score(y_test, y_hat[:, 1])
        print('{}] AUC = {}'.format(k, results[k]))
    return results


def train_senn(train_loader, val_loader, test_loader, n_feats, random_seed, device):
    print('... ... Benchmarking SENN')
    configs = deepcopy(SENN_CONFIG_BASE)
    configs['num_concepts'] = n_feats
    configs['concept_names'] = [str(i) for i in range(n_feats)]
    configs['exp_name'] = 'senn_benchmark_{}'.format(random_seed)
    configs['device'] = device
    configs['hidden_sizes'] = [n_feats, 64, 128, 128, 128, 128, 2*n_feats]
    configs['manual_seed'] = random_seed

    trainer = init_trainer(configs, train_loader, val_loader, test_loader)
    trainer._save = False
    empty_folder(trainer.checkpoint_dir)
    empty_folder(trainer.log_dir)
    trainer.run()
    return trainer.test()


def train_dnns(train_loader, val_loader, test_loader, n_feats, device):
    results = dict()
    for k, dnn in DEEP_MODELS.items():
        print('... ... Benchmarking {}'.format(k))
        model = dnn(n_feats, device)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
        for e in range(N_EPOCHS):
            train_routine(e, model, optimizer, train_loader, device, if_tqdm=False, if_print=False, norm_reg=0.)
        model.eval()
        #_, test_accuracy = test_routine(N_EPOCHS, model, test_loader, device)

        true_labels = []
        preds = []

        for x, y in test_loader:
            x = x.to(device)
            y = y.detach().cpu().numpy()

            true_labels.append(y)
            preds.append(torch.softmax(model(x), dim=1).detach().cpu().numpy())

        true_labels = np.concatenate(true_labels, axis=0)
        preds = np.concatenate(preds, axis=0)

        results[k] = roc_auc_score(true_labels, preds[:, 1])
        print('{}] AUC = {}'.format(k, results[k]))
    return results


def run_experiment(root, dataset, random_seed, device):
    results = dict()

    train_loader, valid_loader, test_loader = get_loaders(dataset, valid_split=0.1,
                                                          test_split=0.2, random_seed=random_seed,
                                                          dataroot=root)

    train_X, train_Y = loader2array(train_loader)
    test_X, test_Y = loader2array(test_loader)

    results.update(train_dnns(train_loader, valid_loader, test_loader,  n_feats=train_X.shape[1], device=device))
    results.update(train_sklearn(train_X, train_Y, test_X, test_Y))
    results['senn'] = train_senn(train_loader, valid_loader, test_loader, n_feats=train_X.shape[1],
                                 random_seed=random_seed, device=device)
    print('{}] AUC = {}'.format('senn', results['senn']))
    return results


if __name__ == '__main__':
    ROOT = '/home/phineas/Documents/repos/kme_net/results/benchmarking'
    now = datetime.now()
    RESULT_PATH = os.path.join(ROOT, 'tabular_benchmarking_{}.pkl'.format(datetime.timestamp(now)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    RANDOM_SEEDS = [1234, 156, 125, 2834, 542, 438, 683, 255, 986, 776]

    all_results = dict()

    for dataset in BENCHMARK_DATASETS:
        print('Running experiment for ...')
        print('... dataset {}'.format(dataset))

        curr_results = []

        for seed in RANDOM_SEEDS:
            print('... seed {}'.format(seed))
            res = run_experiment('/home/phineas/Documents/repos/kme_net/data', dataset, seed, device)
            curr_results.append(res)

        curr_dataset_results = {k: [dic[k] for dic in curr_results] for k in curr_results[0]}
        all_results[dataset] = curr_dataset_results

    with open(RESULT_PATH, 'wb') as out_file:
        pickle.dump(all_results, out_file)
    with open(RESULT_PATH + '.backup', 'wb') as out_file:
        pickle.dump(all_results, out_file)
