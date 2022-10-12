import argparse
import datetime, os
from kme.tools.config import load_config, TRAINING_KEY, save_config, CKPT_KEY, MODEL_KEY


def create_folder_and_save_config(config, root):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    folder = os.path.join(root, timestamp)
    config[CKPT_KEY] = os.path.abspath(folder)
    config_fpath = os.path.join(folder, 'config.json')
    os.makedirs(folder)
    save_config(config, config_fpath)
    return os.path.abspath(config_fpath)


def create_experiments(experiments_root: str, sample_config: str,
                       train_par_search: bool = True, arch_par_search: bool = True):
    config = load_config(sample_config)
    experiments = []

    if train_par_search:
        for bsize in [64]:
            config[TRAINING_KEY]['batch_size'] = bsize
            for lr in [0.001, 0.0001]:
                config[TRAINING_KEY]['optimizer_params']['lr'] = lr
                for optimizer in ['Adam', 'SGD']:
                    config[TRAINING_KEY]['optimizer'] = optimizer
                    for wd in [0.0005, 0.001, 0.0001]: # 1
                        config[TRAINING_KEY]['optimizer_params']['weight_decay'] = wd
                        for scheduler in ['StepLR', 'CosineAnnealingLR']: # 2
                            config[TRAINING_KEY]['scheduler'] = scheduler
                            config[TRAINING_KEY]['scheduler_params'] = {}
                            for momentum in [0.8, 0.9, 0.]: # 2
                                if optimizer == 'SGD':
                                    config[TRAINING_KEY]['optimizer_params']['momentum'] = momentum
                                else:
                                    config[TRAINING_KEY]['optimizer_params'].pop('momentum', None)
                                for gamma in [0.2, 0.5]: # 2
                                    if scheduler == 'StepLR':
                                        config[TRAINING_KEY]['scheduler_params']['step_size'] = 20
                                        config[TRAINING_KEY]['scheduler_params']['gamma'] = gamma
                                        experiments.append(create_folder_and_save_config(config, experiments_root))
                                    elif scheduler == 'CosineAnnealingLR':
                                        config[TRAINING_KEY]['scheduler_params']['T_max'] = 10
                                        config[TRAINING_KEY]['scheduler_params'].pop('gamma', None)
                                        experiments.append(create_folder_and_save_config(config, experiments_root))
                                        break
                                    elif scheduler == 'ExponentialLR':
                                        config[TRAINING_KEY]['scheduler_params']['gamma'] = 0.99
                                        config[TRAINING_KEY]['scheduler_params'].pop('step_size', None)
                                        config[TRAINING_KEY]['scheduler_params'].pop('T_max', None)
                                        break
                                if optimizer != 'SGD':
                                    break

    if arch_par_search:
        for feat_net in ['CIFARNoPosition']:
            config[MODEL_KEY]['feature_net'] = feat_net
            for class_net in ['CIFARHomo3LatentClassifier']:
                config[MODEL_KEY]['classifier'] = class_net
                for latent_dim in [32, 128, 512]:
                    config[MODEL_KEY]['feature_net_args']['latent_dim'] = latent_dim
                    config[MODEL_KEY]['classifier_args']['latent_dim'] = latent_dim
                    for n_filters in [2000]:
                        config[MODEL_KEY]['feature_net_args']['patcher_kwargs']['n_filters'] = n_filters
                        for padding in [2]:
                            config[MODEL_KEY]['feature_net_args']['patcher_kwargs']['padding'] = padding
                            if 'Position' in feat_net:
                                config[MODEL_KEY]['feature_net_args'].pop('pos_emb_dim', None)
                                for patcher in ['CIFARCNNPatcher', 'CIFAR3CNNPatcherFixed']:
                                    config[MODEL_KEY]['feature_net_args']['patcher'] = patcher
                                    if patcher == 'CIFARCNNPatcher':
                                        for patch_size in [4, 8]:
                                            config[MODEL_KEY]['feature_net_args']['patcher_kwargs']['patch_size'] = patch_size
                                            for stride in [2, 3, 4]:
                                                config[MODEL_KEY]['feature_net_args']['patcher_kwargs']['stride'] = stride
                                                experiments.append(create_folder_and_save_config(config, experiments_root))
                                    else:
                                        config[MODEL_KEY]['feature_net_args']['patcher_kwargs'].pop('patch_size', None)
                                        config[MODEL_KEY]['feature_net_args']['patcher_kwargs'].pop('stride', None)
                                        config[MODEL_KEY]['feature_net_args']['patcher_kwargs'].pop('batch_norm', None)
                                        experiments.append(create_folder_and_save_config(config, experiments_root))
                            else:
                                config[MODEL_KEY]['feature_net_args'].pop('patcher', None)
                                for pos_emb_dim in [128]:
                                    config[MODEL_KEY]['feature_net_args']['pos_emb_dim'] = pos_emb_dim
                                    if '3Patcher' in feat_net:
                                        config[MODEL_KEY]['feature_net_args']['patcher_kwargs'].pop('patch_size', None)
                                        config[MODEL_KEY]['feature_net_args']['patcher_kwargs'].pop('stride', None)
                                        config[MODEL_KEY]['feature_net_args']['patcher_kwargs'].pop('batch_norm', None)
                                        experiments.append(create_folder_and_save_config(config, experiments_root))
                                    else:
                                        for patch_size in [2, 4, 8]:
                                            config[MODEL_KEY]['feature_net_args']['patcher_kwargs']['patch_size'] = patch_size
                                            for stride in [1, 2]:
                                                config[MODEL_KEY]['feature_net_args']['patcher_kwargs']['stride'] = stride
                                                experiments.append(create_folder_and_save_config(config, experiments_root))

    return experiments


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grid Search Experiments',
                                     description='Create different experiments with different hyper-parameters.')
    parser.add_argument('--exp_root', type=str, help='Path to the root where to store all the experiments.')
    parser.add_argument('--sample_config', type=str, help='Config to use as reference.')
    parser.add_argument('--train_params', action='store_true', help='Grid Search over train parameters.')
    parser.add_argument('--arch_params', action='store_true', help='Grid Search over architecture parameters.')

    args = parser.parse_args()
    configs2run = create_experiments(args.exp_root, args.sample_config, args.train_params, args.arch_params)
