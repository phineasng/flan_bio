import json


# KEYS
MODEL_KEY = 'model'
TRAINING_KEY = 'train_params'
DATASET_KEY = 'dataset_params'
CKPT_KEY = 'checkpoint_root'


def load_config(fpath: str):
    with open(fpath, 'r') as infile:
        return json.load(infile)


def save_config(config: dict, fpath: str):
    with open(fpath, 'w') as outfile:
        json.dump(config, outfile)
