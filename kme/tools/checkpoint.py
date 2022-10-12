import torch
import os
from kme.data.text_classification import DATASETS as TEXT_DATASETS, MAX_SENTENCE_LEN
from torch.optim import *
import torch


CHECKPOINT_FNAME = 'checkpoint.pth'
MODEL_STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
SCHEDULER_STATE_DICT_KEY = 'scheduler_state_dict'
EPOCH_KEY = 'epoch'


def save_checkpoint(root, model, optimizer, scheduler, epoch):
    state = {
        MODEL_STATE_DICT_KEY: model.state_dict(),
        OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
        EPOCH_KEY: epoch,
    }
    if scheduler is not None:
        state[SCHEDULER_STATE_DICT_KEY] = scheduler.state_dict()
    torch.save(state, os.path.join(root, CHECKPOINT_FNAME))


def load_checkpoint(root, model, optimizer, scheduler, device=None):
    ckpt_fpath = os.path.join(root, CHECKPOINT_FNAME)
    epoch = 0
    if os.path.exists(ckpt_fpath):
        state = torch.load(ckpt_fpath, map_location=device)
        model_state = {k.replace('module.', ''): v for k,
                       v in state[MODEL_STATE_DICT_KEY].items()}
        model.load_state_dict(model_state)
        optimizer.load_state_dict(state[OPTIMIZER_STATE_DICT_KEY])
        if SCHEDULER_STATE_DICT_KEY in state:
            scheduler.load_state_dict(state[SCHEDULER_STATE_DICT_KEY])
        epoch = state[EPOCH_KEY] + 1
    return model, optimizer, scheduler, epoch
