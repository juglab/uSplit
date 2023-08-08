import glob
import os
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn

from usplit.config_utils import get_updated_config
from usplit.core.loss_type import LossType
from usplit.core.model_type import ModelType
from usplit.nets.brave_net import BraveNetPL
from usplit.nets.lvae import LadderVAE
from usplit.nets.unet import UNet


def create_model(config, data_mean, data_std):
    if config.model.model_type == ModelType.LadderVae:
        model = LadderVAE(data_mean, data_std, config)
    elif config.model.model_type == ModelType.UNet:
        model = UNet(data_mean, data_std, config)
    elif config.model.model_type == ModelType.BraveNet:
        model = BraveNetPL(data_mean, data_std, config)
    else:
        raise Exception('Invalid model type:', config.model.model_type)
    return model


def get_best_checkpoint(ckpt_dir):
    output = []
    for filename in glob.glob(ckpt_dir + "/*_best.ckpt"):
        output.append(filename)
    assert len(output) == 1, '\n'.join(output)
    return output[0]


def load_model_checkpoint(ckpt_dir: str,
                          data_mean: float,
                          data_std: float,
                          config=None,
                          model=None) -> pl.LightningModule:
    """
    It loads the model from the checkpoint directory
    """
    import ml_collections  # Needed due to loading in pickle
    if model is None:
        # load config, if the config is not provided
        if config is None:
            with open(os.path.join(ckpt_dir, 'config.pkl'), 'rb') as f:
                config = pickle.load(f)

        config = get_updated_config(config)
        model = create_model(config, data_mean, data_std)
    ckpt_fpath = get_best_checkpoint(ckpt_dir)
    checkpoint = torch.load(ckpt_fpath)
    _ = model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model from ckpt dir', ckpt_dir, f' at epoch:{checkpoint["epoch"]}')
    return model
