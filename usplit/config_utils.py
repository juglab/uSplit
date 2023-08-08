"""
Utility files for configs
 1. Take the diff between two configs.
"""
import os
import pickle

import ml_collections
from usplit.core.loss_type import LossType


def load_config(config_fpath):
    if os.path.isdir(config_fpath):
        config_fpath = os.path.join(config_fpath, 'config.pkl')
    else:
        assert config_fpath[-4:] == '.pkl', f'{config_fpath} is not a pickle file. Aborting'
    with open(config_fpath, 'rb') as f:
        config = pickle.load(f)
    return get_updated_config(config)


def get_updated_config(config):
    """
    It makes sure that older versions of the config also run with current settings.
    """
    frozen_dict = isinstance(config, ml_collections.FrozenConfigDict)
    if frozen_dict:
        config = ml_collections.ConfigDict(config)

    with config.unlocked():
        pass

    if frozen_dict:
        return ml_collections.FrozenConfigDict(config)
    else:
        return config
