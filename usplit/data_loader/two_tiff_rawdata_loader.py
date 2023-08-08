import os

import numpy as np

from usplit.core.data_split_type import DataSplitType, get_datasplit_tuples
from usplit.core.tiff_reader import load_tiff


def get_train_val_data(dirname, data_config, datasplit_type, val_fraction, test_fraction):
    # actin-60x-noise2-highsnr.tif  mito-60x-noise2-highsnr.tif
    fpath1 = os.path.join(dirname, data_config.ch1_fname)
    fpath2 = os.path.join(dirname, data_config.ch2_fname)

    print(f'Loading from {dirname} Channel1: '
          f'{fpath1},{fpath2}, Mode:{DataSplitType.name(datasplit_type)}')

    data1 = load_tiff(fpath1)[..., None]
    data2 = load_tiff(fpath2)[..., None]

    data = np.concatenate([data1, data2], axis=3)
    if data_config.get('enable_poisson_noise', True):
        data = np.random.poisson(data)

    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data), starting_test=True)
    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)
