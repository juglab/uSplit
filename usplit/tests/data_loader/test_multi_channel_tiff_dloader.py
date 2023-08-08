import numpy as np

from usplit.data_loader.multi_channel_train_val_data import _train_val_data


def test_train_val_data():
    nchannels = 20
    val_fraction = 0.2
    data = np.random.rand(60, 512, 256, nchannels)
    channel_1, channel_2 = np.random.choice(nchannels, size=2, replace=False)
    is_train = True
    train_data = _train_val_data(data, is_train, channel_1, channel_2, val_fraction=val_fraction)

    is_train = False
    val_data = _train_val_data(data, is_train, channel_1, channel_2, val_fraction=val_fraction)

    is_train = None
    total_data = _train_val_data(data, is_train, channel_1, channel_2, val_fraction=val_fraction)

    valN = 12
    trainN = 60 - valN
    assert np.abs(data[:trainN, :, :, [channel_1, channel_2]] - train_data).max() < 1e-6
    assert np.abs(data[trainN:, :, :, [channel_1, channel_2]] - val_data).max() < 1e-6
    assert np.abs(data[..., [channel_1, channel_2]] - total_data).max() < 1e-6
