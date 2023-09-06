"""
Here, the idea is to load the data from different data dtypes into a single interface.
"""
from typing import Union

import zarr
from usplit.core.data_split_type import DataSplitType
from usplit.core.data_type import DataType
from usplit.data_loader.multi_channel_train_val_data import train_val_data as _load_tiff_train_val
from usplit.data_loader.sinosoid_dloader import train_val_data as _loadsinosoid
from usplit.data_loader.sinosoid_threecurve_dloader import train_val_data as _loadsinosoid3curve
from usplit.data_loader.two_tiff_rawdata_loader import get_train_val_data as _loadseparatetiff


def get_train_val_data(data_config,
                       fpath,
                       datasplit_type: DataSplitType,
                       val_fraction=None,
                       test_fraction=None,
                       allow_generation=None,
                       ignore_specific_datapoints=None):
    """
    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    assert isinstance(datasplit_type, int)
    if data_config.data_type == DataType.OptiMEM100_014:
        return _load_tiff_train_val(fpath,
                                    data_config,
                                    datasplit_type,
                                    val_fraction=val_fraction,
                                    test_fraction=test_fraction)
    elif data_config.data_type == DataType.CustomSinosoid:
        return _loadsinosoid(fpath,
                             data_config,
                             datasplit_type,
                             val_fraction=val_fraction,
                             test_fraction=test_fraction,
                             allow_generation=allow_generation)

    elif data_config.data_type == DataType.CustomSinosoidThreeCurve:
        return _loadsinosoid3curve(fpath,
                                   data_config,
                                   datasplit_type,
                                   val_fraction=val_fraction,
                                   test_fraction=test_fraction,
                                   allow_generation=allow_generation)

    elif data_config.data_type == DataType.SingleZarrData:
        return zarr.load(fpath)
    elif data_config.data_type == DataType.SeparateTiffData:
        return _loadseparatetiff(fpath, data_config, datasplit_type, val_fraction, test_fraction)
    else:
        raise NotImplementedError(f'{DataType.name(data_config.data_type)} is not implemented')
