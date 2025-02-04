"""
Here, the idea is to load the data from different data dtypes into a single interface.
"""
from typing import Union
import os
import zarr
from usplit.core.data_split_type import DataSplitType
from usplit.core.data_type import DataType
from usplit.data_loader.multi_channel_train_val_data import train_val_data as _load_tiff_train_val
from usplit.data_loader.sinosoid_dloader import train_val_data as _loadsinosoid
from usplit.data_loader.sinosoid_threecurve_dloader import train_val_data as _loadsinosoid3curve
from usplit.data_loader.two_tiff_rawdata_loader import get_train_val_data as _loadseparatetiff
# from usplit.data_loader.ht_lif24_rawdata import get_train_val_data as _load_htlif24_data
from usplit.core.tiff_reader import load_tiff

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
    if data_config.data_type ==DataType.OptiMEM100_014:
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
    elif data_config.data_type == DataType.CosemHela:
        fname = 'train_jrc_hela-3_bleedthrough_EGFP_Venus_R3.0_S4_D1_Ex100.0ms.tif'
        if datasplit_type  == DataSplitType.Val:
            fname = fname.replace('train', 'val')
        elif datasplit_type == DataSplitType.Test:
            fname = fname.replace('train', 'test')

        fpath = os.path.join(fpath,fname)
        return _load_tiff_train_val(fpath,data_config,DataSplitType.All)
    elif data_config.data_type == DataType.HTLIF24:
        fname = 'train_500ms_Ch_B-Ch_D-Ch_BD.tif'
        subdir ='train'
        if datasplit_type  == DataSplitType.Val:
            fname = fname.replace('train', 'val')
            subdir = 'val'
        elif datasplit_type == DataSplitType.Test:
            fname = fname.replace('train', 'test')
            subdir = 'test'

        fpath = os.path.join(fpath,subdir,fname)
        data = load_tiff(fpath)
        # skip the input channel
        data = data[...,:-1]
        print(f'Loaded HTLIF24 data from {fpath}', data.shape)
        return data
        # return _load_htlif24_data(fpath, data_config, datasplit_type, 
        #                           val_fraction=val_fraction, 
        #                           test_fraction=test_fraction)
    else:
        raise NotImplementedError(f'{DataType.name(data_config.data_type)} is not implemented')
