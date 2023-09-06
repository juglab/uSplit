import os
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

import zarr
from usplit.core.data_split_type import DataSplitType
from usplit.data_loader.multiscale_mc_tiff_dloader import MultiScaleTiffDloader
from usplit.data_loader.patch_index_manager import GridAlignement


class MultiScaleZarrDloader(MultiScaleTiffDloader):

    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 use_one_mu_std=None,
                 num_scales: int = None,
                 enable_random_cropping=False,
                 padding_kwargs: dict = None,
                 allow_generation: bool = False,
                 lowres_supervision=None,
                 max_val=None,
                 grid_alignment=GridAlignement.LeftTop,
                 overlapping_padding_kwargs=None):
        msg = 'To avoid whole data to get loaded in the current implementation, {} should be disabled'
        assert data_config.get("empty_patch_replacement_enabled",
                               False) is False, msg.format('empty_patch_replacement_enabled')
        assert data_config.get('background_quantile', 0.0) == 0.0, msg.format('background_quantile')
        self._config_max_val = data_config.get('max_val', None)
        self._datasplit_type = datasplit_type
        self._fpath = fpath
        self._channels = [data_config.channel_1, data_config.channel_2]
        self._lowres_supervision = lowres_supervision

        self.num_scales = num_scales
        assert self.num_scales is not None
        self._scaled_data = []
        assert isinstance(self.num_scales, int) and self.num_scales >= 1

        self._padding_kwargs = padding_kwargs  # mode=padding_mode, constant_values=constant_value
        if overlapping_padding_kwargs is not None:
            assert self._padding_kwargs == overlapping_padding_kwargs, 'During evaluation, overlapping_padding_kwargs should be same as padding_args. \
                It should be so since we just use overlapping_padding_kwargs when it is not None'

        else:
            overlapping_padding_kwargs = padding_kwargs

        # skip the parent class constructor and call parent's parent constructor.
        super(MultiScaleTiffDloader, self).__init__(data_config,
                                                    fpath,
                                                    datasplit_type=datasplit_type,
                                                    val_fraction=val_fraction,
                                                    test_fraction=test_fraction,
                                                    normalized_input=normalized_input,
                                                    enable_rotation_aug=enable_rotation_aug,
                                                    enable_random_cropping=enable_random_cropping,
                                                    use_one_mu_std=use_one_mu_std,
                                                    allow_generation=allow_generation,
                                                    max_val=max_val,
                                                    grid_alignment=grid_alignment,
                                                    overlapping_padding_kwargs=overlapping_padding_kwargs)
        assert isinstance(self._padding_kwargs, dict)
        assert 'mode' in self._padding_kwargs

    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None):
        for scale_idx in tqdm(range(0, self.num_scales)):
            basedir = self._fpath
            tiff_fpath = os.path.join(basedir, DataSplitType.name(self._datasplit_type), f'{scale_idx}.zarr')
            if not os.path.exists(tiff_fpath):
                raise FileNotFoundError(f'{tiff_fpath} does not exist. Exiting!')
            self._scaled_data.append(zarr.load(tiff_fpath))

        self._data = self._scaled_data[0]
        self.N = len(self._data)

    def set_max_val(self, max_val, datasplit_type):
        assert max_val is not None or self._config_max_val is not None, 'max_val should be precomputed and saved to config to avoid computation on of whole data'
        self.max_val = max_val if max_val is not None else self._config_max_val

    def upperclip_data(self):
        """
        TODO: Implement patchwise upperclip function so that it is not computed on the whole data.
        """
        return

    def _load_img(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx = index[0]

        loaded_imgs = [self._data[self.idx_manager.get_t(idx), ..., ch][None] for ch in self._channels]
        return tuple(loaded_imgs)

    def _load_scaled_img(self, scaled_index, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index
        imgs = [self._scaled_data[scaled_index][idx % self.N, ..., ch][None] for ch in self._channels]
        return imgs


if __name__ == '__main__':
    from usplit.configs.microscopy_multi_channel_lvae_config import get_config
    cfg = get_config()
    fpath = '/Users/ashesh.ashesh/Documents/Datasets/test_microscopy/downsampled_data'
    datasplit_type = DataSplitType.Train
    num_scales = 5
    padding_kwargs = {'mode': cfg.data.padding_mode}
    if 'padding_value' in cfg.data and cfg.data.padding_value is not None:
        padding_kwargs['constant_values'] = cfg.data.padding_value

    dset = MultiScaleZarrDloader(cfg.data,
                                 fpath,
                                 datasplit_type,
                                 val_fraction=0.1,
                                 test_fraction=0.1,
                                 normalized_input=True,
                                 enable_rotation_aug=False,
                                 use_one_mu_std=None,
                                 num_scales=num_scales,
                                 enable_random_cropping=False,
                                 max_val=None,
                                 overlapping_padding_kwargs=None,
                                 padding_kwargs=padding_kwargs)
    print('started\n')
    inp, tar = dset[0]
