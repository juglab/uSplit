"""
Multiscale data loader for zarr data, to be used with μSplit models
"""
import os
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

import zarr
from usplit.core.data_split_type import DataSplitType
from usplit.data_loader.lc_tiff_dloader import MultiScaleTiffDloader
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

    def get_channelordered_metadata(self, zarr_metadata):
        idx = [ ]
        for ch in self._channels:
            matching_rows = np.where(zarr_metadata[:,0] ==ch)[0]
            assert len(matching_rows) == 1
            idx.append(matching_rows[0])
        return zarr_metadata[idx]
    
    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None, test_img_arr=None):
        assert test_img_arr is None, 'test_img_arr is not supported for zarr data'
        for scale_idx in tqdm(range(0, self.num_scales)):
            basedir = self._fpath
            tiff_fpath = os.path.join(basedir, DataSplitType.name(self._datasplit_type), f'{scale_idx}.zarr')
            if not os.path.exists(tiff_fpath):
                raise FileNotFoundError(f'{tiff_fpath} does not exist. Exiting!')
            self._scaled_data.append(zarr.load(tiff_fpath))

        self._data = self._scaled_data[0]['raw']
        self.N = len(self._data)
    
    def compute_max_val(self):
        quantile_data = self.get_channelordered_metadata(self._scaled_data[0]['quantile'][:])
        msg = 'Quantile mismatch. Data has {} but this run expects {}.'
        msg += ' Run the multiscale_zarr_data_generator.py again with the correct quantile value or update the quantile value in the config for this run'
        assert np.all(quantile_data[:,1] == self._quantile), msg
        quantile = quantile_data[:,2]

        if self._channelwise_quantile:
            max_val_arr = quantile
            return max_val_arr
        else:
            # TODO: This is not the correct implementation for getting a single quantile.
            # However, this is not used in the current implementation.
            return np.mean(quantile)

    def compute_individual_mean_std(self):
        assert self._is_train is True, 'This is just allowed for training data'
        mean_data = self.get_channelordered_metadata(self._scaled_data[0]['mean'][:])[:,1]
        std_data = self.get_channelordered_metadata(self._scaled_data[0]['std'][:])[:,1]
        return mean_data[None, :, None, None], std_data[None, :, None, None]

    def compute_mean_std(self, allow_for_validation_data=False):
        """
        Note that we must compute this only for training data.
        """
        assert self._is_train is True, 'This is just allowed for training data'
        mean_data = self.get_channelordered_metadata(self._scaled_data[0]['mean'][:])[:,1]
        std_data = self.get_channelordered_metadata(self._scaled_data[0]['std'][:])[:,1]

        if self._use_one_mu_std is True:
            if self._input_is_sum:
                mean = np.sum(mean_data).reshape(1, 1, 1, 1)
                std = np.linalg.norm(std_data).reshape(1, 1, 1, 1)
            else:
                mean = np.mean(mean_data).reshape(1, 1, 1, 1)
                std = np.std(std_data).reshape(1, 1, 1, 1)
            
            mean = np.repeat(mean, 2, axis=1)
            std = np.repeat(std, 2, axis=1)

            if self._skip_normalization_using_mean:
                mean = np.zeros_like(mean)

            return mean, std

        elif self._use_one_mu_std is False:
            return self.compute_individual_mean_std()

        elif self._use_one_mu_std is None:
            return np.array([0.0, 0.0]).reshape(1, 2, 1, 1), np.array([1.0, 1.0]).reshape(1, 2, 1, 1)


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
        imgs = [self._scaled_data[scaled_index]['raw'][idx % self.N, ..., ch][None] for ch in self._channels]
        return imgs


if __name__ == '__main__':
    from usplit.configs.lc_paviaATN_config import get_config
    cfg = get_config()
    fpath = '/group/jug/ashesh/data/microscopy_zarr/'
    num_scales = 5
    padding_kwargs = {'mode': cfg.data.padding_mode}
    if 'padding_value' in cfg.data and cfg.data.padding_value is not None:
        padding_kwargs['constant_values'] = cfg.data.padding_value

    dset = MultiScaleZarrDloader(cfg.data,
                                 fpath,
                                 DataSplitType.Train,
                                 val_fraction=cfg.training.val_fraction,
                                 test_fraction=cfg.training.test_fraction,
                                 normalized_input=cfg.data.normalized_input,
                                 enable_rotation_aug=cfg.data.train_aug_rotate,
                                 use_one_mu_std=cfg.data.use_one_mu_std,
                                 num_scales=num_scales,
                                 enable_random_cropping=False,
                                 max_val=None,
                                 overlapping_padding_kwargs=None,
                                 padding_kwargs=padding_kwargs)
    mean, std = dset.compute_mean_std(
    )
    dset.set_mean_std(mean, std)
    inp, tar = dset[0]
    print(inp.shape, tar.shape)
#     (Pdb) dset.compute_individual_mean_std()[0]
#           array([[[[558.14064789]],
#         [[252.59873675]]]])
# (Pdb) dset.compute_individual_mean_std()[1]
#           array([[[[240.40617161]],
#         [[101.30987838]]]])