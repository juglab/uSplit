"""
Here, the input image is of multiple resolutions. Target image is the same.
"""
from typing import List, Tuple, Union

import numpy as np
from skimage.transform import resize

from usplit.core.data_split_type import DataSplitType
from usplit.core.data_type import DataType
from usplit.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from usplit.data_loader.patch_index_manager import GridAlignement


class MultiScaleTiffDloader(MultiChDeterministicTiffDloader):

    def __init__(
        self,
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
        overlapping_padding_kwargs=None,
    ):
        """
        Args:
            num_scales: The number of resolutions at which we want the input. Note that the target is formed at the
                        highest resolution.
        """
        self._padding_kwargs = padding_kwargs  # mode=padding_mode, constant_values=constant_value
        if overlapping_padding_kwargs is not None:
            assert self._padding_kwargs == overlapping_padding_kwargs, 'During evaluation, overlapping_padding_kwargs should be same as padding_args. \
                It should be so since we just use overlapping_padding_kwargs when it is not None'

        else:
            overlapping_padding_kwargs = padding_kwargs

        super().__init__(data_config,
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
        self.num_scales = num_scales
        assert self.num_scales is not None
        self._scaled_data = [self._data]
        assert isinstance(self.num_scales, int) and self.num_scales >= 1
        self._lowres_supervision = lowres_supervision
        assert isinstance(self._padding_kwargs, dict)
        assert 'mode' in self._padding_kwargs
        self.load_scaled_data()

    def load_scaled_data(self):
        for _ in range(1, self.num_scales):
            shape = self._scaled_data[-1].shape
            assert len(shape) == 4
            new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
            ds_data = resize(self._scaled_data[-1], new_shape)
            self._scaled_data.append(ds_data)

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f' Pad:{self._padding_kwargs}'
        return msg

    def _load_scaled_img(self, scaled_index, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index
        imgs = self._scaled_data[scaled_index][idx % self.N]
        return tuple([imgs[None, :, :, i] for i in range(imgs.shape[-1])])

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        return self._crop_img_with_padding(img, h_start, w_start)

    def _get_img(self, index: int):
        """
        Returns the primary patch along with low resolution patches centered on the primary patch.
        """
        img_tuples = self._load_img(index)
        assert self._img_sz is not None
        h, w = img_tuples[0].shape[-2:]
        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)

        cropped_img_tuples = [self._crop_flip_img(img, h_start, w_start, False, False) for img in img_tuples]

        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        allres_versions = {i: [cropped_img_tuples[i]] for i in range(len(cropped_img_tuples))}
        for scale_idx in range(1, self.num_scales):
            scaled_img_tuples = self._load_scaled_img(scale_idx, index)

            h_center = h_center // 2
            w_center = w_center // 2

            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2

            scaled_cropped_img_tuples = [
                self._crop_flip_img(img, h_start, w_start, False, False) for img in scaled_img_tuples
            ]
            for ch_idx in range(len(img_tuples)):
                allres_versions[ch_idx].append(scaled_cropped_img_tuples[ch_idx])

        output_img_tuples = tuple([np.concatenate(allres_versions[ch_idx]) for ch_idx in range(len(img_tuples))])
        return output_img_tuples

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        img_tuples = self._get_img(index)
        assert self._enable_rotation is False

        if self._lowres_supervision:
            target = np.concatenate([img[:, None] for img in img_tuples], axis=1)
        else:
            target = np.concatenate([img[:1] for img in img_tuples], axis=0)

        inp, alpha = self._compute_input(img_tuples)

        output = [inp, target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)

        # if isinstance(index, int):
        #     return inp, target

        # _, grid_size = index
        # return inp, target, grid_size

