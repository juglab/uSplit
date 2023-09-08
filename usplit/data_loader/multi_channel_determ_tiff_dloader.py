from typing import Tuple, Union

import albumentations as A
import numpy as np

from usplit.core.data_split_type import DataSplitType
from usplit.core.data_type import DataType
from usplit.core.empty_patch_fetcher import EmptyPatchFetcher
from usplit.data_loader.patch_index_manager import GridAlignement, GridIndexManager
from usplit.data_loader.train_val_data import get_train_val_data


class MultiChDeterministicTiffDloader:

    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 normalized_input=None,
                 enable_rotation_aug: bool = False,
                 enable_random_cropping: bool = False,
                 use_one_mu_std=None,
                 allow_generation=False,
                 max_val=None,
                 grid_alignment=GridAlignement.LeftTop,
                 overlapping_padding_kwargs=None,
                 test_img_arr: np.ndarray = None):
        """
        Here, an image is split into grids of size img_sz.
        Args:
            repeat_factor: Since we are doing a random crop, repeat_factor is
                given which can repeatedly sample from the same image. If self.N=12
                and repeat_factor is 5, then index upto 12*5 = 60 is allowed.
            use_one_mu_std: If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different meean and stdev are used.
            test_img_arr: If this is not None, then this is used for testing.

        """
        self._fpath = fpath
        self._data = self.N = None

        # NOTE: Input is the sum of the different channels. It is not the average of the different channels.
        self._input_is_sum = data_config.get('input_is_sum', False)

        self.load_data(data_config,
                       datasplit_type,
                       val_fraction=val_fraction,
                       test_fraction=test_fraction,
                       allow_generation=allow_generation,
                       test_img_arr=test_img_arr)
        self._normalized_input = normalized_input
        self._quantile = data_config.get('clip_percentile', 0.995)
        self._channelwise_quantile = data_config.get('channelwise_quantile', False)
        self._background_quantile = data_config.get('background_quantile', 0.0)
        self._clip_background_noise_to_zero = data_config.get('clip_background_noise_to_zero', False)
        self._skip_normalization_using_mean = data_config.get('skip_normalization_using_mean', False)

        self._background_values = None

        self._grid_alignment = grid_alignment
        self._overlapping_padding_kwargs = overlapping_padding_kwargs
        if self._grid_alignment == GridAlignement.LeftTop:
            assert self._overlapping_padding_kwargs is None or data_config.multiscale_lowres_count is not None, "Padding is not used with this alignement style"
        elif self._grid_alignment == GridAlignement.Center:
            assert self._overlapping_padding_kwargs is not None, 'With Center grid alignment, padding is needed.'

        self._is_train = datasplit_type == DataSplitType.Train

        # input = alpha * ch1 + (1-alpha)*ch2.
        # alpha is sampled randomly between these two extremes
        self._ch1_max_alpha = self._ch1_min_alpha = self._return_alpha = None

        self._img_sz = self._grid_sz = self._repeat_factor = self.idx_manager = None
        if self._is_train:
            self._ch1_min_alpha = data_config.get('ch1_min_alpha', None)
            self._ch1_max_alpha = data_config.get('ch1_max_alpha', None)
            self.set_img_sz(data_config.image_size,
                            data_config.grid_size if 'grid_size' in data_config else data_config.image_size)
        else:

            self.set_img_sz(data_config.image_size,
                            data_config.val_grid_size if 'val_grid_size' in data_config else data_config.image_size)

        self._return_alpha = data_config.get('return_alpha', False)

        self._empty_patch_replacement_enabled = data_config.get("empty_patch_replacement_enabled",
                                                                False) and self._is_train
        if self._empty_patch_replacement_enabled:
            self._empty_patch_replacement_channel_idx = data_config.empty_patch_replacement_channel_idx
            self._empty_patch_replacement_probab = data_config.empty_patch_replacement_probab
            data_frames = self._data[..., self._empty_patch_replacement_channel_idx]
            # NOTE: This is on the raw data. So, it must be called before removing the background.
            self._empty_patch_fetcher = EmptyPatchFetcher(self.idx_manager,
                                                          self._img_sz,
                                                          data_frames,
                                                          max_val_threshold=data_config.empty_patch_max_val_threshold)

        self.rm_bkground_set_max_val_and_upperclip_data(max_val, datasplit_type)

        # For overlapping dloader, image_size and repeat_factors are not related. hence a different function.

        self._mean = None
        self._std = None
        self._use_one_mu_std = use_one_mu_std
        self._enable_rotation = enable_rotation_aug
        self._enable_random_cropping = enable_random_cropping
        # Randomly rotate [-90,90]

        self._rotation_transform = None
        if self._enable_rotation:
            self._rotation_transform = A.Compose([A.Flip(), A.RandomRotate90()])

        msg = self._init_msg()
        print(msg)

    def get_data_shape(self):
        return self._data.shape

    def load_data(self, data_config, datasplit_type, val_fraction=None, test_fraction=None, allow_generation=None,test_img_arr=None):
        if test_img_arr is not None:
            assert datasplit_type == DataSplitType.Test, 'test_img_arr can only be used for testing.'
            assert len(test_img_arr.shape) ==3, 'test_img_arr must be a 3D array. (Batch, H, W)'
            self._data = np.repeat(test_img_arr[...,None], 2, axis=-1)
            
            if self._input_is_sum:
                # sum of the two channels should be the input.
                self._data = self._data/2

        else:
            self._data = get_train_val_data(data_config,
                                            self._fpath,
                                            datasplit_type,
                                            val_fraction=val_fraction,
                                            test_fraction=test_fraction,
                                            allow_generation=allow_generation)
        self.N = len(self._data)

    def save_background(self, channel_idx, frame_idx, background_value):
        self._background_values[frame_idx, channel_idx] = background_value

    def get_background(self, channel_idx, frame_idx):
        return self._background_values[frame_idx, channel_idx]

    def remove_background(self):

        self._background_values = np.zeros((self._data.shape[0], self._data.shape[-1]))

        if self._background_quantile == 0.0:
            assert self._clip_background_noise_to_zero is False, 'This operation currently happens later in this function.'
            return

        if self._data.dtype in [np.uint16]:
            # unsigned integer creates havoc
            self._data = self._data.astype(np.int32)

        for ch in range(self._data.shape[-1]):
            for idx in range(self._data.shape[0]):
                qval = np.quantile(self._data[idx, ..., ch], self._background_quantile)
                assert np.abs(
                    qval
                ) > 20, "We are truncating the qval to an integer which will only make sense if it is large enough"
                # NOTE: Here, there can be an issue if you work with normalized data
                qval = int(qval)
                self.save_background(ch, idx, qval)
                self._data[idx, ..., ch] -= qval

        if self._clip_background_noise_to_zero:
            self._data[self._data < 0] = 0

    def rm_bkground_set_max_val_and_upperclip_data(self, max_val, datasplit_type):
        self.remove_background()
        self.set_max_val(max_val, datasplit_type)
        self.upperclip_data()

    def upperclip_data(self):
        if isinstance(self.max_val, list):
            chN = self._data.shape[-1]
            assert chN == len(self.max_val)
            for ch in range(chN):
                ch_data = self._data[..., ch]
                ch_q = self.max_val[ch]
                ch_data[ch_data > ch_q] = ch_q
                self._data[..., ch] = ch_data
        else:
            self._data[self._data > self.max_val] = self.max_val

    def compute_max_val(self):
        if self._channelwise_quantile:
            max_val_arr = [np.quantile(self._data[..., i], self._quantile) for i in range(self._data.shape[-1])]
            return max_val_arr
        else:
            return np.quantile(self._data, self._quantile)

    def set_max_val(self, max_val, datasplit_type):
        if datasplit_type == DataSplitType.Train:
            assert max_val is None
            self.max_val = self.compute_max_val()
        else:
            assert max_val is not None
            self.max_val = max_val

    def get_max_val(self):
        return self.max_val

    def get_img_sz(self):
        return self._img_sz

    def set_img_sz(self, image_size, grid_size):
        """
        If one wants to change the image size on the go, then this can be used.
        Args:
            image_size: size of one patch
            grid_size: frame is divided into square grids of this size. A patch centered on a grid having size `image_size` is returned.
        """

        self._img_sz = image_size
        self._grid_sz = grid_size
        self.idx_manager = GridIndexManager(self._data.shape, self._grid_sz, self._img_sz, self._grid_alignment)
        self.set_repeat_factor()

    def set_repeat_factor(self):
        self._repeat_factor = self.idx_manager.grid_rows(self._grid_sz) * self.idx_manager.grid_cols(self._grid_sz)

    def _init_msg(self, ):
        msg = f'[{self.__class__.__name__}] Sz:{self._img_sz}'
        msg += f' Train:{int(self._is_train)} N:{self.N} NumPatchPerN:{self._repeat_factor}'
        msg += f' NormInp:{self._normalized_input}'
        msg += f' SingleNorm:{self._use_one_mu_std}'
        msg += f' Rot:{self._enable_rotation}'
        msg += f' RandCrop:{self._enable_random_cropping}'
        msg += f' Q:{self._quantile}'
        msg += f' SummedInput:{self._input_is_sum}'
        msg += f' ReplaceWithRandSample:{self._empty_patch_replacement_enabled}'
        if self._empty_patch_replacement_enabled:
            msg += f'-{self._empty_patch_replacement_channel_idx}-{self._empty_patch_replacement_probab}'

        msg += f' BckQ:{self._background_quantile}'
        if self._ch1_min_alpha is not None:
            msg += f' Alpha:[{self._ch1_min_alpha},{self._ch1_max_alpha}]'
        return msg

    def _crop_imgs(self, index, *img_tuples: np.ndarray):
        h, w = img_tuples[0].shape[-2:]
        if self._img_sz is None:
            return (*img_tuples, {'h': [0, h], 'w': [0, w], 'hflip': False, 'wflip': False})

        if self._enable_random_cropping:
            h_start, w_start = self._get_random_hw(h, w)
        else:
            h_start, w_start = self._get_deterministic_hw(index)

        cropped_imgs = []
        for img in img_tuples:
            img = self._crop_flip_img(img, h_start, w_start, False, False)
            cropped_imgs.append(img)

        return (*tuple(cropped_imgs), {
            'h': [h_start, h_start + self._img_sz],
            'w': [w_start, w_start + self._img_sz],
            'hflip': False,
            'wflip': False,
        })

    def _crop_img(self, img: np.ndarray, h_start: int, w_start: int):
        if self._grid_alignment == GridAlignement.LeftTop:
            # In training, this is used.
            # NOTE: It is my opinion that if I just use self._crop_img_with_padding, it will work perfectly fine.
            # The only benefit this if else loop provides is that it makes it easier to see what happens during training.
            new_img = img[..., h_start:h_start + self._img_sz, w_start:w_start + self._img_sz]
            return new_img
        elif self._grid_alignment == GridAlignement.Center:
            # During evaluation, this is used. In this situation, we can have negative h_start, w_start. Or h_start +self._img_sz can be larger than frame
            # In these situations, we need some sort of padding. This is not needed  in the LeftTop alignement.
            return self._crop_img_with_padding(img, h_start, w_start)

    def get_begin_end_padding(self, start_pos, max_len):
        """
        The effect is that the image with size self._grid_sz is in the center of the patch with sufficient
        padding on all four sides so that the final patch size is self._img_sz.
        """
        pad_start = 0
        pad_end = 0
        if start_pos < 0:
            pad_start = -1 * start_pos

        pad_end = max(0, start_pos + self._img_sz - max_len)

        return pad_start, pad_end

    def _crop_img_with_padding(self, img: np.ndarray, h_start: int, w_start: int):
        _, H, W = img.shape
        h_on_boundary = self.on_boundary(h_start, H)
        w_on_boundary = self.on_boundary(w_start, W)

        assert h_start < H
        assert w_start < W

        assert h_start + self._img_sz <= H or h_on_boundary
        assert w_start + self._img_sz <= W or w_on_boundary
        # max() is needed since h_start could be negative.
        new_img = img[..., max(0, h_start):h_start + self._img_sz, max(0, w_start):w_start + self._img_sz]
        padding = np.array([[0, 0], [0, 0], [0, 0]])

        if h_on_boundary:
            pad = self.get_begin_end_padding(h_start, H)
            padding[1] = pad
        if w_on_boundary:
            pad = self.get_begin_end_padding(w_start, W)
            padding[2] = pad

        if not np.all(padding == 0):
            new_img = np.pad(new_img, padding, **self._overlapping_padding_kwargs)

        return new_img

    def _crop_flip_img(self, img: np.ndarray, h_start: int, w_start: int, h_flip: bool, w_flip: bool):
        new_img = self._crop_img(img, h_start, w_start)
        if h_flip:
            new_img = new_img[..., ::-1, :]
        if w_flip:
            new_img = new_img[..., :, ::-1]

        return new_img.astype(np.float32)

    def __len__(self):
        return self.N * self._repeat_factor

    def _load_img(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx = index[0]

        imgs = self._data[self.idx_manager.get_t(idx)]
        loaded_imgs = [imgs[None, ..., i] for i in range(imgs.shape[-1])]
        return tuple(loaded_imgs)

    def get_mean_std(self):
        return self._mean, self._std

    def set_mean_std(self, mean_val, std_val):
        self._mean = mean_val
        self._std = std_val

    def normalize_img(self, *img_tuples):
        mean, std = self.get_mean_std()
        mean = mean.squeeze()
        std = std.squeeze()
        normalized_imgs = []
        for i, img in enumerate(img_tuples):
            img = (img - mean[i]) / std[i]
            normalized_imgs.append(img)
        return tuple(normalized_imgs)

    def get_grid_size(self):
        return self._grid_sz

    def get_idx_manager(self):
        return self.idx_manager

    def per_side_overlap_pixelcount(self):
        return (self._img_sz - self._grid_sz) // 2

    def on_boundary(self, cur_loc, frame_size):
        return cur_loc + self._img_sz > frame_size or cur_loc < 0

    def _get_deterministic_hw(self, index: Union[int, Tuple[int, int]]):
        """
        It returns the top-left corner of the patch corresponding to index.
        """
        if isinstance(index, int):
            idx = index
            grid_size = self._grid_sz
        else:
            idx, grid_size = index

        h_start, w_start = self.idx_manager.get_deterministic_hw(idx, grid_size=grid_size)
        if self._grid_alignment == GridAlignement.LeftTop:
            return h_start, w_start
        elif self._grid_alignment == GridAlignement.Center:
            pad = self.per_side_overlap_pixelcount()
            return h_start - pad, w_start - pad

    def compute_individual_mean_std(self):
        # numpy 1.19.2 has issues in computing for large arrays. https://github.com/numpy/numpy/issues/8869
        # mean = np.mean(self._data, axis=(0, 1, 2))
        # std = np.std(self._data, axis=(0, 1, 2))
        mean_arr = []
        std_arr = []
        for ch_idx in range(self._data.shape[-1]):
            mean_ = 0.0 if self._skip_normalization_using_mean else self._data[..., ch_idx].mean()
            std_ = self._data[..., ch_idx].std()
            mean_arr.append(mean_)
            std_arr.append(std_)

        mean = np.array(mean_arr)
        std = np.array(std_arr)

        return mean[None, :, None, None], std[None, :, None, None]

    def compute_mean_std(self, allow_for_validation_data=False):
        """
        Note that we must compute this only for training data.
        """
        assert self._is_train is True or allow_for_validation_data, 'This is just allowed for training data'
        if self._use_one_mu_std is True:
            if self._input_is_sum:
                mean = [np.mean(self._data[..., k:k + 1], keepdims=True) for k in range(self._data.shape[-1])]
                mean = np.sum(mean, keepdims=True)[0]
                std = np.linalg.norm(
                    [np.std(self._data[..., k:k + 1], keepdims=True) for k in range(self._data.shape[-1])],
                    keepdims=True)[0]
            else:
                mean = np.mean(self._data, keepdims=True).reshape(1, 1, 1, 1)
                std = np.std(self._data, keepdims=True).reshape(1, 1, 1, 1)
            mean = np.repeat(mean, 2, axis=1)
            std = np.repeat(std, 2, axis=1)

            if self._skip_normalization_using_mean:
                mean = np.zeros_like(mean)

            return mean, std

        elif self._use_one_mu_std is False:
            return self.compute_individual_mean_std()

        elif self._use_one_mu_std is None:
            return np.array([0.0, 0.0]).reshape(1, 2, 1, 1), np.array([1.0, 1.0]).reshape(1, 2, 1, 1)

    def _get_random_hw(self, h: int, w: int):
        """
        Random starting position for the crop for the img with index `index`.
        """
        if h != self._img_sz:
            h_start = np.random.choice(h - self._img_sz)
            w_start = np.random.choice(w - self._img_sz)
        else:
            h_start = 0
            w_start = 0
        return h_start, w_start

    def _get_img(self, index: Union[int, Tuple[int, int]]):
        """
        Loads an image.
        Crops the image such that cropped image has content.
        """
        img_tuples = self._load_img(index)
        cropped_img_tuples = self._crop_imgs(index, *img_tuples)[:-1]
        return cropped_img_tuples

    def replace_with_empty_patch(self, img_tuples):
        empty_index = self._empty_patch_fetcher.sample()
        empty_img_tuples = self._get_img(empty_index)
        final_img_tuples = []
        for tuple_idx in range(len(img_tuples)):
            if tuple_idx == self._empty_patch_replacement_channel_idx:
                final_img_tuples.append(empty_img_tuples[tuple_idx])
            else:
                final_img_tuples.append(img_tuples[tuple_idx])
        return tuple(final_img_tuples)

    def _compute_input_with_alpha(self, img_tuples, alpha):
        assert len(img_tuples) == 2, f'Expected {len(img_tuples)} to be 2'
        assert self._normalized_input is True, "normalization should happen here"

        inp = img_tuples[0] * alpha + img_tuples[1] * (1 - alpha)
        mean, std = self.get_mean_std()
        mean = mean.squeeze()
        std = std.squeeze()
        assert mean[0] == mean[1] and len(mean) == 2
        assert std[0] == std[1] and len(std) == 2

        inp = (inp - mean[0]) / std[0]
        return inp.astype(np.float32)

    def _sample_alpha(self):
        alpha_width = self._ch1_max_alpha - self._ch1_min_alpha
        alpha = np.random.rand() * (alpha_width) + self._ch1_min_alpha
        return alpha

    def _compute_input(self, img_tuples):
        alpha = 0.5
        if self._ch1_min_alpha is not None:
            alpha = self._sample_alpha()

        inp = self._compute_input_with_alpha(img_tuples, alpha)
        if self._input_is_sum:
            inp = 2 * inp
        return inp, alpha

    def __getitem__(self, index: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        img_tuples = self._get_img(index)
        if self._empty_patch_replacement_enabled:
            if np.random.rand() < self._empty_patch_replacement_probab:
                img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            # passing just the 2D input. 3rd dimension messes up things.
            assert len(img_tuples) == 2
            rot_dic = self._rotation_transform(image=img_tuples[0][0], mask=img_tuples[1][0])
            img1 = rot_dic['image'][None]
            img2 = rot_dic['mask'][None]

        target = np.concatenate(img_tuples, axis=0)
        inp, alpha = self._compute_input(img_tuples)

        output = [inp, target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)


if __name__ == '__main__':
    from usplit.configs.microscopy_multi_channel_lvae_config import get_config
    config = get_config()
    dset = MultiChDeterministicTiffDloader(config.data,
                                           '/group/jug/ashesh/data/microscopy/OptiMEM100x014.tif',
                                           DataSplitType.Train,
                                           val_fraction=config.training.val_fraction,
                                           test_fraction=config.training.test_fraction,
                                           normalized_input=config.data.normalized_input,
                                           enable_rotation_aug=config.data.normalized_input,
                                           enable_random_cropping=config.data.deterministic_grid is False,
                                           use_one_mu_std=config.data.use_one_mu_std,
                                           allow_generation=False,
                                           max_val=None,
                                           grid_alignment=GridAlignement.LeftTop,
                                           overlapping_padding_kwargs=None)

    mean, std = dset.compute_mean_std()
    dset.set_mean_std(mean, std)

    inp, target, alpha = dset[0]
