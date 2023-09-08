# import gunpowder as gp
import argparse
import os

import numpy as np
from skimage.transform import resize
from tqdm import tqdm

import zarr
from usplit.core.data_split_type import DataSplitType, get_datasplit_tuples

def _generate_output_zarrs(output_directory, datasplit_subdir, num_samples, data_shape, num_scales, overwrite):
    output_zarrs = []
    for scale_idx in range(0, num_scales):
        zarr_path = os.path.join(output_directory, datasplit_subdir, f'{scale_idx}.zarr')
        if os.path.exists(zarr_path):
            if overwrite:
                print(f'Overwriting {zarr_path}')
            else:
                raise FileExistsError(f'{zarr_path} exists and overwrite is set to False. Exiting!')

        dsample_factor = int(np.power(2, scale_idx))
        dsample_shape = (num_samples, data_shape[1] // dsample_factor, data_shape[2] // dsample_factor, data_shape[3])
        # print(scale_idx, dsample_shape)
        dsample_chunks = (1, *dsample_shape[1:3], 1)
        dsample_output_zarr = zarr.open(zarr_path, mode='w')
        _ = dsample_output_zarr.create_dataset('raw', shape=dsample_shape, chunks=dsample_chunks, dtype='i4')
        output_zarrs.append(dsample_output_zarr)
    return output_zarrs


def generate(output_directory:str , 
             input_zarr_path: str,
             input_zarr_group:str,
             channel_order: str,
             num_scales: int,
             val_fraction: float = None,
             test_fraction: float = None,
             overwrite: bool = True,
             quantile: float = 0.995):
    """
    Generate multiscale data from a zarr file.
    A zarr file is created for each scale. scale=0 corresponds to original resolution.
    """
    assert set(channel_order).issubset('THWNZC')
    assert len(channel_order) == 4, "Only 4 dimensional data is supported"
    assert channel_order[-3:] in ['HWC', 'WHC'], 'Last dimension must be the channel dimension'

    data = zarr.open(input_zarr_path, mode='r')
    if input_zarr_group:
        data = data[input_zarr_group]

    trainidx, validx, testidx = get_datasplit_tuples(val_fraction, test_fraction, data.shape[0])
    train_output_zarrs = _generate_output_zarrs(output_directory, DataSplitType.name(DataSplitType.Train),
                                                len(trainidx), data.shape, num_scales, overwrite)
    val_output_zarrs = _generate_output_zarrs(output_directory, DataSplitType.name(DataSplitType.Val), len(validx),
                                              data.shape, num_scales, overwrite)
    test_output_zarrs = _generate_output_zarrs(output_directory, DataSplitType.name(DataSplitType.Test), len(testidx),
                                               data.shape, num_scales, overwrite)
    train_n_idx = val_n_idx = test_n_idx = 0
    mean_dict = {ch_idx: [] for ch_idx in range(data.shape[-1])}
    std_dict = {ch_idx: [] for ch_idx in range(data.shape[-1])}
    quantile_dict = {ch_idx: [] for ch_idx in range(data.shape[-1])}

    for n_idx in tqdm(range(data.shape[0])):
        for ch_idx in tqdm(range(data.shape[-1])):
            frame = data[n_idx, ..., ch_idx]
            downscaled_frame = frame
            for scale_idx in range(0, num_scales):
                # Compute the mean,std
                if scale_idx == 0 and n_idx in trainidx:
                    mean_dict[ch_idx].append(np.mean(frame))
                    std_dict[ch_idx].append(np.std(frame))
                    quantile_dict[ch_idx].append(np.quantile(frame, quantile))

                if scale_idx > 0:
                    # there is no downscaling for 1st.
                    downscaled_frame = resize(downscaled_frame, train_output_zarrs[scale_idx]['raw'].shape[1:3], 
                                              preserve_range=True, anti_aliasing=True)

                if n_idx in trainidx:
                    train_output_zarrs[scale_idx]['raw'][train_n_idx, ..., ch_idx] = downscaled_frame

                if n_idx in validx:
                    val_output_zarrs[scale_idx]['raw'][val_n_idx, ..., ch_idx] = downscaled_frame

                if n_idx in testidx:
                    test_output_zarrs[scale_idx]['raw'][test_n_idx, ..., ch_idx] = downscaled_frame

        if n_idx in trainidx:
            train_n_idx += 1
        elif n_idx in validx:
            val_n_idx += 1
        elif n_idx in testidx:
            test_n_idx += 1

    train_output_zarrs[0]['mean'] = [(k, np.mean(mean_dict[k])) for k in sorted(mean_dict.keys())]
    train_output_zarrs[0]['std'] = [(k, np.mean(std_dict[k])) for k in sorted(std_dict.keys())]
    train_output_zarrs[0]['quantile'] = [(k,quantile, np.mean(quantile_dict[k])) for k in sorted(quantile_dict.keys())]

    return train_output_zarrs, val_output_zarrs, test_output_zarrs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_zarr_path',
        help=
        'Path to zarr data. Note that we expect the data shape to be (N,H,W,C) where N can be either Time, Z, or independent acquisition dimension. C is expected to be the channel dimension.'
    )

    parser.add_argument('output_directory', help='Directory where the output zarrs will be stored')
    parser.add_argument('channel_order')
    parser.add_argument('num_scales', type=int, help='Desired number of downsampling scales', default=5)
    parser.add_argument(
        '--input_zarr_group',
        help=
        'Name of the group in the zarr file. If the zarr file is not grouped, then this argument can be ignored.',
        default=None,
    )
    parser.add_argument('--val_fraction', type=float, help='Fraction of data to be used as validation set', default=0.1)
    parser.add_argument('--test_fraction', type=float, help='Fraction of data to be used as test set', default=0.1)
    parser.add_argument('--overwrite',
                        help='If the flag is set, then existing low resolution data will be overwritten',
                        action='store_true')

    args = parser.parse_args()
    data = generate(args.output_directory,
                    args.input_zarr_path,
                    args.input_zarr_group,
                    args.channel_order,
                    args.num_scales,
                    val_fraction=args.val_fraction,
                    test_fraction=args.test_fraction,
                    overwrite=args.overwrite)

# raw = gp.ArrayKey('RAW')
# source = gp.ZarrSource(
# 'data2.zarr',  # the zarr container
# {raw: 'raw'},  # which dataset to associate to the array key
# {raw: gp.ArraySpec(interpolatable=True)}  # meta-information
# )

# dsample = gp.ArrayKey('Dsample')
# voxel_size = gp.Coordinate((1, 1, 1, 1))
# downsample_factor = gp.Coordinate(1, 2, 2, 1)
# target_voxel_size = voxel_size * downsample_factor
# pipeline = source + gp.Resample(raw, target_voxel_size, dsample)
# #+ gp.RandomLocation()
# pipeline = source + gp.DownSample(raw, 2, dsample)
# request = gp.BatchRequest()
# request[raw] = gp.Roi((1,33, 33,1), (1,64, 64,1))
# request[dsample] = gp.Roi((1,2, 2,1), (1,128, 128,1))
# with gp.build(pipeline):
#     batch = pipeline.request_batch(request)
# batch[dsample].shape
# batch[raw].data
# batch[raw].data.shape
# batch[dsample].data.shape
