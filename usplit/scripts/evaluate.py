import argparse
import glob
import os
import pickle
import random
import re
import sys
from posixpath import basename

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

import ml_collections
from usplit.analysis.mmse_prediction import get_dset_predictions
from usplit.analysis.results_handler import PaperResultsHandler
from usplit.analysis.stitch_prediction import stitch_predictions
from usplit.config_utils import load_config
from usplit.core.data_split_type import DataSplitType
from usplit.core.data_type import DataType
from usplit.core.model_type import ModelType
from usplit.core.psnr import PSNR, RangeInvariantPsnr
from usplit.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from usplit.data_loader.multiscale_mc_tiff_dloader import MultiScaleTiffDloader
from usplit.data_loader.patch_index_manager import TilingMode
from usplit.training import create_model

torch.multiprocessing.set_sharing_strategy('file_system')
DATA_ROOT = 'PUT THE ROOT DIRECTORY FOR THE DATASET HERE'
CODE_ROOT = 'PUT THE ROOT DIRECTORY FOR THE CODE HERE'


def _avg_psnr(target, prediction, psnr_fn):
    output = np.mean([psnr_fn(target[i:i + 1], prediction[i:i + 1]).item() for i in range(len(prediction))])
    return round(output, 2)


def avg_range_inv_psnr(target, prediction):
    return _avg_psnr(target, prediction, RangeInvariantPsnr)


def avg_psnr(target, prediction):
    return _avg_psnr(target, prediction, PSNR)


def compute_masked_psnr(mask, tar1, tar2, pred1, pred2):
    mask = mask.astype(bool)
    mask = mask[..., 0]
    tmp_tar1 = tar1[mask].reshape((len(tar1), -1, 1))
    tmp_pred1 = pred1[mask].reshape((len(tar1), -1, 1))
    tmp_tar2 = tar2[mask].reshape((len(tar2), -1, 1))
    tmp_pred2 = pred2[mask].reshape((len(tar2), -1, 1))
    psnr1 = avg_range_inv_psnr(tmp_tar1, tmp_pred1)
    psnr2 = avg_range_inv_psnr(tmp_tar2, tmp_pred2)
    return psnr1, psnr2


def avg_ssim(target, prediction):
    ssim = [
        structural_similarity(target[i], prediction[i], data_range=target[i].max() - target[i].min())
        for i in range(len(target))
    ]
    return np.mean(ssim), np.std(ssim)


def fix_seeds():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True


def main(
    ckpt_dir,
    DEBUG,
    image_size_for_grid_centers=64,
    mmse_count=1,
    custom_image_size=64,
    batch_size=16,
    num_workers=4,
    COMPUTE_LOSS=False,
    use_deterministic_grid=None,
    threshold=None,  # 0.02,
    compute_kl_loss=False,
    evaluate_train=False,
    eval_datasplit_type=DataSplitType.Val,
    val_repeat_factor=None,
    psnr_type='range_invariant',
    ignored_last_pixels=0,
    ignore_first_pixels=0,
    print_token='',
    normalized_ssim=True,
):
    global DATA_ROOT, CODE_ROOT

    homedir = os.path.expanduser('~')
    nodename = os.uname().nodename

    if nodename == 'capablerutherford-02aa4':
        DATA_ROOT = '/mnt/ashesh/'
        CODE_ROOT = '/home/ubuntu/ashesh/'
    elif nodename in ['capableturing-34a32', 'colorfuljug-fa782', 'agileschroedinger-a9b1c', 'rapidkepler-ca36f']:
        DATA_ROOT = '/home/ubuntu/ashesh/data/'
        CODE_ROOT = '/home/ubuntu/ashesh/'
    elif (re.match('lin-jug-\d{2}', nodename) or re.match('gnode\d{2}', nodename)
          or re.match('lin-jug-m-\d{2}', nodename) or re.match('lin-jug-l-\d{2}', nodename)):
        DATA_ROOT = '/group/jug/ashesh/data/'
        CODE_ROOT = '/home/ashesh.ashesh/'

    dtype = int(ckpt_dir.split('/')[-2].split('-')[0][1:])

    if DEBUG:
        if dtype == DataType.CustomSinosoid:
            data_dir = f'{DATA_ROOT}/sinosoid/'
        elif dtype == DataType.OptiMEM100_014:
            data_dir = f'{DATA_ROOT}/microscopy/'
    else:
        if dtype == DataType.CustomSinosoid:
            data_dir = f'{DATA_ROOT}/sinosoid/'
        elif dtype == DataType.CustomSinosoidThreeCurve:
            data_dir = f'{DATA_ROOT}/sinosoid/'
        elif dtype == DataType.OptiMEM100_014:
            data_dir = f'{DATA_ROOT}/microscopy/'
        elif dtype == DataType.Prevedel_EMBL:
            data_dir = f'{DATA_ROOT}/Prevedel_EMBL/PKG_3P_dualcolor_stacks/NoAverage_NoRegistration/'
        elif dtype == DataType.AllenCellMito:
            data_dir = f'{DATA_ROOT}/allencell/2017_03_08_Struct_First_Pass_Seg/AICS-11/'
        elif dtype == DataType.SeparateTiffData:
            data_dir = f'{DATA_ROOT}/ventura_gigascience'

    homedir = os.path.expanduser('~')
    nodename = os.uname().nodename

    def get_best_checkpoint(ckpt_dir):
        output = []
        for filename in glob.glob(ckpt_dir + "/*_best.ckpt"):
            output.append(filename)
        assert len(output) == 1, '\n'.join(output)
        return output[0]

    config = load_config(ckpt_dir)
    config = ml_collections.ConfigDict(config)
    old_image_size = None
    with config.unlocked():
        try:
            if 'batchnorm' not in config.model.encoder:
                config.model.encoder.batchnorm = config.model.batchnorm
                assert 'batchnorm' not in config.model.decoder
                config.model.decoder.batchnorm = config.model.batchnorm

            if 'conv2d_bias' not in config.model.decoder:
                config.model.decoder.conv2d_bias = True

            if config.model.model_type == ModelType.LadderVaeSepEncoder:
                if 'use_random_for_missing_inp' not in config.model:
                    config.model.use_random_for_missing_inp = False
                if 'learnable_merge_tensors' not in config.model:
                    config.model.learnable_merge_tensors = False
        except:
            pass

        if config.model.model_type == ModelType.UNet and 'n_levels' not in config.model:
            config.model.n_levels = 4
        if 'test_fraction' not in config.training:
            config.training.test_fraction = 0.0

        if 'datadir' not in config:
            config.datadir = ''
        if 'encoder' not in config.model:
            config.model.encoder = ml_collections.ConfigDict()
            assert 'decoder' not in config.model
            config.model.decoder = ml_collections.ConfigDict()

            config.model.encoder.dropout = config.model.dropout
            config.model.decoder.dropout = config.model.dropout
            config.model.encoder.blocks_per_layer = config.model.blocks_per_layer
            config.model.decoder.blocks_per_layer = config.model.blocks_per_layer
            config.model.encoder.n_filters = config.model.n_filters
            config.model.decoder.n_filters = config.model.n_filters

        if 'multiscale_retain_spatial_dims' not in config.model.decoder:
            config.model.decoder.multiscale_retain_spatial_dims = False

        if 'res_block_kernel' not in config.model.encoder:
            config.model.encoder.res_block_kernel = 3
            assert 'res_block_kernel' not in config.model.decoder
            config.model.decoder.res_block_kernel = 3

        if 'res_block_skip_padding' not in config.model.encoder:
            config.model.encoder.res_block_skip_padding = False
            assert 'res_block_skip_padding' not in config.model.decoder
            config.model.decoder.res_block_skip_padding = False

        if config.data.data_type == DataType.CustomSinosoid:
            if 'max_vshift_factor' not in config.data:
                config.data.max_vshift_factor = config.data.max_shift_factor
                config.data.max_hshift_factor = 0
            if 'encourage_non_overlap_single_channel' not in config.data:
                config.data.encourage_non_overlap_single_channel = False

        if 'skip_bottom_layers_count' in config.model:
            config.model.skip_bottom_layers_count = 0

        if 'logvar_lowerbound' not in config.model:
            config.model.logvar_lowerbound = None
        if 'train_aug_rotate' not in config.data:
            config.data.train_aug_rotate = False
        if 'multiscale_lowres_separate_branch' not in config.model:
            config.model.multiscale_lowres_separate_branch = False
        if 'multiscale_retain_spatial_dims' not in config.model:
            config.model.multiscale_retain_spatial_dims = False
        config.data.train_aug_rotate = False

        if 'randomized_channels' not in config.data:
            config.data.randomized_channels = False

        if 'predict_logvar' not in config.model:
            config.model.predict_logvar = None
        if config.data.data_type in [
                DataType.OptiMEM100_014, DataType.CustomSinosoid, DataType.CustomSinosoidThreeCurve,
                DataType.SeparateTiffData
        ]:
            if custom_image_size is not None:
                old_image_size = config.data.image_size
                config.data.image_size = custom_image_size
            if use_deterministic_grid is not None:
                config.data.deterministic_grid = use_deterministic_grid
            if threshold is not None:
                config.data.threshold = threshold
            if val_repeat_factor is not None:
                config.training.val_repeat_factor = val_repeat_factor
            config.model.mode_pred = not compute_kl_loss

    print(config)
    with config.unlocked():
        config.model.skip_nboundary_pixels_from_loss = None

    ## Disentanglement setup.
    ####
    ####
    grid_alignment = TilingMode.ShiftBoundary
    if image_size_for_grid_centers is not None:
        old_grid_size = config.data.get('grid_size', "grid_size not present")
        with config.unlocked():
            config.data.grid_size = image_size_for_grid_centers
            config.data.val_grid_size = image_size_for_grid_centers

    padding_kwargs = {
        'mode': config.data.get('padding_mode', 'constant'),
    }

    if padding_kwargs['mode'] == 'constant':
        padding_kwargs['constant_values'] = config.data.get('padding_value', 0)

    dloader_kwargs = {'overlapping_padding_kwargs': padding_kwargs, 'grid_alignment': grid_alignment}

    if 'multiscale_lowres_count' in config.data and config.data.multiscale_lowres_count is not None:
        data_class = MultiScaleTiffDloader
        dloader_kwargs['num_scales'] = config.data.multiscale_lowres_count
        dloader_kwargs['padding_kwargs'] = padding_kwargs
    elif config.data.data_type == DataType.SemiSupBloodVesselsEMBL:
        data_class = SingleChannelDloader
    else:
        data_class = MultiChDeterministicTiffDloader
    if config.data.data_type in [
            DataType.CustomSinosoid, DataType.CustomSinosoidThreeCurve, DataType.AllenCellMito,
            DataType.SeparateTiffData, DataType.SemiSupBloodVesselsEMBL
    ]:
        datapath = data_dir
    elif config.data.data_type == DataType.OptiMEM100_014:
        datapath = os.path.join(data_dir, 'OptiMEM100x014.tif')
    elif config.data.data_type == DataType.Prevedel_EMBL:
        datapath = os.path.join(data_dir, 'MS14__z0_8_sl4_fr10_p_10.1_lz510_z13_bin5_00001.tif')

    normalized_input = config.data.normalized_input
    use_one_mu_std = config.data.use_one_mu_std
    train_aug_rotate = config.data.train_aug_rotate
    enable_random_cropping = config.data.deterministic_grid is False

    train_dset = data_class(config.data,
                            datapath,
                            datasplit_type=DataSplitType.Train,
                            val_fraction=config.training.val_fraction,
                            test_fraction=config.training.test_fraction,
                            normalized_input=normalized_input,
                            use_one_mu_std=use_one_mu_std,
                            enable_rotation_aug=train_aug_rotate,
                            enable_random_cropping=enable_random_cropping,
                            **dloader_kwargs)
    import gc
    gc.collect()
    max_val = train_dset.get_max_val()
    val_dset = data_class(
        config.data,
        datapath,
        datasplit_type=eval_datasplit_type,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        normalized_input=normalized_input,
        use_one_mu_std=use_one_mu_std,
        enable_rotation_aug=False,  # No rotation aug on validation
        enable_random_cropping=False,
        # No random cropping on validation. Validation is evaluated on determistic grids
        max_val=max_val,
        **dloader_kwargs)

    # For normalizing, we should be using the training data's mean and std.
    mean_val, std_val = train_dset.compute_mean_std()
    train_dset.set_mean_std(mean_val, std_val)
    val_dset.set_mean_std(mean_val, std_val)

    if evaluate_train:
        val_dset = train_dset
    data_mean, data_std = train_dset.get_mean_std()

    with config.unlocked():
        if config.data.data_type in [
                DataType.OptiMEM100_014, DataType.CustomSinosoid, DataType.CustomSinosoidThreeCurve,
                DataType.SeparateTiffData
        ] and old_image_size is not None:
            config.data.image_size = old_image_size

    if config.data.target_separate_normalization is True:
        model = create_model(config, *train_dset.compute_individual_mean_std())
    else:
        model = create_model(config, *train_dset.get_mean_std())

    ckpt_fpath = get_best_checkpoint(ckpt_dir)
    checkpoint = torch.load(ckpt_fpath)

    _ = model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    _ = model.cuda()

    model.data_mean = model.data_mean.cuda()
    model.data_std = model.data_std.cuda()
    print('Loading from epoch', checkpoint['epoch'])

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Model has {count_parameters(model)/1000_000:.3f}M parameters')

    if config.data.multiscale_lowres_count is not None and custom_image_size is not None:
        model.reset_for_different_output_size(custom_image_size)

    pred_tiled, rec_loss, *_ = get_dset_predictions(
        model,
        val_dset,
        batch_size,
        num_workers=num_workers,
        mmse_count=mmse_count,
        model_type=config.model.model_type,
    )
    pred = stitch_predictions(pred_tiled, val_dset)
    print('Stitched predictions shape', pred.shape)

    def print_ignored_pixels():
        ignored_pixels = 1
        while (pred[0, -ignored_pixels:, -ignored_pixels:, ].std() == 0):
            ignored_pixels += 1
        ignored_pixels -= 1
        # print(f'In {pred.shape}, {ignored_pixels} many rows and columns are all zero.')
        return ignored_pixels

    actual_ignored_pixels = print_ignored_pixels()
    tar = val_dset._data

    def ignore_pixels(arr):
        if ignore_first_pixels:
            arr = arr[:, ignore_first_pixels:, ignore_first_pixels:]
        if ignored_last_pixels:
            arr = arr[:, :-ignored_last_pixels, :-ignored_last_pixels]
        return arr

    pred = ignore_pixels(pred)
    tar = ignore_pixels(tar)

    sep_mean, sep_std = model.data_mean, model.data_std
    sep_mean = sep_mean.squeeze()[None, None, None]
    sep_std = sep_std.squeeze()[None, None, None]

    ch1_pred_unnorm = pred[..., 0] * sep_std[..., 0].cpu().numpy() + sep_mean[..., 0].cpu().numpy()
    ch2_pred_unnorm = pred[..., 1] * sep_std[..., 1].cpu().numpy() + sep_mean[..., 1].cpu().numpy()

    # pred is already normalized. no need to do it.
    pred1, pred2 = pred[..., 0].astype(np.float32), pred[..., 1].astype(np.float32)
    # tar1, tar2 = val_dset.normalize_img(tar[...,0], tar[...,1])
    tar_normalized = (tar - sep_mean.cpu().numpy()) / sep_std.cpu().numpy()
    tar1 = tar_normalized[..., 0]
    tar2 = tar_normalized[..., 1]

    rmse1 = np.sqrt(((pred1 - tar1)**2).reshape(len(pred1), -1).mean(axis=1))
    rmse2 = np.sqrt(((pred2 - tar2)**2).reshape(len(pred2), -1).mean(axis=1))

    rmse = (rmse1 + rmse2) / 2
    rmse = np.round(rmse, 3)

    if not normalized_ssim:
        ssim1_mean, ssim1_std = avg_ssim(tar[..., 0], ch1_pred_unnorm)
        ssim2_mean, ssim2_std = avg_ssim(tar[..., 1], ch2_pred_unnorm)
    else:
        ssim1_mean, ssim1_std = avg_ssim(tar_normalized[..., 0], pred[..., 0])
        ssim2_mean, ssim2_std = avg_ssim(tar_normalized[..., 1], pred[..., 1])

    # Computing the output statistics.
    output_stats = {}
    output_stats['rec_loss'] = rec_loss.mean()
    output_stats['rmse'] = [np.mean(rmse1), np.mean(rmse2), np.mean(rmse)]
    output_stats['psnr'] = [avg_psnr(tar1, pred1), avg_psnr(tar2, pred2)]
    output_stats['rangeinvpsnr'] = [avg_range_inv_psnr(tar1, pred1), avg_range_inv_psnr(tar2, pred2)]
    output_stats['ssim'] = [ssim1_mean, ssim2_mean, ssim1_std, ssim2_std]
    output_stats['normalized_ssim'] = normalized_ssim

    print(print_token)
    print('Rec Loss', np.round(output_stats['rec_loss'], 3))
    print('RMSE', output_stats['rmse'][0].round(3), output_stats['rmse'][1].round(3), output_stats['rmse'][2].round(3))
    print('PSNR', output_stats['psnr'][0], output_stats['psnr'][1])
    print('RangeInvPSNR', output_stats['rangeinvpsnr'][0], output_stats['rangeinvpsnr'][1])
    if normalized_ssim:
        print('SSIM normalized:', round(ssim1_mean, 3), round(ssim2_mean, 3), '±', round((ssim1_std + ssim2_std) / 2,
                                                                                         4))
    else:
        print('SSIM:', round(ssim1_mean, 3), round(ssim2_mean, 3), '±', round((ssim1_std + ssim2_std) / 2, 4))
    print()

    # if config.data.data_type == DataType.SeparateTiffData:
    #     # comparing psnr with highres data
    #     if eval_datasplit_type == DataSplitType.Val:
    #         N = len(pred1) / config.training.val_fraction
    #     elif eval_datasplit_type == DataSplitType.Test:
    #         N = len(pred1) / config.training.test_fraction

    #     train_idx, val_idx_list, test_idx_list = get_datasplit_tuples(config.training.val_fraction,
    #                                                                   config.training.test_fraction,
    #                                                                   N,
    #                                                                   starting_test=True)
    #     highres_actin = load_tiff('/home/ashesh.ashesh/data/ventura_gigascience/actin-60x-noise2-highsnr.tif')[...,
    #                                                                                                            None]
    #     highres_mito = load_tiff('/home/ashesh.ashesh/data/ventura_gigascience/mito-60x-noise2-highsnr.tif')[..., None]

    #     if eval_datasplit_type == DataSplitType.Val:
    #         highres_data = np.concatenate([highres_actin[val_idx_list], highres_mito[val_idx_list]],
    #                                       axis=-1).astype(np.float32)
    #     elif eval_datasplit_type == DataSplitType.Test:
    #         highres_data = np.concatenate([highres_actin[test_idx_list], highres_mito[test_idx_list]],
    #                                       axis=-1).astype(np.float32)

    #     thresh = np.quantile(highres_data, config.data.clip_percentile)
    #     highres_data[highres_data > thresh] = thresh

    #     output_stats['highres_psnr'] = [avg_psnr(highres_data[..., 0], pred1), avg_psnr(highres_data[..., 1], pred2)]
    #     output_stats['highres_rinvpsnr'] = [
    #         avg_range_inv_psnr(highres_data[..., 0], pred1),
    #         avg_range_inv_psnr(highres_data[..., 1], pred2)
    #     ]
    #     print('PSNR with HighRes', output_stats['highres_psnr'][0], output_stats['highres_psnr'][1])
    #     print('RangeInvPSNR with HighRes', output_stats['highres_rinvpsnr'][0], output_stats['highres_rinvpsnr'][1])
    return output_stats


def save_hardcoded_ckpt_evaluations_to_file(normalized_ssim=True):
    ckpt_dirs = [
        '/home/ashesh.ashesh/training/disentangle/2210/D7-M3-S0-L0/79',
    ]
    if ckpt_dirs[0].startswith('/home/ashesh.ashesh'):
        OUTPUT_DIR = os.path.expanduser('/group/jug/ashesh/data/paper_stats/')
    elif ckpt_dirs[0].startswith('/home/ubuntu/ashesh'):
        OUTPUT_DIR = os.path.expanduser('~/data/paper_stats/')
    else:
        raise Exception('Invalid server')

    ckpt_dirs = [x[:-1] if '/' == x[-1] else x for x in ckpt_dirs]
    mmse_count = 1

    patchsz_gridsz_tuples = [(64, 32)]
    for custom_image_size, image_size_for_grid_centers in patchsz_gridsz_tuples:
        for eval_datasplit_type in [DataSplitType.Test]:
            for ckpt_dir in ckpt_dirs:
                ignored_last_pixels = 32 if os.path.basename(os.path.dirname(ckpt_dir)).split('-')[0][1:] == '3' else 0
                handler = PaperResultsHandler(OUTPUT_DIR, eval_datasplit_type, custom_image_size,
                                              image_size_for_grid_centers, mmse_count, ignored_last_pixels)
                data = main(
                    ckpt_dir,
                    DEBUG,
                    image_size_for_grid_centers=image_size_for_grid_centers,
                    mmse_count=mmse_count,
                    custom_image_size=custom_image_size,
                    batch_size=8,
                    num_workers=4,
                    COMPUTE_LOSS=False,
                    use_deterministic_grid=None,
                    threshold=None,  # 0.02,
                    compute_kl_loss=False,
                    evaluate_train=False,
                    eval_datasplit_type=eval_datasplit_type,
                    val_repeat_factor=None,
                    psnr_type='range_invariant',
                    ignored_last_pixels=ignored_last_pixels,
                    ignore_first_pixels=0,
                    print_token=handler.dirpath(),
                    normalized_ssim=normalized_ssim,
                )
                fpath = handler.save(ckpt_dir, data)
                # except:
                #     print('FAILED for ', handler.get_output_fpath(ckpt_dir))
                #     continue
                print(handler.load(fpath))
                print('')
                print('')
                print('')


if __name__ == '__main__':
    DEBUG = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--grid_size', type=int, default=16)
    parser.add_argument('--hardcoded', action='store_true')
    parser.add_argument('--unnormalized_ssim', action='store_true')

    args = parser.parse_args()
    if args.hardcoded:
        print('Ignoring ckpt_dir,patch_size and grid_size')
        save_hardcoded_ckpt_evaluations_to_file(normalized_ssim=not args.unnormalized_ssim)
    else:
        mmse_count = 1
        ignored_last_pixels = 32 if os.path.basename(os.path.dirname(args.ckpt_dir)).split('-')[0][1:] == '3' else 0
        OUTPUT_DIR = ''
        eval_datasplit_type = DataSplitType.Test
        data = main(
            args.ckpt_dir,
            DEBUG,
            image_size_for_grid_centers=args.grid_size,
            mmse_count=mmse_count,
            custom_image_size=args.patch_size,
            batch_size=32,
            num_workers=4,
            COMPUTE_LOSS=False,
            use_deterministic_grid=None,
            threshold=None,  # 0.02,
            compute_kl_loss=False,
            evaluate_train=False,
            eval_datasplit_type=eval_datasplit_type,
            val_repeat_factor=None,
            psnr_type='range_invariant',
            ignored_last_pixels=ignored_last_pixels,
            ignore_first_pixels=0,
            normalized_ssim=args.normalized_ssim,
        )

        print('')
        print('Paper Related Stats')
        print('PSNR', np.mean(data['rangeinvpsnr']))
        print('SSIM', np.mean(data['ssim'][:2]))
