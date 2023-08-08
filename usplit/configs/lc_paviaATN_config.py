from tkinter.tix import Tree

import numpy as np

from usplit.configs.default_config import get_default_config
from usplit.core.data_type import DataType
from usplit.core.loss_type import LossType
from usplit.core.model_type import ModelType
from usplit.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 64
    data.data_type = DataType.OptiMEM100_014
    data.channel_1 = 2
    data.channel_2 = 3

    data.sampler_type = SamplerType.DefaultSampler
    data.deterministic_grid = False
    data.normalized_input = True
    data.clip_percentile = 0.995
    data.background_quantile = 0.0
    # With background quantile, one is setting the avg background value to 0. With this, any negative values are also set to 0.
    # This, together with correct background_quantile should altogether get rid of the background. The issue here is that
    # the background noise is also a distribution. So, some amount of background noise will remain.
    data.clip_background_noise_to_zero = False

    # we will not subtract the mean of the dataset from every patch. We just want to subtract the background and normalize using std. This way, background will be very close to 0.
    # this will help in the all scaling related approaches where we want to multiply the frame with some factor and then add them. we will then effectively just do these scaling on the
    # foreground pixels and the background will anyways will remain very close to 0.
    data.skip_normalization_using_mean = False

    data.input_is_sum = False

    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = True
    data.train_aug_rotate = False
    data.randomized_channels = False
    data.multiscale_lowres_count = 5
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = True

    # This is for intensity augmentation
    # data.ch1_min_alpha = 0.4
    # data.ch1_max_alpha = 0.55
    # data.return_alpha = True

    loss = config.loss
    loss.loss_type = LossType.Elbo
    # loss.mixed_rec_weight = 1

    loss.kl_weight = 1
    loss.kl_annealing = False
    loss.kl_annealtime = 10
    loss.kl_start = -1
    loss.kl_min = 1e-7
    loss.free_bits = 0.0
    # loss.ch1_recons_w = 1
    # loss.ch2_recons_w = 5

    model = config.model
    model.model_type = ModelType.LadderVae
    model.z_dims = [128, 128, 128, 128]

    model.encoder.batchnorm = True
    model.encoder.blocks_per_layer = 1
    model.encoder.n_filters = 64
    model.encoder.dropout = 0.1
    model.encoder.res_block_kernel = 3
    model.encoder.res_block_skip_padding = False

    model.decoder.batchnorm = True
    model.decoder.blocks_per_layer = 1
    model.decoder.n_filters = 64
    model.decoder.dropout = 0.1
    model.decoder.res_block_kernel = 3
    model.decoder.res_block_skip_padding = False

    model.decoder.multiscale_retain_spatial_dims = True
    config.model.decoder.conv2d_bias = True

    model.skip_nboundary_pixels_from_loss = None
    model.nonlin = 'elu'
    model.merge_type = 'residual'
    model.stochastic_skip = True
    model.learn_top_prior = True
    model.img_shape = None
    model.res_block_type = 'bacdbacd'

    model.gated = True
    model.no_initial_downscaling = True
    model.analytical_kl = False
    model.mode_pred = False
    model.var_clip_max = 20
    # predict_logvar takes one of the four values: [None,'global','channelwise','pixelwise']
    model.predict_logvar = 'pixelwise'
    model.logvar_lowerbound = -5  # -2.49 is log(1/12), from paper "Re-parametrizing VAE for stablity."
    model.multiscale_lowres_separate_branch = False
    model.multiscale_retain_spatial_dims = True
    model.monitor = 'val_psnr'  # {'val_loss','val_psnr'}
    model.non_stochastic_version = False
    model.enable_noise_model = False
    model.noise_model_ch1_fpath = None
    model.noise_model_ch1_fpath = None

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 15
    training.max_epochs = 200
    training.batch_size = 32
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1
    training.earlystop_patience = 100
    training.precision = 16

    return config
