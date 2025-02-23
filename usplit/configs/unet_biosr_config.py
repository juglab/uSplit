from usplit.configs.default_config import get_default_config
from usplit.core.data_type import DataType
from usplit.core.loss_type import LossType
from usplit.core.model_type import ModelType
from usplit.core.sampler_type import SamplerType


def get_config():
    config = get_default_config()
    data = config.data
    data.image_size = 512
    data.data_type = DataType.BioSR
    # data.channel_1 = 0
    # data.channel_2 = 1
    # data.ch1_fname = 'actin-60x-noise2-highsnr.tif'
    # data.ch2_fname = 'mito-60x-noise2-highsnr.tif'

    data.sampler_type = SamplerType.DefaultSampler
    data.threshold = 0.02
    data.deterministic_grid = False
    data.normalized_input = True
    data.clip_percentile = 1.0
    # If this is set to true, then one mean and stdev is used for both channels. Otherwise, two different
    # meean and stdev are used.
    data.use_one_mu_std = True
    data.train_aug_rotate = False
    data.randomized_channels = False
    data.multiscale_lowres_count = None
    data.padding_mode = 'reflect'
    data.padding_value = None
    # If this is set to True, then target channels will be normalized from their separate mean.
    # otherwise, target will be normalized just the same way as the input, which is determined by use_one_mu_std
    data.target_separate_normalization = True

    loss = config.loss
    loss.loss_type = LossType.MSE
    # loss.mixed_rec_weight = 1

    model = config.model
    model.model_type = ModelType.UNet
    model.n_levels = 5
    model.init_channel_count = 32
    model.enable_context_transfer = False
    model.context_transfer_initial_weight_factor = 0
    model.multiscale_lowres_separate_branch = True
    model.monitor = 'val_psnr'

    training = config.training
    training.lr = 0.001
    training.lr_scheduler_patience = 30
    training.max_epochs = 400
    training.batch_size = 8
    training.num_workers = 4
    training.val_repeat_factor = None
    training.train_repeat_factor = None
    training.val_fraction = 0.1
    training.test_fraction = 0.1
    training.earlystop_patience = 200
    # training.precision = 16

    return config
