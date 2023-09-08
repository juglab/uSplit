import glob
import logging
import os
import pickle

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from usplit.core.data_split_type import DataSplitType
from usplit.core.data_type import DataType
from usplit.core.loss_type import LossType
from usplit.core.metric_monitor import MetricMonitor
from usplit.core.model_type import ModelType
from usplit.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from usplit.data_loader.lc_tiff_dloader import MultiScaleTiffDloader
from usplit.data_loader.lc_zarr_dloader import MultiScaleZarrDloader
from usplit.nets.model_utils import create_model
from usplit.training_utils import ValEveryNSteps


def create_dataset(config, datadir, raw_data_dict=None, skip_train_dataset=False):

    if config.data.data_type in [
            DataType.OptiMEM100_014, DataType.CustomSinosoid, DataType.CustomSinosoidThreeCurve,
            DataType.SeparateTiffData, DataType.SingleZarrData
    ]:
        if config.data.data_type == DataType.OptiMEM100_014:
            datapath = os.path.join(datadir, 'OptiMEM100x014.tif')
        else:
            datapath = datadir

        normalized_input = config.data.normalized_input
        use_one_mu_std = config.data.use_one_mu_std
        train_aug_rotate = config.data.train_aug_rotate
        enable_random_cropping = config.data.deterministic_grid is False
        lowres_supervision = config.model.model_type == ModelType.LadderVAEMultiTarget
        if 'multiscale_lowres_count' in config.data and config.data.multiscale_lowres_count is not None:
            padding_kwargs = {'mode': config.data.padding_mode}
            if 'padding_value' in config.data and config.data.padding_value is not None:
                padding_kwargs['constant_values'] = config.data.padding_value

            dataclass = MultiScaleZarrDloader if config.data.data_type == DataType.SingleZarrData else MultiScaleTiffDloader
            train_data = None if skip_train_dataset else dataclass(config.data,
                                                                   datapath,
                                                                   datasplit_type=DataSplitType.Train,
                                                                   val_fraction=config.training.val_fraction,
                                                                   test_fraction=config.training.test_fraction,
                                                                   normalized_input=normalized_input,
                                                                   use_one_mu_std=use_one_mu_std,
                                                                   enable_rotation_aug=train_aug_rotate,
                                                                   enable_random_cropping=enable_random_cropping,
                                                                   num_scales=config.data.multiscale_lowres_count,
                                                                   lowres_supervision=lowres_supervision,
                                                                   padding_kwargs=padding_kwargs,
                                                                   allow_generation=True)
            max_val = train_data.get_max_val()

            val_data = dataclass(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Val,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=False,  # No rotation aug on validation
                enable_random_cropping=False,
                # No random cropping on validation. Validation is evaluated on determistic grids
                num_scales=config.data.multiscale_lowres_count,
                lowres_supervision=lowres_supervision,
                padding_kwargs=padding_kwargs,
                allow_generation=False,
                max_val=max_val,
            )

        else:
            train_data_kwargs = {'allow_generation': True}
            val_data_kwargs = {'allow_generation': False}
            train_data_kwargs['enable_random_cropping'] = enable_random_cropping
            val_data_kwargs['enable_random_cropping'] = False
            data_class = MultiChDeterministicTiffDloader
            train_data = None if skip_train_dataset else data_class(config.data,
                                                                    datapath,
                                                                    datasplit_type=DataSplitType.Train,
                                                                    val_fraction=config.training.val_fraction,
                                                                    test_fraction=config.training.test_fraction,
                                                                    normalized_input=normalized_input,
                                                                    use_one_mu_std=use_one_mu_std,
                                                                    enable_rotation_aug=train_aug_rotate,
                                                                    **train_data_kwargs)

            max_val = train_data.get_max_val()
            val_data = data_class(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Val,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=False,  # No rotation aug on validation
                max_val=max_val,
                **val_data_kwargs,
            )

        # For normalizing, we should be using the training data's mean and std.
        mean_val, std_val = train_data.compute_mean_std()
        train_data.set_mean_std(mean_val, std_val)
        val_data.set_mean_std(mean_val, std_val)
    return train_data, val_data


def create_model_and_train(config, data_mean, data_std, logger, checkpoint_callback, train_loader, val_loader):
    # tensorboard previous files.
    for filename in glob.glob(config.workdir + "/events*"):
        os.remove(filename)

    # checkpoints
    for filename in glob.glob(config.workdir + "/*.ckpt"):
        os.remove(filename)

    model = create_model(config, data_mean, data_std)
    if config.model.model_type == ModelType.LadderVaeStitch2Stage:
        assert config.training.pre_trained_ckpt_fpath and os.path.exists(config.training.pre_trained_ckpt_fpath)

    if config.training.pre_trained_ckpt_fpath:
        print('Starting with pre-trained model', config.training.pre_trained_ckpt_fpath)
        checkpoint = torch.load(config.training.pre_trained_ckpt_fpath)
        _ = model.load_state_dict(checkpoint['state_dict'], strict=False)

    # print(model)
    estop_monitor = config.model.get('monitor', 'val_loss')
    estop_mode = MetricMonitor(estop_monitor).mode()

    callbacks = [
        EarlyStopping(monitor=estop_monitor,
                      min_delta=1e-6,
                      patience=config.training.earlystop_patience,
                      verbose=True,
                      mode=estop_mode),
        checkpoint_callback,
    ]
    if 'val_every_n_steps' in config.training and config.training.val_every_n_steps is not None:
        callbacks.append(ValEveryNSteps(config.training.val_every_n_steps))

    logger.experiment.config.update(config.to_dict())
    # wandb.init(config=config)
    if torch.cuda.is_available():
        # profiler = pl.profiler.AdvancedProfiler(output_filename=os.path.join(config.workdir, 'advance_profile.txt'))
        try:
            # older version has this code
            trainer = pl.Trainer(
                gpus=1,
                max_epochs=config.training.max_epochs,
                gradient_clip_val=None
                if model.automatic_optimization == False else config.training.grad_clip_norm_value,
                # gradient_clip_algorithm=config.training.gradient_clip_algorithm,
                logger=logger,
                # fast_dev_run=10,
                #  profiler=profiler,
                # overfit_batches=20,
                callbacks=callbacks,
                precision=config.training.precision)
        except:
            trainer = pl.Trainer(
                # gpus=1,
                max_epochs=config.training.max_epochs,
                gradient_clip_val=None
                if model.automatic_optimization == False else config.training.grad_clip_norm_value,
                # gradient_clip_algorithm=config.training.gradient_clip_algorithm,
                logger=logger,
                # fast_dev_run=10,
                #  profiler=profiler,
                # overfit_batches=20,
                callbacks=callbacks,
                precision=config.training.precision)

    else:
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            logger=logger,
            gradient_clip_val=config.training.grad_clip_norm_value,
            gradient_clip_algorithm=config.training.gradient_clip_algorithm,
            callbacks=callbacks,
            # fast_dev_run=10,
            # overfit_batches=10,
            precision=config.training.precision)
    trainer.fit(model, train_loader, val_loader)


def train_network(train_loader, val_loader, data_mean, data_std, config, model_name, logdir):
    ckpt_monitor = config.model.get('monitor', 'val_loss')
    ckpt_mode = MetricMonitor(ckpt_monitor).mode()
    checkpoint_callback = ModelCheckpoint(
        monitor=ckpt_monitor,
        dirpath=config.workdir,
        filename=model_name + '_best',
        save_last=True,
        save_top_k=1,
        mode=ckpt_mode,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = model_name + "_last"
    logger = WandbLogger(name=os.path.join(config.hostname, config.exptname),
                         save_dir=logdir,
                         project="Disentanglement")
    # logger = TensorBoardLogger(config.workdir, name="", version="", default_hp_metric=False)

    # pl.utilities.distributed.log.setLevel(logging.ERROR)
    posterior_collapse_count = 0
    collapse_flag = True
    while collapse_flag and posterior_collapse_count < 20:
        collapse_flag = create_model_and_train(config, data_mean, data_std, logger, checkpoint_callback, train_loader,
                                               val_loader)
        if collapse_flag is None:
            print('CTRL+C inturrupt. Ending')
            return

        if collapse_flag:
            posterior_collapse_count = posterior_collapse_count + 1

    if collapse_flag:
        print("Posterior collapse limit reached, attempting training with KL annealing turned on!")
        while collapse_flag:
            config.loss.kl_annealing = True
            collapse_flag = create_model_and_train(config, data_mean, data_std, logger, checkpoint_callback,
                                                   train_loader, val_loader)
            if collapse_flag is None:
                print('CTRL+C inturrupt. Ending')
                return


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    from usplit.configs.deepencoder_lvae_config import get_config

    config = get_config()
    train_data, val_data = create_dataset(config, '/group/jug/ashesh/data/microscopy/')

    dset = val_data
    idx = 0
    _, ax = plt.subplots(figsize=(9, 3), ncols=3)
    inp, target, alpha_val, ch1_idx, ch2_idx = dset[(idx, idx, 64, 19)]
    ax[0].imshow(inp[0])
    ax[1].imshow(target[0])
    ax[2].imshow(target[1])

    print(len(train_data), len(val_data))
    print(inp.mean(), target.mean())
