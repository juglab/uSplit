""" 
Adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from usplit.core.metric_monitor import MetricMonitor
from usplit.metrics.running_psnr import RunningPSNR
from usplit.nets.context_transfer_module import ContextTransferModule
from usplit.nets.lvae_layers import BottomUpDeterministicResBlock, MergeLowRes
from usplit.nets.unet_parts import *


class UNet(pl.LightningModule):

    def __init__(self, data_mean, data_std, config):
        super(UNet, self).__init__()
        bilinear = True
        self.bilinear = bilinear
        self.lr = config.training.lr
        self.n_levels = config.model.n_levels
        self.lr_scheduler_patience = config.training.lr_scheduler_patience
        self.lr_scheduler_monitor = config.model.get('monitor', 'val_loss')
        self.lr_scheduler_mode = MetricMonitor(self.lr_scheduler_monitor).mode()
        self.enable_context_transfer = config.model.get('enable_context_transfer', False)
        self.ct_modules = nn.ModuleList()
        init_ch = config.model.get('init_channel_count', 64)
        self.multiscale_lowres_separate_branch = config.model.multiscale_lowres_separate_branch
        self._img_sz = config.data.image_size

        if self.enable_context_transfer:
            hw = config.data.image_size
            cur_ch = init_ch
            for i in range(1, self.n_levels + 1):
                self.ct_modules.append(
                    ContextTransferModule((cur_ch, hw, hw),
                                          initial_weight_factor=config.model.context_transfer_initial_weight_factor))
                cur_ch *= 2
                hw //= 2

        self.inc = DoubleConv(1, init_ch)
        ch = init_ch
        for i in range(1, self.n_levels):
            setattr(self, f'down{i}', Down(ch, 2 * ch))
            ch = 2 * ch

        factor = 2 if bilinear else 1
        setattr(self, f'down{self.n_levels}', Down(ch, 2 * ch // factor))
        ch = 2 * ch
        for i in range(1, self.n_levels):
            setattr(self, f'up{i}', Up(ch, (ch // 2) // factor, bilinear))
            ch = ch // 2

        setattr(self, f'up{self.n_levels}', Up(ch, ch // 2, bilinear))
        ch = ch // 2
        self.outc = OutConv(ch, 2)

        # multiscale architecture
        self.lowres_first_bottom_ups = self._multiscale_count = self.lowres_merge = self.lowres_net = None
        self._init_multires(config, init_ch)

        self.normalized_input = config.data.normalized_input
        self.data_mean = torch.Tensor(data_mean) if isinstance(data_mean, np.ndarray) else data_mean
        self.data_std = torch.Tensor(data_std) if isinstance(data_std, np.ndarray) else data_std
        self.label1_psnr = RunningPSNR()
        self.label2_psnr = RunningPSNR()
        print(
            f'[{self.__class__.__name__}] ContextTransfer:{self.enable_context_transfer} SepBranch:{self.multiscale_lowres_separate_branch}'
        )

    def reset_for_different_output_size(self, output_size):
        assert self._img_sz == output_size, f"{self._img_sz}!={output_size}. This model does not support different output size due to context transfer module"

    def _init_multires(self, config, init_n_filters):
        """
        Initialize everything related to multiresolution approach.
        """
        self.batchnorm = True
        # self.encoder_n_filters = 34
        multiscale_retain_spatial_dims = True
        res_block_type = 'bacdbacd'
        res_block_skip_padding = False
        # assuming no initial downscaling. otherwise it will be 2
        stride = 1
        nonlin = nn.ELU
        self._multiscale_count = config.data.multiscale_lowres_count
        if self._multiscale_count is None:
            self._multiscale_count = 1

        msg = "Multiscale count({}) should not exceed the number of bottom up layers ({}) by more than 1"
        msg = msg.format(config.data.multiscale_lowres_count, config.model.n_levels)
        assert self._multiscale_count <= 1 or config.data.multiscale_lowres_count <= 1 + config.model.n_levels, msg

        # msg = "if multiscale is enabled, then we are just working with monocrome images."
        # assert self._multiscale_count == 1 or self.color_ch == 1, msg
        lowres_first_bottom_up_list = []
        lowres_merge_list = []
        lowres_net_list = []

        multiscale_lowres_size_factor = 1
        n_filters = init_n_filters
        for i in range(1, self._multiscale_count):
            layer_enable_multiscale = self._multiscale_count > i + 1
            multiscale_lowres_size_factor *= (1 + int(layer_enable_multiscale))

            first_bottom_up = nn.Sequential(
                nn.Conv2d(1, n_filters, 5, padding=2, stride=stride), nonlin(),
                BottomUpDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    nonlin=nonlin,
                    batchnorm=self.batchnorm,
                    dropout=0,
                    res_block_type=res_block_type,
                    skip_padding=res_block_skip_padding,
                ))
            lowres_first_bottom_up_list.append(first_bottom_up)
            lowres_merge = MergeLowRes(channels=2 * n_filters,
                                       merge_type='residual',
                                       nonlin=nonlin,
                                       batchnorm=self.batchnorm,
                                       dropout=0,
                                       res_block_type=res_block_type,
                                       multiscale_retain_spatial_dims=multiscale_retain_spatial_dims,
                                       multiscale_lowres_size_factor=multiscale_lowres_size_factor)

            lowres_merge_list.append(lowres_merge)

            net = getattr(self, f'down{i}')
            net = net.maxpool_conv[1]  # skipping the maxpool
            if self.multiscale_lowres_separate_branch:
                net = deepcopy(net)
            lowres_net_list.append(net)

            n_filters = 2 * n_filters

        self.lowres_net = nn.ModuleList(lowres_net_list) if len(lowres_net_list) else None
        self.lowres_first_bottom_ups = nn.ModuleList(lowres_first_bottom_up_list) if len(
            lowres_first_bottom_up_list) else None

        self.lowres_merge = nn.ModuleList(lowres_merge_list) if len(lowres_merge_list) else None

    def forward(self, x):
        if self._multiscale_count == 1:
            x1 = self.inc(x)
        else:
            x1 = self.inc(x[:, :1])

        latents = []
        x_end = x1
        latents.append(x1)
        for i in range(1, self.n_levels + 1):
            x_end = getattr(self, f'down{i}')(x_end)

            if i < self._multiscale_count:
                lowres_x = self.lowres_first_bottom_ups[i - 1](x[:, i:i + 1])
                # lowres_net = getattr(self, f'down{i}')
                # lowres_net = lowres_net.maxpool_conv[1]  # skipping the maxpool
                lowres_flow = self.lowres_net[i - 1](lowres_x)
                x_end = self.lowres_merge[i - 1](x_end, lowres_flow)

            latents.append(x_end)

        if self.enable_context_transfer:
            for i in range(len(latents) - 1):
                latents[i] = self.ct_modules[i](latents[i])

        for i in range(1, self.n_levels + 1):
            x_end = getattr(self, f'up{i}')(x_end, latents[-1 * (i + 1)])
            if x_end.shape[-1] > x.shape[-1]:
                x_end = F.center_crop(x_end, x.shape[-2:])

        pred = self.outc(x_end)
        return pred

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=self.lr, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         self.lr_scheduler_mode,
                                                         patience=self.lr_scheduler_patience,
                                                         factor=0.5,
                                                         min_lr=1e-12,
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_scheduler_monitor}

    def normalize_input(self, x):
        if self.normalized_input:
            return x
        return (x - self.data_mean.mean()) / self.data_std.mean()

    def normalize_target(self, target):
        return (target - self.data_mean) / self.data_std

    def power_of_2(self, x):
        assert isinstance(x, int)
        if x == 1:
            return True
        if x == 0:
            # happens with validation
            return False
        if x % 2 == 1:
            return False
        return self.power_of_2(x // 2)

    def set_params_to_same_device_as(self, correct_device_tensor):
        if self.enable_context_transfer:
            for i in range(len(self.ct_modules)):
                self.ct_modules[i].set_params_to_same_device_as(correct_device_tensor)

        if isinstance(self.data_mean, torch.Tensor):
            if self.data_mean.device != correct_device_tensor.device:
                self.data_mean = self.data_mean.to(correct_device_tensor.device)
                self.data_std = self.data_std.to(correct_device_tensor.device)

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out = self.forward(x_normalized)
        net_loss = self.get_reconstruction_loss(out, target_normalized)

        self.log('reconstruction_loss', net_loss, on_epoch=True)

        output = {
            'loss': net_loss,
            'reconstruction_loss': net_loss,
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def get_reconstruction_loss(self, reconstruction, input):
        loss_fn = nn.MSELoss()
        return loss_fn(reconstruction, input)

    def validation_step(self, batch, batch_idx):
        x, target = batch
        self.set_params_to_same_device_as(target)

        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out = self.forward(x_normalized)
        recons_img = out
        recons_loss = self.get_reconstruction_loss(out, target_normalized)

        self.log('val_loss', recons_loss, on_epoch=True)
        self.label1_psnr.update(recons_img[:, 0], target_normalized[:, 0])
        self.label2_psnr.update(recons_img[:, 1], target_normalized[:, 1])

        if batch_idx == 0 and self.power_of_2(self.current_epoch):
            sample = self(x_normalized[0:1, ...])

            sample = sample * self.data_std + self.data_mean
            sample = sample.cpu()
            self.log_images_for_tensorboard(sample[:, 0, ...], target[0, 0, ...], 'label1')
            self.log_images_for_tensorboard(sample[:, 1, ...], target[0, 1, ...], 'label2')

    def log_images_for_tensorboard(self, pred, target, label):
        clamped_pred = torch.clamp((pred - pred.min()) / (pred.max() - pred.min()), 0, 1)
        if target is not None:
            clamped_input = torch.clamp((target - target.min()) / (target.max() - target.min()), 0, 1)
            img = wandb.Image(clamped_input[None].cpu().numpy())
            self.logger.experiment.log({f'target_for{label}': img})
            # self.trainer.logger.experiment.add_image(f'target_for{label}', clamped_input[None], self.current_epoch)

        img = wandb.Image(clamped_pred.cpu().numpy())
        self.logger.experiment.log({f'{label}/sample_0': img})

    def on_validation_epoch_end(self):
        psnrl1 = self.label1_psnr.get()
        psnrl2 = self.label2_psnr.get()
        psnr = (psnrl1 + psnrl2) / 2
        self.log('val_psnr', psnr, on_epoch=True)
        self.label1_psnr.reset()
        self.label2_psnr.reset()


if __name__ == '__main__':
    from usplit.configs.unet_config import get_config
    cnf = get_config()
    model = UNet(0.0, 1.0, cnf)
    # print(model)G
    inp = torch.rand((12, 4, 64, 64))
    model(inp)
