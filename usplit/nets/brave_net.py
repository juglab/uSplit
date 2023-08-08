import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from usplit.core.metric_monitor import MetricMonitor
from usplit.metrics.running_psnr import RunningPSNR
from usplit.nets.brave_net_raw import BraveNet


class BraveNetPL(pl.LightningModule):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__()
        self.data_mean = torch.Tensor(data_mean) if isinstance(data_mean, np.ndarray) else data_mean
        self.data_std = torch.Tensor(data_std) if isinstance(data_std, np.ndarray) else data_std
        self.normalized_input = config.data.normalized_input
        self.model = BraveNet(config.model.num_kernels, config.model.kernel_size, 1, config.model.padding,
                              config.model.activation, config.model.dropout, config.model.batch_normalization,
                              config.model.final_activation)

        self.label1_psnr = RunningPSNR()
        self.label2_psnr = RunningPSNR()
        self.lr = config.training.lr
        self.lr_scheduler_patience = config.training.lr_scheduler_patience
        self.lr_scheduler_monitor = config.model.get('monitor', 'val_loss')
        self.lr_scheduler_mode = MetricMonitor(self.lr_scheduler_monitor).mode()

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=self.lr, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         self.lr_scheduler_mode,
                                                         patience=self.lr_scheduler_patience,
                                                         factor=0.5,
                                                         min_lr=1e-12,
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_scheduler_monitor}

    def reset_for_different_output_size(self, output_size):
        return None

    def normalize_input(self, x):
        if self.normalized_input:
            return x
        return (x - self.data_mean.mean()) / self.data_std.mean()

    def normalize_target(self, target):
        return (target - self.data_mean) / self.data_std

    def forward(self, x):
        inp = x[:, :1]
        lowres_inp = x[:, 1:2]
        return self.model(inp, lowres_inp)

    def set_params_to_same_device_as(self, correct_device_tensor):
        if isinstance(self.data_mean, torch.Tensor):
            if self.data_mean.device != correct_device_tensor.device:
                self.data_mean = self.data_mean.to(correct_device_tensor.device)
                self.data_std = self.data_std.to(correct_device_tensor.device)

    def get_reconstruction_loss(self, reconstruction, input):
        loss_fn = nn.MSELoss()
        return loss_fn(reconstruction, input)

    def compute_loss(self, out_array, target_normalized):
        loss_arr = [self.get_reconstruction_loss(out, target_normalized) for out in out_array]
        loss_primary = loss_arr[0]
        loss_lowres = 0
        for loss_tmp in loss_arr[1:]:
            loss_lowres += loss_tmp / len(loss_arr[1:])
        loss = (loss_primary + loss_lowres) / 2
        return {'loss': loss, 'loss_primary': loss_primary}

    def training_step(self, batch, batch_idx, enable_logging=True):
        x, target = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        out_array = self.forward(x_normalized)
        loss_dict = self.compute_loss(out_array, target_normalized)
        net_loss = loss_dict['loss']
        self.log('reconstruction_loss', loss_dict['loss_primary'], on_epoch=True)
        self.log('reconstruction_loss_total', net_loss, on_epoch=True)
        output = {
            'loss': net_loss,
            'reconstruction_loss': net_loss.detach(),
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        return output

    def validation_step(self, batch, batch_idx):
        x, target = batch
        self.set_params_to_same_device_as(target)
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        out_array = self.forward(x_normalized)
        loss_dict = self.compute_loss(out_array, target_normalized)
        self.log('val_loss', loss_dict['loss_primary'], on_epoch=True)
        self.log('val_loss_total', loss_dict['loss'], on_epoch=True)
        recons_img = out_array[0]
        self.label1_psnr.update(recons_img[:, 0], target_normalized[:, 0])
        self.label2_psnr.update(recons_img[:, 1], target_normalized[:, 1])

    def on_validation_epoch_end(self):
        psnrl1 = self.label1_psnr.get()
        psnrl2 = self.label2_psnr.get()
        psnr = (psnrl1 + psnrl2) / 2
        self.log('val_psnr', psnr, on_epoch=True)
        self.label1_psnr.reset()
        self.label2_psnr.reset()
