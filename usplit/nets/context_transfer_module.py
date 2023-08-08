"""
Context Transfer module coded following https://www.researchgate.net/publication/331159375_Context-Aware_U-Net_for_Biomedical_Image_Segmentation
"""
import torch
import torch.nn as nn


class ContextTransferModule(nn.Module):

    def __init__(self, tensor_shape, initial_weight_factor=0):
        super().__init__()
        self.C, self.H, self.W = tensor_shape
        # UP, DOWN, LEFT, RIGHT
        self.ct_weights = nn.Parameter(initial_weight_factor * torch.ones((4, self.H, self.W)), requires_grad=True)
        self.final_layer = nn.Sequential(nn.Conv2d(4 * self.C, self.C, 1, padding=0), nn.ReLU(inplace=False))
        print(f'[{self.__class__.__name__}] {tensor_shape} {initial_weight_factor}')

    def set_params_to_same_device_as(self, correct_device_tensor):
        if isinstance(self.ct_weights, torch.Tensor):
            if self.ct_weights.device != correct_device_tensor.device:
                self.ct_weights = self.ct_weights.to(correct_device_tensor.device)

    def get_up_W(self):
        return torch.sigmoid(self.ct_weights[0])

    def get_down_W(self):
        return torch.sigmoid(self.ct_weights[1])

    def get_left_W(self):
        return torch.sigmoid(self.ct_weights[2])

    def get_right_W(self):
        return torch.sigmoid(self.ct_weights[3])

    def up_context(self, inp):
        out = inp.clone()
        assert out.shape[1] == self.C
        assert out.shape[2] == self.H
        assert out.shape[3] == self.W
        w = self.get_up_W()
        for i in range(1, self.H):
            old_version = out[:, :, i].clone()
            new_version = w[i - 1] * out[:, :, i - 1].clone() + old_version
            new_version[new_version < 0] = 0
            out[:, :, i] = new_version
        return out

    def down_context(self, inp):
        out = inp.clone()
        assert out.shape[1] == self.C
        assert out.shape[2] == self.H
        assert out.shape[3] == self.W
        w = self.get_down_W()
        rel_idx = -1
        for i in range(self.H - 2, -1, -1):
            old_version = out[:, :, i].clone()
            new_version = w[i - rel_idx] * out[:, :, i - rel_idx].clone() + old_version
            new_version[new_version < 0] = 0
            out[:, :, i] = new_version
        return out

    def right_context(self, inp):
        out = inp.clone()
        assert out.shape[1] == self.C
        assert out.shape[2] == self.H
        assert out.shape[3] == self.W
        w = self.get_right_W()
        rel_idx = -1
        for i in range(self.W - 2, -1, -1):
            old_version = out[:, :, :, i].clone()
            new_version = w[:, i - rel_idx] * out[:, :, :, i - rel_idx].clone() + old_version
            new_version[new_version < 0] = 0
            out[:, :, :, i] = new_version
        return out

    def left_context(self, inp):
        out = inp.clone()
        assert out.shape[1] == self.C
        assert out.shape[2] == self.H
        assert out.shape[3] == self.W
        w = self.get_left_W()
        rel_idx = 1
        for i in range(1, self.W):
            old_version = out[:, :, :, i].clone()
            new_version = w[:, i - rel_idx] * out[:, :, :, i - rel_idx].clone() + old_version
            new_version[new_version < 0] = 0
            out[:, :, :, i] = new_version
        return out

    def forward(self, inp):
        lc = self.left_context(inp)
        rc = self.right_context(inp)
        uc = self.up_context(inp)
        dc = self.down_context(inp)
        context = torch.cat([lc, rc, uc, dc], dim=1)
        return self.final_layer(context)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # from usplit.nets.context_transfer_module import ContextTransferModule

    shape = (64, 128, 128)
    cxt = ContextTransferModule(shape, initial_weight_factor=10)
    inp = torch.zeros((2, *shape))
    # inp[:, :, :1] = 1
    inp[:, :, -1:] = 1
    # inp[:, :, :, :1] = 1
    # inp[:, :, :, -1:] = 1

    # out = cxt(inp).detach().cpu().numpy()
    out = cxt.down_context(inp).detach().cpu().numpy()
    # out = out / out.max()
    _, ax = plt.subplots(figsize=(8, 4), ncols=2)
    sns.heatmap(inp[0, 0], ax=ax[0])
    sns.heatmap(np.log(out[0, 0] + 1), ax=ax[1])
    # import pdb;pdb.set_trace()
    # out1 = cxt(inp)
    # import pdb
    # pdb.set_trace()
