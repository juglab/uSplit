"""
This file defines the unet architecture.
"""

import numpy as np
import torch
import torch.nn as nn


def get_activation(activation_str):
    if activation_str == 'relu':
        return nn.ReLU()
    elif activation_str is None:
        return None
    else:
        raise Exception('Invalid activation string:', activation_str)


def merge_conv_block(last_num_channels):
    modules = []
    modules.append(
        convolution_layer(2 * last_num_channels,
                          last_num_channels,
                          1,
                          stride=1,
                          padding=0,
                          activation='relu',
                          dropout=0,
                          bn=False))

    modules.append(
        convolution_layer(last_num_channels,
                          last_num_channels,
                          1,
                          stride=1,
                          padding=0,
                          activation='relu',
                          dropout=0,
                          bn=False))
    return nn.Sequential(*modules)


def downscale_upscale_conv_block(in_channels, out_channels, kernel_size, strides, padding, activation, dropout, bn):
    modules = []
    modules.append(
        convolution_layer(in_channels,
                          out_channels,
                          kernel_size,
                          stride=strides,
                          padding=padding,
                          activation=activation,
                          dropout=dropout,
                          bn=bn))

    modules.append(
        convolution_layer(out_channels,
                          out_channels,
                          kernel_size,
                          stride=strides,
                          padding=padding,
                          activation=activation,
                          dropout=0,
                          bn=bn))

    return nn.Sequential(*modules)


def down_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout, bn):
    """
    define bottom up layers
    """
    blocks = nn.ModuleList([])
    input_ch_N = 1
    for ch_N in num_kernels:
        blocks.append(
            downscale_upscale_conv_block(input_ch_N, ch_N, kernel_size, strides, padding, activation, dropout, bn))
        input_ch_N = ch_N
    return blocks


def convolution_layer(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      padding=0,
                      activation=None,
                      dropout=0.0,
                      bn=True):
    branch = []
    branch.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))
    nonlin = get_activation(activation)
    if nonlin is not None:
        branch.append(nonlin)
    if bn:
        branch.append(nn.BatchNorm2d(out_channels))
    if dropout > 0:
        branch.append(nn.Dropout(p=dropout))

    return nn.Sequential(*branch)


def lowres_output_branches(num_kernels, final_activation, dropout):
    blocks = nn.ModuleList([])
    N = len(num_kernels)
    for i in range(N - 2):
        branch = convolution_layer(
            num_kernels[N - i - 2],
            2,
            1,
            stride=1,
            padding=0,  #TODO: check
            activation=final_activation,
            dropout=dropout,
            bn=False)  #TODO: check this
        blocks.append(branch)
    return blocks


def up_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout, bn):
    blocks = nn.ModuleList([])
    input_ch_N = num_kernels[-1]
    for i in range(len(num_kernels) - 1):
        out_ch_N = num_kernels[len(num_kernels) - i - 2]
        blocks.append(
            downscale_upscale_conv_block(2 * input_ch_N, out_ch_N, kernel_size, strides, padding, activation, dropout,
                                         bn))
        input_ch_N = out_ch_N
    return blocks


class BraveNet(nn.Module):

    def __init__(self, num_kernels, kernel_size, strides, padding, activation, dropout, bn, final_activation):
        super().__init__()
        self.num_kernels = num_kernels
        self.input_bn = nn.BatchNorm2d(1)
        self.lowres_input_bn = nn.BatchNorm2d(1)

        self.bottom_up_layers = down_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout, bn)
        self.lowres_bottom_up_layers = down_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout,
                                                       bn)

        # Merging bu layer output with lowres bu layer output
        self.merge_block = merge_conv_block(num_kernels[-1])
        self.lowres_output_branches = lowres_output_branches(num_kernels, final_activation, dropout)
        self.output_branch = convolution_layer(num_kernels[0],
                                               2,
                                               1,
                                               stride=1,
                                               activation=final_activation,
                                               dropout=dropout,
                                               padding=0,
                                               bn=False)
        self.num_kernels = num_kernels
        self.top_down_layers = up_scale_path(num_kernels, kernel_size, strides, padding, activation, dropout, bn)

    def bottom_up(self, input, bu_layers):
        residuals = {}
        conv_down = input
        for i in range(len(self.num_kernels)):
            # level i
            conv_down = bu_layers[i](conv_down)
            residuals[f"conv_{i}"] = conv_down
            if i < len(self.num_kernels) - 1:
                conv_down = nn.MaxPool2d(2, stride=2)(conv_down)

        return conv_down, residuals

    def top_down(self, bu_output, residuals, output_dim):
        """
        Returns a list of predictions.
        first element will be the primary output.
        """
        outputs = []
        conv_up = bu_output
        for i in range(len(self.num_kernels) - 1):
            conv_up = nn.Upsample(scale_factor=2, mode='nearest')(conv_up)
            bu_tensor = residuals["conv_" + str(len(self.num_kernels) - i - 2)]
            conv_up = torch.cat([conv_up, bu_tensor], dim=1)
            conv_up = self.top_down_layers[i](conv_up)
            if i < len(self.num_kernels) - 2:
                temp_output = nn.Upsample(size=output_dim, mode='nearest')(conv_up)
                temp_output = self.lowres_output_branches[i](temp_output)
                outputs.append(temp_output)

        output = self.output_branch(conv_up)
        outputs.append(output)
        return outputs[::-1]

    def get_merged_residuals(self, bu_res, lr_bu_res):
        ### CONCAT/PREPARE RESIDUALS
        merged_residuals = {}
        for key in bu_res.keys():
            merged_residuals[key] = torch.cat([bu_res[key], lr_bu_res[key]], dim=1)
        return merged_residuals

    def forward(self, input, lowres_input):
        output_dim = input.shape[-2:]
        input = self.input_bn(input)
        lowres_input = self.lowres_input_bn(lowres_input)

        bu_out, bu_res = self.bottom_up(input, self.bottom_up_layers)
        lr_bu_out, lr_bu_res = self.bottom_up(lowres_input, self.lowres_bottom_up_layers)
        bu_out = torch.cat([bu_out, lr_bu_out], dim=1)
        bu_out = self.merge_block(bu_out)
        residuals = self.get_merged_residuals(bu_res, lr_bu_res)
        outputs = self.top_down(bu_out, residuals, output_dim)
        return outputs


if __name__ == '__main__':
    num_kernels = [32, 64, 128, 256]
    kernel_size = 3
    padding = 1
    activation = 'relu'
    final_activation = 'relu'
    dropout = 0.1
    bn = True
    strides = 1
    model = BraveNet(num_kernels, kernel_size, strides, padding, activation, dropout, bn)
    inp = torch.randn(5, 1, 64, 64)
    lowres_inp = torch.randn(5, 1, 64, 64)
    out = model(inp, lowres_inp)
    import pdb
    pdb.set_trace()
    # print(model)
