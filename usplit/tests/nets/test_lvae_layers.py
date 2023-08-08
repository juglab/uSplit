import torch
import torch.nn as nn

from usplit.nets.lvae_layers import TopDownLayer


def test_pixel_intensity_invariance():
    """
    Ensure the following constraint: f(10*x) = 10*f(x)
    Here, f is the TopDownLayer
    """
    res_block_type = 'bacdbacd'
    res_block_kernel = 3
    res_block_skip_padding = False
    gated = False
    conv2d_bias = False
    z_dim = 64
    n_res_blocks = 2
    n_filters = 64
    is_top_layer = False
    downsampling_steps = 1
    nonlin = nn.LeakyReLU
    merge_type = 'residual_ungated'
    batchnorm = False
    dropout = 0.0
    stochastic_skip = True
    groups = 1
    learn_top_prior = True
    analytical_kl = False
    top_prior_param_shape = (1, 128, 8, 8)
    bottomup_no_padding_mode = False
    topdown_no_padding_mode = False
    retain_spatial_dims = False
    non_stochastic_version = True
    input_image_shape = (64, 64)
    normalize_latent_factor = 1

    td_block = TopDownLayer(
        z_dim,
        n_res_blocks,
        n_filters,
        is_top_layer=is_top_layer,
        downsampling_steps=downsampling_steps,
        nonlin=nonlin,
        merge_type=merge_type,
        batchnorm=batchnorm,
        dropout=dropout,
        stochastic_skip=stochastic_skip,
        res_block_type=res_block_type,
        res_block_kernel=res_block_kernel,
        res_block_skip_padding=res_block_skip_padding,
        groups=groups,
        gated=gated,
        learn_top_prior=learn_top_prior,
        top_prior_param_shape=top_prior_param_shape,
        analytical_kl=analytical_kl,
        bottomup_no_padding_mode=bottomup_no_padding_mode,
        topdown_no_padding_mode=topdown_no_padding_mode,
        retain_spatial_dims=retain_spatial_dims,
        input_image_shape=input_image_shape,
        normalize_latent_factor=normalize_latent_factor,
        non_stochastic_version=non_stochastic_version,
        conv2d_bias=conv2d_bias,
    )
    with torch.no_grad():
        out = torch.rand(16, 64, 8, 8)
        skip_input = out
        inference_mode = True
        bu_value = torch.rand(16, 64, 8, 8)
        n_img_prior = None
        use_mode = True
        force_constant_output = None
        forced_latent = None
        mode_pred = False
        use_uncond_mode = False
        var_clip_max = None

        td_out1 = td_block(out,
                           skip_connection_input=skip_input,
                           inference_mode=inference_mode,
                           bu_value=bu_value,
                           n_img_prior=n_img_prior,
                           use_mode=use_mode,
                           force_constant_output=force_constant_output,
                           forced_latent=forced_latent,
                           mode_pred=mode_pred,
                           use_uncond_mode=use_uncond_mode,
                           var_clip_max=var_clip_max)

        td_out2 = td_block(out * 10,
                           skip_connection_input=skip_input * 10,
                           inference_mode=inference_mode,
                           bu_value=bu_value * 10,
                           n_img_prior=n_img_prior,
                           use_mode=use_mode,
                           force_constant_output=force_constant_output,
                           forced_latent=forced_latent,
                           mode_pred=mode_pred,
                           use_uncond_mode=use_uncond_mode,
                           var_clip_max=var_clip_max)

        assert (td_out1[0] * 10 - td_out2[0]).abs().max().item() < 1e-5
