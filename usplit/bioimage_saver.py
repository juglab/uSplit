# import sys
# path = '/home/ashesh.ashesh/code/Disentangle'
# while path in sys.path:
#     sys.path.remove(path)

# sys.path.append('/home/ashesh.ashesh/code/uSplit/')

import os
import hashlib

# the imports for bioimage.io model export
import pytorch_lightning as pl
import bioimageio.core
import numpy as np
import torch
import torch.nn as nn
from bioimageio.core.build_spec import build_model, add_weights

root_bioimage_output_dir = os.path.expanduser("~/bioimage_usplit_models")
root_ckpt_dir = os.path.expanduser("~/paper_models")
if '/' != root_ckpt_dir[-1]:
    root_ckpt_dir = root_ckpt_dir + '/'

data_dir = '/group/jug/ashesh/data/microscopy/'
ckpt_dir = "/home/ashesh.ashesh/paper_models/PaviaATN/TubNuc/DeepLC/"
assert ckpt_dir[:len(root_ckpt_dir)] == root_ckpt_dir
outputdir = os.path.join(root_bioimage_output_dir, ckpt_dir[len(root_ckpt_dir):])
os.makedirs(outputdir, exist_ok=True)
print(outputdir)
print('')

import random
import os
import numpy as np
import torch
import pickle
import ml_collections
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from usplit.training import create_dataset, create_model
import matplotlib.pyplot as plt
from usplit.core.loss_type import LossType
from usplit.config_utils import load_config
from usplit.analysis.lvae_utils import get_img_from_forward_output
from usplit.analysis.plot_utils import clean_ax
from usplit.core.data_type import DataType
from usplit.core.psnr import PSNR
from usplit.analysis.plot_utils import get_k_largest_indices,plot_imgs_from_idx
from usplit.core.psnr import PSNR, RangeInvariantPsnr
from usplit.core.data_split_type import DataSplitType
from usplit.analysis.stitch_prediction import stitch_predictions
from usplit.analysis.mmse_prediction import get_dset_predictions


torch.multiprocessing.set_sharing_strategy('file_system')

image_size_for_grid_centers = 32
mmse_count = 1
custom_image_size = 64



batch_size = 32
num_workers = 4
use_deterministic_grid = None
threshold = None # 0.02
compute_kl_loss = False
evaluate_train = False# inspect training performance
eval_datasplit_type = DataSplitType.Test
val_repeat_factor = None

def get_best_checkpoint(ckpt_dir):
    output = []
    for filename in glob.glob(ckpt_dir + "/*_best.ckpt"):
        output.append(filename)
    assert len(output) == 1, '\n'.join(output)
    return output[0]

from usplit.core.model_type import ModelType
config = load_config(ckpt_dir)
config = ml_collections.ConfigDict(config)
old_image_size = None
with config.unlocked():
    if 'test_fraction' not in config.training:
        config.training.test_fraction =0.0
        
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
    config.data.train_aug_rotate=False
    
    if 'randomized_channels' not in config.data:
        config.data.randomized_channels = False
        
    if 'predict_logvar' not in config.model:
        config.model.predict_logvar=None
    
    if 'batchnorm' in config.model and 'batchnorm' not in config.model.encoder:
        assert 'batchnorm' not in config.model.decoder
        config.model.decoder.batchnorm = config.model.batchnorm
        config.model.encoder.batchnorm = config.model.batchnorm
    if 'conv2d_bias' not in config.model.decoder:
        config.model.decoder.conv2d_bias = True
        
    
    if custom_image_size is not None:
        old_image_size = config.data.image_size
        config.data.image_size = custom_image_size
    if image_size_for_grid_centers is not None:
        old_grid_size = config.data.get('grid_size', "grid_size not present")
        config.data.grid_size = image_size_for_grid_centers
        config.data.val_grid_size = image_size_for_grid_centers

    if use_deterministic_grid is not None:
        config.data.deterministic_grid = use_deterministic_grid
    if threshold is not None:
        config.data.threshold = threshold
    if val_repeat_factor is not None:
        config.training.val_repeat_factor = val_repeat_factor
    config.model.mode_pred = not compute_kl_loss

    config.model.skip_nboundary_pixels_from_loss = None
    if config.model.model_type == ModelType.UNet and 'n_levels' not in config.model:
        config.model.n_levels = 4
    
    if config.model.model_type == ModelType.UNet and 'init_channel_count' not in config.model:
        config.model.init_channel_count = 64
    
    if 'skip_receptive_field_loss_tokens' not in config.loss:
        config.loss.skip_receptive_field_loss_tokens = []
    
    if 'lowres_merge_type' not in config.model.encoder:
        config.model.encoder.lowres_merge_type = 0
print(config)


from usplit.data_loader.multi_channel_determ_tiff_dloader import MultiChDeterministicTiffDloader
from usplit.data_loader.lc_tiff_dloader import MultiScaleTiffDloader
from usplit.core.data_split_type import DataSplitType
from usplit.data_loader.patch_index_manager import GridAlignement

padding_kwargs = {
    'mode':config.data.get('padding_mode','constant'),
}

if padding_kwargs['mode'] == 'constant':
    padding_kwargs['constant_values'] = config.data.get('padding_value',0)

dloader_kwargs = {'overlapping_padding_kwargs':padding_kwargs}
if 'multiscale_lowres_count' in config.data and config.data.multiscale_lowres_count is not None:
    data_class = MultiScaleTiffDloader
    dloader_kwargs['num_scales'] = config.data.multiscale_lowres_count
    dloader_kwargs['padding_kwargs'] = padding_kwargs
else:
    data_class = MultiChDeterministicTiffDloader

if config.data.data_type in [DataType.CustomSinosoid, DataType.CustomSinosoidThreeCurve, 
                             DataType.SeparateTiffData,
                            ]:
    datapath = data_dir
elif config.data.data_type == DataType.OptiMEM100_014:
    datapath = os.path.join(data_dir, 'OptiMEM100x014.tif')
else:
    raise NotImplementedError(config.data.data_type)

normalized_input = config.data.normalized_input
use_one_mu_std = config.data.use_one_mu_std
train_aug_rotate = config.data.train_aug_rotate
enable_random_cropping = False
grid_alignment = GridAlignement.Center
print(data_class)

train_dset = data_class(
                config.data,
                datapath,
                datasplit_type=DataSplitType.Train,
                val_fraction=config.training.val_fraction,
                test_fraction=config.training.test_fraction,
                normalized_input=normalized_input,
                use_one_mu_std=use_one_mu_std,
                enable_rotation_aug=train_aug_rotate,
                enable_random_cropping=enable_random_cropping,
                grid_alignment=grid_alignment,
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
                grid_alignment=grid_alignment,
                max_val=max_val,
                **dloader_kwargs
                
            )

# For normalizing, we should be using the training data's mean and std.
mean_val, std_val = train_dset.compute_mean_std()
train_dset.set_mean_std(mean_val, std_val)
val_dset.set_mean_std(mean_val, std_val)


if evaluate_train:
    val_dset = train_dset
data_mean, data_std = train_dset.get_mean_std()



with config.unlocked():
    if old_image_size is not None:
        config.data.image_size = old_image_size

if config.data.target_separate_normalization is True:
    mean_fr_model, std_fr_model = train_dset.compute_individual_mean_std()
else:
    mean_fr_model, std_fr_model = train_dset.get_mean_std()


model = create_model(config, mean_fr_model,std_fr_model)

ckpt_fpath = get_best_checkpoint(ckpt_dir)
checkpoint = torch.load(ckpt_fpath)

_ = model.load_state_dict(checkpoint['state_dict'])
# model.eval()
# just a placeholder since model will be overwritten.
pl_model = model
# _= model.cuda()

# model.set_params_to_same_device_as(torch.Tensor(1).cuda())

print('Loading from epoch', checkpoint['epoch'])

# a very simple pytorch model: just a few convolutions
# model = nn.Sequential(
#     nn.Conv2d(1, 16, 3),
#     nn.Conv2d(16, 32, 3),
#     nn.Conv2d(32, 16, 3),
#     nn.Conv2d(16, 1, 1)
# )

class PredictiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
    def forward(self, x):
        out, _ =self.model(x)
        return self.model.likelihood.parameter_net(out)

input_ = val_dset[0][0][None]  # an example input
# predmodel = PredictiveModel()
# modelscript = predmodel.to_torchscript()
modelscript = torch.jit.trace(PredictiveModel, (input_))

# save the model weights
bioimage_ckpt_fpath = os.path.join(outputdir, 'model.pt')
torch.jit.save(modelscript, bioimage_ckpt_fpath)

bioimage_testinput_fpath = os.path.join(outputdir, 'test-input.npy')
bioimage_testoutput_fpath = os.path.join(outputdir, 'test-output.npy')
np.save(bioimage_testinput_fpath, input_)
with torch.no_grad():
    output = modelscript(torch.from_numpy(input_)).numpy()
np.save(bioimage_testoutput_fpath, output)
bioimage_doc_fpath = os.path.join(outputdir, 'doc.md')
with open(bioimage_doc_fpath, "w") as f:
    f.write("# My First Model\n")
    f.write("This model was trained on a very big dataset.\n")
    f.write("You should not let it get wet or feed it after midnight.\n")
    f.write("To validate its predictins, make sure that it does not produce any evil clones.\n")

build_model(
    # the weight file and the type of the weights
    weight_uri=bioimage_ckpt_fpath,
    weight_type="torchscript",
    # the test input and output data as well as the description of the tensors
    # these are passed as list because we support multiple inputs / outputs per model
    test_inputs=[bioimage_testinput_fpath],
    test_outputs=[bioimage_testoutput_fpath],
    input_axes=["bcyx"],
    output_axes=["bcyx"],
    # where to save the model zip, how to call the model and a short description of it
    output_path=os.path.join(outputdir,'usplit.zip'),
    name="usplit",
    description="uspilt model",
    # additional metadata about authors, licenses, citation etc.
    authors=[{"name": "Ashesh"}],
    license="CC-BY-4.0",
    documentation=bioimage_doc_fpath,
    tags=["image decomposition"],  # the tags are used to make models more findable on the website
    cite=[{"text": "Ashesh et al.", "doi": "doi:10.48550/arXiv.2211.12872"}],
)