"""
Here, we create the input which is needed for the doubledip to work upon.
Every data point will have 2 mixed input. Here, we are simply passing the channels to the output
"""
import os.path

import numpy as np


def dump_individual_channels(dset, idx_list, outputdir, label):
    outputdir = os.path.join(outputdir, label)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    for idx in idx_list:
        _, tar = dset[idx]
        fpath = os.path.join(outputdir, f'{idx}.npy')
        np.save(fpath, tar)
