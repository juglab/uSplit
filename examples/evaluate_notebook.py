import argparse
import papermill as pm
from datetime import datetime
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a notebook')
    parser.add_argument('--notebook', type=str, help='Notebook to run', default='/home/ashesh.ashesh/code/uSplit/examples/EvaluateWithDifferentMixingWeights.ipynb')
    parser.add_argument('--outputdir', type=str, help='Output notebook directory', default='/group/jug/ashesh/indiSplit/notebook_results_baselines/')
    # parser.add_argument('parameters', type=str, help='Parameters for the notebook')
    parser.add_argument('--ckpt_dir', type=str, help='Checkpoint to use. eg. /home/ashesh.ashesh/paper_models/Hagen/MitoVsAct/DeepLC/')
    parser.add_argument('--data_dir', type=str, help='Data directory', default='/group/jug/ashesh/data/ventura_gigascience/')
    parser.add_argument('--mmse_count', type=int, help='Number of mmse values to generate', default=5)
    parser.add_argument('--MIXING_WEIGHT', type=float, help='Mixing parameter for input generation', default=0.5)
    parser.add_argument('--image_size_for_grid_centers', type=int, help='Image size for grid centers', default=32)
    parser.add_argument('--custom_image_size', type=int, help='Custom image size', default=64)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    args = parser.parse_args()

    # get a year-month-day hour-minute formatted string
    param_str = f"T-{args.MIXING_WEIGHT}_MMSE-{args.mmse_count}"
    now = datetime.now().strftime("%Y%m%d.%H.%M")
    ckpt_dir = args.ckpt_dir    
    if ckpt_dir[-1] == '/':
        ckpt_dir = ckpt_dir[:-1]
    
    outputdir = os.path.join(args.outputdir, '_'.join(ckpt_dir.split('/')[-3:]))
    fname = os.path.basename(args.notebook)
    fname = fname.replace('.ipynb','')
    fname = f"{fname}_{param_str}_{now}.ipynb"
    output_fpath = os.path.join(outputdir, fname)
    output_config_fpath = os.path.join(outputdir,'config', fname.replace('.ipynb','.txt'))
    os.makedirs(os.path.dirname(output_config_fpath), exist_ok=True)
    # save the configuration
    # convert args to dict
    args_dict = vars(args)
    # save as json
    with open(output_config_fpath, 'w') as f:
        f.write(str(args_dict))

    print(output_fpath, '\n', output_config_fpath)
    pm.execute_notebook(
        args.notebook,
        output_fpath,
        parameters = {
            'ckpt_dir':args.ckpt_dir,
            'data_dir':args.data_dir,
            'mmse_count':args.mmse_count,
            'MIXING_WEIGHT':args.MIXING_WEIGHT,
            'image_size_for_grid_centers':args.image_size_for_grid_centers,
            'custom_image_size':args.custom_image_size,
            'batch_size':args.batch_size
        }
    )