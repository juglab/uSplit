import argparse
import os
import pickle

from usplit.analysis.results_handler import PaperResultsHandler


def rnd(obj):
    return f'{obj:.3f}'


def show(ckpt_dir, results_dir, only_test=True, skip_last_pixels=None):
    if ckpt_dir[-1] == '/':
        ckpt_dir = ckpt_dir[:-1]
    if results_dir[-1] == '/':
        results_dir = results_dir[:-1]

    fname = PaperResultsHandler.get_fname(ckpt_dir)
    print(ckpt_dir)
    for dir in sorted(os.listdir(results_dir)):
        if only_test and dir[:4] != 'Test':
            continue
        if skip_last_pixels is not None:
            sktoken = dir.split('_')[-1]
            assert sktoken[:2] == 'Sk'
            if int(sktoken[2:]) != skip_last_pixels:
                continue

        fpath = os.path.join(results_dir, dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                out = pickle.load(f)

            print('')
            print(dir)
            print('RMSE', ' '.join([rnd(x) for x in out['rmse']]))
            print('PSNR', ' '.join([rnd(x) for x in out['psnr']]))
            print('RangeInvPSNR', ' '.join([rnd(x) for x in out['rangeinvpsnr']]))
            print('SSIM', ' '.join(rnd(x) for x in out['ssim']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', type=str)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('--skip_last_pixels', type=int)
    args = parser.parse_args()

    # ckpt_dir = '/home/ashesh.ashesh/training/disentangle/2210/D3-M3-S0-L0/117'
    # results_dir = '/home/ashesh.ashesh/data/paper_stats/'
    show(args.ckpt_dir, args.results_dir, only_test=True, skip_last_pixels=args.skip_last_pixels)
