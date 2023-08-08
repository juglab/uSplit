import numpy as np
import torch

from usplit.core.psnr import PSNR, RangeInvariantPsnr

# range_ = np.max(gt) - np.min(gt)
# mse = np.mean((gt - pred) ** 2)
# return 20 * np.log10((range_) / np.sqrt(mse))


def test_PSNR():
    target = torch.Tensor([
        [10, 11, 12],
        [100, 120, 140],
    ])
    pred = torch.Tensor([
        [15, 10, 13],
        [10, 13, 14],
    ])

    rmse0 = torch.sqrt(torch.Tensor([25 + 1 + 1]) / 3)
    actual_psnr0 = 20 * torch.log10(2 / rmse0)

    rmse1 = torch.sqrt(torch.Tensor([90**2 + 107**2 + 126**2]) / 3)
    actual_psnr1 = 20 * torch.log10(40 / rmse1)

    psnr = PSNR(target[..., None], pred[..., None])

    assert len(psnr) == 2
    assert torch.abs(psnr[0] - actual_psnr0).item() < 1e-6
    assert torch.abs(psnr[1] - actual_psnr1).item() < 1e-6


def _working_PSNR(gt, pred, range_=None):
    '''
        Compute PSNR.
        Parameters
        ----------
        gt: array
            Ground truth image.
        img: array
            Predicted image.
    '''
    if range_ is None:
        range_ = np.max(gt) - np.min(gt)
    mse = np.mean((gt - pred)**2)
    return 20 * np.log10((range_) / np.sqrt(mse))


def _working_zero_mean(x):
    return x - np.mean(x)


def _working_fix_range(gt, x):
    a = np.sum(gt * x) / (np.sum(x * x))
    return x * a


def _working_fix(gt, x):
    gt_ = _working_zero_mean(gt)
    return _working_fix_range(gt_, _working_zero_mean(x))


def _working_RangeInvariantPsnr(gt, pred):
    """
    Taken from https://github.com/juglab/ScaleInvPSNR/blob/master/psnr.py
    It rescales the prediction to ensure that the prediction has the same range as the ground truth.
    """
    ra = (np.max(gt) - np.min(gt)) / np.std(gt)
    gt_ = _working_zero_mean(gt) / np.std(gt)
    return _working_PSNR(_working_zero_mean(gt_), _working_fix(gt_, pred), ra)


def test_RangeInvariantPSNR():
    target = torch.Tensor([
        [10, 11, 12],
        [100, 120, 140],
    ])
    pred = torch.Tensor([
        [15, 10, 13],
        [10, 13, 14],
    ])

    rmse0 = torch.sqrt(torch.Tensor([25 + 1 + 1]) / 3)
    actual_psnr0 = _working_RangeInvariantPsnr(target[0].numpy(), pred[0].numpy())
    actual_psnr1 = _working_RangeInvariantPsnr(target[1].numpy(), pred[1].numpy())

    psnr = RangeInvariantPsnr(target[..., None], pred[..., None])

    assert len(psnr) == 2
    assert torch.abs(psnr[0] - actual_psnr0).item() < 1e-5
    assert torch.abs(psnr[1] - actual_psnr1).item() < 1e-5
