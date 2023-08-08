import numpy as np
import torch

from usplit.core.stable_exp import StableExponential


def test_stable_exponential_give_correct_values():

    def exp(v):
        return torch.exp(torch.Tensor([v]))[0]

    x = torch.Tensor([1, 2, 100, -1, -4])
    expected_output = torch.Tensor([2, 3, 101, exp(-1), exp(-4)])
    output = StableExponential(x).exp()
    assert torch.all(torch.abs(output - expected_output) < 1e-7)


def test_stable_exponential_has_correct_log():
    """
    Taking torch.log() on output of exp() has the same effect.
    """
    x = np.arange(-10, 100, 0.01)
    gen = StableExponential(torch.Tensor(x))
    exp = gen.exp()
    log1 = gen.log()
    log2 = torch.log(exp)

    assert torch.all(torch.abs(log2 - log1).max() < 1e-6)
