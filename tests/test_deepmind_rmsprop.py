import math

import pytest
import torch

from rl_core.optim import DeepMindRMSprop


def test_deepmind_rmsprop_uses_centered_denominator_with_epsilon_inside_square_root():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = DeepMindRMSprop([parameter], lr=1.0, alpha=0.5, eps=0.25)
    parameter.grad = torch.tensor([2.0])

    optimizer.step()

    expected = 1.0 - 2.0 / math.sqrt(2.0 - 1.0 + 0.25)
    assert parameter.item() == pytest.approx(expected)


def test_deepmind_rmsprop_rejects_sparse_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = DeepMindRMSprop([parameter])
    parameter.grad = torch.sparse_coo_tensor(
        indices=torch.tensor([[0]]),
        values=[1.0],
        size=(2,),
    )

    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()
