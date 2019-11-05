import pytest

import numpy as np
import torch
import torch.nn.functional as F

import losses


BATCH_SIZE = 10


def np_relu(x):
    return np.max(x, 0)


def np_softplus(x):
    return np.log(1 + np.exp(x))


@pytest.mark.parametrize('loss_type', [('hinge'), ('dcgan'), ('sngan')])
def test_construct_dis_loss(loss_type: str) -> None:
    fail = False
    try:
        DisLoss(loss_type)
    except Exception:
        fail = True

    if loss_type == 'sngan':
        assert fail
    else:
        assert (not fail)


@pytest.mark.parametrize('loss_type', [('hinge'), ('dcgan'), ('sngan')])
def test_construct_gen_loss(loss_type: str) -> None:
    fail = False
    try:
        GenLoss(loss_type)
    except Exception:
        fail = True

    if loss_type == 'sngan':
        assert fail
    else:
        assert (not fail)
