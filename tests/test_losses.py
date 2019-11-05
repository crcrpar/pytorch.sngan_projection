import pytest

import numpy as np
import torch
import torch.nn.functional as F

from losses import DisLoss
from losses import GenLoss


BATCH_SIZE = 10


def np_relu(x):
    return np.max(x, 0)


def np_softplus(x):
    return np.log(1 + np.exp(x))


@pytest.mark.parametrize('loss_type', [('hinge'), ('dcgan'), ('sngan')])
def test_construct_dis_loss(loss_type: str) -> None:
    if loss_type == 'sngan':
        for klass in (DisLoss, GenLoss):
            with pytest.raises(ValueError):
                klass(loss_type)
    else:
        for klass in (DisLoss, GenLoss):
            fail = False
            try:
                klass(loss_type)
            except Exception:
                fail = True
            assert not fail
