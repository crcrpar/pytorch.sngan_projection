from typing import Any, Optional, Tuple  # NOQA

import torch
import torch.nn.functional as F
from torch import Tensor  # NOQA


AVAILABLE_LOSSES = ["hinge", "dcgan"]


def dis_hinge(dis_fake: Tensor, dis_real: Tensor) -> Tensor:
    loss = torch.mean(torch.relu(1. - dis_real)) +\
        torch.mean(torch.relu(1. + dis_fake))
    return loss


def gen_hinge(dis_fake: Tensor, dis_real: Optional[Tensor] = None) -> Tensor:
    return -torch.mean(dis_fake)


def dis_dcgan(dis_fake: Tensor, dis_real: Tensor) -> Tensor:
    loss = torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    return loss


def gen_dcgan(dis_fake: Tensor, dis_real: Optional[Tensor] = None) -> Tensor:
    return torch.mean(F.softplus(-dis_fake))


class _Loss(object):

    """GAN Loss base class.

    Args:
        loss_type (str)
        is_relativistic (bool)

    """

    def __init__(self, loss_type: str, is_relativistic: bool = False) -> None:
        if loss_type not in AVAILABLE_LOSSES:
            raise ValueError('Invalid loss type')
        self.loss_type = loss_type
        self.is_relativistic = is_relativistic

    def _preprocess(self, dis_fake: Tensor, dis_real: Tensor) -> Tuple[Tensor, Tensor]:
        C_xf_tilde = torch.mean(dis_fake, dim=0, keepdim=True).expand_as(dis_fake)
        C_xr_tilde = torch.mean(dis_real, dim=0, keepdim=True).expand_as(dis_real)
        return dis_fake - C_xr_tilde, dis_real - C_xf_tilde


class DisLoss(_Loss):

    """Discriminator Loss."""

    def __call__(self, dis_fake: Tensor, dis_real: Tensor, **kwargs: Any) -> Tensor:
        if not self.is_relativistic:
            if self.loss_type == "hinge":
                return dis_hinge(dis_fake, dis_real)
            elif self.loss_type == "dcgan":
                return dis_dcgan(dis_fake, dis_real)
        else:
            d_xf, d_xr = self._preprocess(dis_fake, dis_real)
            if self.loss_type == "hinge":
                return dis_hinge(d_xf, d_xr)
            elif self.loss_type == "dcgan":
                D_xf = torch.sigmoid(d_xf)
                D_xr = torch.sigmoid(d_xr)
                return -torch.log(D_xr) - torch.log(1.0 - D_xf)
            else:
                raise NotImplementedError


class GenLoss(_Loss):

    """Generator Loss."""

    def __call__(self, dis_fake: Tensor, dis_real: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
        if not self.is_relativistic:
            if self.loss_type == "hinge":
                return gen_hinge(dis_fake, dis_real)
            elif self.loss_type == "dcgan":
                return gen_dcgan(dis_fake, dis_real)
        else:
            assert dis_real is not None, "Relativistic Generator loss requires `dis_real`."
            d_xf, d_xr = self._preprocess(dis_fake, dis_real)
            if self.loss_type == "hinge":
                return dis_hinge(d_xr, d_xf)
            elif self.loss_type == "dcgan":
                D_xf = torch.sigmoid(d_xf)
                D_xr = torch.sigmoid(d_xr)
                return -torch.log(D_xf) - torch.log(1.0 - D_xr)
            else:
                raise NotImplementedError
