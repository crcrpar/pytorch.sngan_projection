import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from links import CategoricalConditionalBatchNorm2d


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class Block(nn.Module):

    """Residual block of Generator.

    Args:
        in_ch (int): Number of channels of input.
        out_ch (int): Number of channels of output.
        h_ch (int, optional): Number of channels of hidden
        ksize (int, optional): Kernel size. (Default: 3)
        pad (int, optional): width of padding. (Default: 1)
        padding (str, optional): Padding method. (Default: zero)
        activation (callable, optional): Default: ReLU
        upsample (bool, optional): Default: False
        num_classes (int, optional):
            Argument for categorical conditional batch normalization

    """

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 padding='zero', activation=F.relu, upsample=False,
                 num_classes=0):
        super(Block, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        if padding == 'reflection':
            self.padding = nn.ReflectionPad2d(pad)
        else:
            self.padding = nn.ZeroPad2d(pad)

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, 0)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, 0)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(
                num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(
                num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.tensor, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.tensor, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.tensor, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(self.padding(h))
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.padding(self.activation(h)))
