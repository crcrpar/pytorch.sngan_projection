import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils


class Block(nn.Module):

    """Residual block of Discriminator.

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
                 padding='zero', activation=F.relu, downsample=False):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = in_ch != out_ch or downsample
        if h_ch is None:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, 0))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, 0))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, ksize, 1, 0)
            )

        self._initialize()

        if padding == 'reflection':
            self.padding = nn.ReflectionPad2d(pad)
        else:
            self.padding = nn.ZeroPad2d(pad)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(self.padding(x), 2, padding=0)
        return x

    def residual(self, x):
        h = self.c1(self.padding(self.activation(x)))
        h = self.c2(self.padding(self.activation(h)))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, padding='zero', activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, 0))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, 0))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

        if padding == 'reflection':
            self.padding = nn.ReflectionPad2d(pad)
        else:
            self.padding = nn.ZeroPad2d(pad)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(self.padding(x)))
        return F.avg_pool2d(self.c2(self.padding(h)), 2)
