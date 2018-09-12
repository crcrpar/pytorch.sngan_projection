import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from models.discriminators.resblocks import Block
from models.discriminators.resblocks import OptimizedBlock


class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_features, padding='zero', num_classes=0,
                 activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features, padding=padding)
        self.block2 = Block(num_features, num_features * 2,
                            padding=padding,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            padding=padding,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            padding=padding,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            padding=padding,
                            activation=activation, downsample=True)
        self.block6 = Block(num_features * 16, num_features * 16,
                            padding=padding,
                            activation=activation, downsample=True)
        self.l7 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


class SNResNetConcatDiscriminator(nn.Module):

    def __init__(self, num_features, padding='zero', num_classes=0, activation=F.relu,
                 dim_emb=128):
        super(SNResNetConcatDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dim_emb = dim_emb
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features, padding=padding)
        self.block2 = Block(num_features, num_features * 2,
                            padding=padding,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            padding=padding,
                            activation=activation, downsample=True)
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, dim_emb))
        self.block4 = Block(num_features * 4 + dim_emb, num_features * 8,
                            padding=padding,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            padding=padding,
                            activation=activation, downsample=True)
        self.block6 = Block(num_features * 16, num_features * 16,
                            padding=padding,
                            activation=activation, downsample=False)
        self.l7 = utils.spectral_norm(nn.Linear(num_features * 16, 1))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        if hasattr(self, 'l_y'):
            init.xavier_uniform_(self.l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 4):
            h = getattr(self, 'block{}'.format(i))(h)
        if y is not None:
            emb = self.l_y(y).unsqueeze(-1).unsqueeze(-1)
            emb = emb.expand(emb.size(0), emb.size(1), h.size(2), h.size(3))
            h = torch.cat((h, emb), dim=1)
        for i in range(4, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = torch.sum(self.activation(h), dim=(2, 3))
        return self.l7(h)
