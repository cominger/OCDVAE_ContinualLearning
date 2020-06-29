import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .resblocks import Block


class ResNetGenerator(nn.Module):
    """Generator generates 128x128."""

    def __init__(self, num_features=64, bottom_width=4,
                 activation=F.relu, num_classes=0):
        super(ResNetGenerator, self).__init__()
        
        self.num_features = int(num_features/10)
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes

        self.block2 = Block(num_features * 16, num_features * 16,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = Block(num_features * 16, num_features * 8,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = Block(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block5 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block6 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b7 = nn.BatchNorm2d(num_features)
        self.conv7 = nn.Conv2d(num_features, 3, 1, 1)

    # def _initialize(self):
    #     init.xavier_uniform_(self.l1.weight.tensor)
    #     init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        # h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        h = z.view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in [2, 3, 4, 5, 6]:
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b7(h))
        return self.conv7(h)
