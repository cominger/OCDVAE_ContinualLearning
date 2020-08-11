import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

from .resblocks import Block
from .resblocks import OptimizedBlock


class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu, args=None):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)   # 64
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True) # 32
        self.block3 = Block(num_features * 2, num_features * 4, 
                            activation=activation, downsample=True) # 16
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True) # 8
        self.block5 = Block(num_features * 8, num_features * 16,
                            activation=activation, downsample=True) # 4
        self.block6 = Block(num_features * 16, num_features * 16,
                            activation=activation, downsample=True) # 2
        self.l7 = utils.spectral_norm(nn.Linear(num_features * 16*2*2, 1))
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
        # h = self.activation(h)
        # # Global pooling
        # h = torch.sum(h, dim=(2, 3))
        h = h.view(-1, self.num_features*16*2*2)
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output

    # def gradient_penalty(self, y, x):
    #     weight = torch.ones_like(y)
    #     gradient = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
    #     gradient = gradient.view(gradient.size(0),-1)       
    #     gradient = ((gradient.norm(2,1)-1) **2).mean()

    #     return gradient
    def gradient_penalty(self, y, x):
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0),-1)       
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2,dim=1))

        return torch.mean((dydx_l2norm-1)**2)

class SNResNetProjectionDiscriminator_simple(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu, args=None):
        super(SNResNetProjectionDiscriminator_simple, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        # self.block1 = OptimizedBlock(3, num_features)   # 64
        self.block1 = nn.Conv2d(3, num_features, kernel_size=3, stride=1,padding=1)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True) # 32
        self.block3 = Block(num_features * 2, num_features * 4, 
                            activation=activation, downsample=True) # 16
        # self.block4 = Block(num_features * 4, num_features * 8,
        #                     activation=activation, downsample=True) # 8
        # self.block5 = Block(num_features * 8, num_features * 16,
        #                     activation=activation, downsample=True) # 4
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True) # 2
        self.l7 = utils.spectral_norm(nn.Linear(num_features * 8*8*8, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None, feature_wise_loss=False):
        h = x
        h = self.block1(h)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h)
            if feature_wise_loss and i == 2:
                feature_response = h
        # h = self.activation(h)
        # # Global pooling
        # h = torch.sum(h, dim=(2, 3))
        h = h.view(-1, self.num_features*8*8*8)
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        if feature_wise_loss:
            return output, feature_response
        return output

    # def gradient_penalty(self, y, x):
    #     weight = torch.ones_like(y)
    #     gradient = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
    #     gradient = gradient.view(gradient.size(0),-1)       
    #     gradient = ((gradient.norm(2,1)-1) **2).mean()

    #     return gradient
    def gradient_penalty(self, y, x):
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0),-1)       
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2,dim=1))

        return torch.mean((dydx_l2norm-1)**2)

class SNResNetConcatDiscriminator(nn.Module):

    def __init__(self, num_features, num_classes, activation=F.relu,
                 dim_emb=128):
        super(SNResNetConcatDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dim_emb = dim_emb
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        if num_classes > 0:
            self.l_y = utils.spectral_norm(nn.Embedding(num_classes, dim_emb))
        self.block4 = Block(num_features * 4 + dim_emb, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.block6 = Block(num_features * 16, num_features * 16,
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
