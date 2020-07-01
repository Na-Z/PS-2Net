import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class BatchNorm1d(_BatchNorm):
    """
    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, momentum_decay_step=None, momentum_decay_rate=1):
        super(BatchNorm1d, self).__init__(num_features, eps, momentum, affine)
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay_rate = momentum_decay_rate
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm1d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None) and (epoch >= 1) and (self.momentum_decay_step is not None) and (self.momentum_decay_step > 0):
            # perform momentum decay
            self.momentum = self.momentum_original * (self.momentum_decay_rate**(epoch//self.momentum_decay_step))
            if self.momentum < 0.01:
                self.momentum = 0.01

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class Conv1d_wrapper(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, normalization=None, activation=None, norm_momentum=0.1):
        super(Conv1d_wrapper, self).__init__()

        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv1d(num_in_channels, num_out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        if self.normalization == 'batch':
            self.norm = BatchNorm1d(num_out_channels, momentum=norm_momentum, affine=True)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm1d(num_out_channels, momentum=norm_momentum, affine=True)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU(alpha=1.0)
        elif self.activation == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                #Initialize the scalar in Normalization, when set "affine=True"
                m.weight.data.fill_(1) #TODO implement a different normalization method
                m.bias.data.fill_(0)

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization=='batch':
            x = self.norm(x, epoch)
        elif self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)
        return x


class BatchNorm2d(_BatchNorm):
    """
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, momentum_decay_step=None, momentum_decay_rate=1):
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine)
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay_rate = momentum_decay_rate
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None) and (epoch >= 1) and (self.momentum_decay_step is not None) and (self.momentum_decay_step > 0):
            # perform momentum decay
            self.momentum = self.momentum_original * (self.momentum_decay_rate**(epoch//self.momentum_decay_step))
            if self.momentum < 0.01:
                self.momentum = 0.01

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class Conv2d_wrapper(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, kernel_size, normalization=None, activation=None, norm_momentum=0.1):
        super(Conv2d_wrapper, self).__init__()

        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv2d(num_in_channels, num_out_channels, kernel_size, stride=(1,1), padding=0, bias=True)
        if self.normalization == 'batch':
            self.norm = BatchNorm2d(num_out_channels, momentum=norm_momentum, affine=True)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm2d(num_out_channels, momentum=norm_momentum, affine=True)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU(alpha=1.0)
        elif self.activation == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                #Initialize the scalar in Normalization, when set "affine=True"
                m.weight.data.fill_(1) #TODO implement a different normalization method
                m.bias.data.fill_(0)

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization=='batch':
            x = self.norm(x, epoch)
        elif self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)
        return x


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=13, input_feat_size=64, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()

        self.num_clusters = num_clusters
        self.dim = input_feat_size
        self.normalize_input = normalize_input
        self.conv = nn.Conv1d(self.dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, self.dim))

        self._init_params()

    def _init_params(self):
        """
        :return: self.conv.weight: torch.Tensor (out_channels, in_channels, kernel_size)
                 self.conv.bias: torch.Tensor (out_channels)
        """
        self.conv.weight.data.normal_(0, 1 / math.sqrt(self.dim))
        self.conv.bias.data.normal_(0, 1 / math.sqrt(self.dim))

    def forward(self, x):
        """
        :param x: torch.Tensor (B,C,N) input feature
        :return: vlad: torch.Tensor (B, M*C) (M is the number of clusters)
                        the feartures of clusters
        """
        B, _, N = x.size()

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x)
        soft_assign = F.softmax(soft_assign, dim=1) #(B,M,N)
        _, index = torch.max(soft_assign, dim=1, keepdim=False) #(B,N)

        # calculate residuals to each clusters
        residual = x.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                    self.centroids.expand(N, -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2) #(B,M,C,N)
        vlad = residual.sum(dim=-1) #(B,M,C)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(B, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class NetVLAD_wrapper(nn.Module):
    def __init__(self, num_in_channels, num_clusters, num_out_channels, normalization, activation, norm_momentum):
        super(NetVLAD_wrapper, self).__init__()
        self.netvlad = NetVLAD(num_clusters, num_in_channels, normalize_input=True)
        self.reduct_feat = Conv1d_wrapper(num_clusters*num_in_channels, num_out_channels, normalization, activation, norm_momentum)

    def forward(self, x):
        y = self.netvlad(x) #(B,C')
        y = y.unsqueeze(-1)
        y = self.reduct_feat(y)
        return y