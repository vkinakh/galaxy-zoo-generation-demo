from typing import Optional, Dict
from functools import partial
import math

import torch
import torch.nn as nn


def get_activation(activation: str = "lrelu"):
    actv_layers = {
        "relu":  nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, 0.2),
    }
    assert activation in actv_layers, f"activation [{activation}] not implemented"
    return actv_layers[activation]


def get_normalization(normalization: str = "batch_norm"):
    norm_layers = {
        "instance_norm": nn.InstanceNorm2d,
        "batch_norm":    nn.BatchNorm2d,
        "group_norm":    partial(nn.GroupNorm, num_groups=8),
        "layer_norm":    partial(nn.GroupNorm, num_groups=1),
    }
    assert normalization in norm_layers, f"normalization [{normalization}] not implemented"
    return norm_layers[normalization]


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = 1,
        padding_mode: str = "zeros",
        groups: int = 1,
        bias: bool = True,
        transposed: bool = False,
        normalization: Optional[str] = None,
        activation: Optional[str] = "lrelu",
        pre_activate: bool = False,
    ):
        if transposed:
            conv = partial(nn.ConvTranspose2d, output_padding=stride-1)
            padding_mode = "zeros"
        else:
            conv = nn.Conv2d
        layers = [
            conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                groups=groups,
                bias=bias,
            )
        ]

        norm_actv = []
        if normalization is not None:
            norm_actv.append(
                get_normalization(normalization)(
                    num_channels=in_channels if pre_activate else out_channels
                )
            )
        if activation is not None:
            norm_actv.append(
                get_activation(activation)(inplace=True)
            )

        if pre_activate:
            layers = norm_actv + layers
        else:
            layers = layers + norm_actv

        super().__init__(
            *layers
        )


class SubspaceLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_basis: int,
    ):
        super().__init__()

        self.U = nn.Parameter(torch.empty(n_basis, dim))
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]))
        self.mu = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return (self.L * z) @ self.U + self.mu


class EigenBlock(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        in_channels: int,
        out_channels: int,
        n_basis: int,
    ):
        super().__init__()

        self.projection = SubspaceLayer(dim=width*height*in_channels, n_basis=n_basis)
        self.subspace_conv1 = ConvLayer(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            transposed=True,
            activation=None,
            normalization=None,
        )
        self.subspace_conv2 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            transposed=True,
            activation=None,
            normalization=None,
        )

        self.feature_conv1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            transposed=True,
            pre_activate=True,
        )
        self.feature_conv2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            transposed=True,
            pre_activate=True,
        )

    def forward(self, z, h):
        phi = self.projection(z).view(h.shape)
        h = self.feature_conv1(h + self.subspace_conv1(phi))
        h = self.feature_conv2(h + self.subspace_conv2(phi))
        return h


class ConditionalGenerator(nn.Module):

    """Conditional generator
    It generates images from one hot label + noise sampled from N(0, 1) with explorable z injection space
    Based on EigenGAN
    """

    def __init__(self,
                 size: int,
                 y_size: int,
                 z_size: int,
                 out_channels: int = 3,
                 n_basis: int = 6,
                 noise_dim: int = 512,
                 base_channels: int = 16,
                 max_channels: int = 512,
                 y_type: str = 'one_hot'):

        if y_type not in ['one_hot', 'multi_label', 'mixed', 'real']:
            raise ValueError('Unsupported `y_type`')

        super(ConditionalGenerator, self).__init__()

        assert (size & (size - 1) == 0) and size != 0, "img size should be a power of 2"

        self.y_type = y_type
        self.y_size = y_size
        self.eps_size = z_size

        self.noise_dim = noise_dim
        self.n_basis = n_basis
        self.n_blocks = int(math.log(size, 2)) - 2

        def get_channels(i_block):
            return min(max_channels, base_channels * (2 ** (self.n_blocks - i_block)))

        self.y_fc = nn.Linear(self.y_size, self.y_size)
        self.concat_fc = nn.Linear(self.y_size + self.eps_size, self.noise_dim)

        self.fc = nn.Linear(self.noise_dim, 4 * 4 * get_channels(0))

        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            self.blocks.append(
                EigenBlock(
                    width=4 * (2 ** i),
                    height=4 * (2 ** i),
                    in_channels=get_channels(i),
                    out_channels=get_channels(i + 1),
                    n_basis=self.n_basis,
                )
            )

        self.out = nn.Sequential(
            ConvLayer(base_channels, out_channels, kernel_size=7, stride=1, padding=3, pre_activate=True),
            nn.Tanh(),
        )

    def forward(self,
                y: torch.Tensor,
                eps: Optional[torch.Tensor] = None,
                zs: Optional[torch.Tensor] = None,
                return_eps: bool = False):

        bs = y.size(0)

        if eps is None:
            eps = self.sample_eps(bs)

        if zs is None:
            zs = self.sample_zs(bs)

        y_out = self.y_fc(y)
        concat = torch.cat((y_out, eps), dim=1)
        concat = self.concat_fc(concat)

        out = self.fc(concat).view(len(eps), -1, 4, 4)
        for block, z in zip(self.blocks, zs.permute(1, 0, 2)):
            out = block(z, out)
        out = self.out(out)

        if return_eps:
            return out, concat

        return out

    def sample_zs(self, batch: int, truncation: float = 1.):
        device = self.get_device()
        zs = torch.randn(batch, self.n_blocks, self.n_basis, device=device)

        if truncation < 1.:
            zs = torch.zeros_like(zs) * (1 - truncation) + zs * truncation
        return zs

    def sample_eps(self, batch: int, truncation: float = 1.):
        device = self.get_device()
        eps = torch.randn(batch, self.eps_size, device=device)

        if truncation < 1.:
            eps = torch.zeros_like(eps) * (1 - truncation) + eps * truncation
        return eps

    def get_device(self):
        return self.fc.weight.device

    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).mean()
                )
        return sum(reg) / len(reg)
