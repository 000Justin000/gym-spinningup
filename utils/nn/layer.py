from .module import BaseModule, register_module
from typing import Dict
from torch import Tensor
import torch
import torch.nn as nn


@register_module
class BaseLayer(BaseModule):

    def __init__(self):
        super().__init__()

    @classmethod
    def setup_from_config(cls, config):
        return cls(**config)


@register_module
class LinearLayer(BaseLayer):

    def __init__(
        self,
        in_key: str,
        out_key: str,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.module = nn.Linear(in_dim, out_dim, bias=bias)
        self.in_key = in_key
        self.out_key = out_key

    def forward(self, dict: Dict[str, Tensor]):
        x = dict[self.in_key]
        dict[self.out_key] = self.module(x)
        return dict


@register_module
class ReLULayer(BaseLayer):
    def __init__(
        self,
        in_key: str,
        out_key: str,
    ):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key

    def forward(self, dict: Dict[str, Tensor]):
        x = dict[self.in_key]
        dict[self.out_key] = torch.relu(x)
        return dict


@register_module
class Conv2dLayer(BaseLayer):
    def __init__(
        self,
        in_key: str,
        out_key: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self.module = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, dict: Dict[str, Tensor]):
        x = dict[self.in_key]
        dict[self.out_key] = self.module(x)
        return dict
