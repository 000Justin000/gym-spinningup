from module import BaseModule, register_module, get_module, setup_module
from typing import Dict, List, Any, Optional, Union
from torch import Tensor
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
