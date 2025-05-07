from abc import ABC, abstractmethod
from typing import Dict
from torch import Tensor
import torch.nn as nn

REGISTRY = {}


def register_module(cls):
    if cls.__name__ in REGISTRY:
        raise ValueError(f"Module {cls.__name__} already registered")
    REGISTRY[cls.__name__] = cls
    return cls


def get_module(name):
    if name not in REGISTRY:
        raise ValueError(f"Module {name} not found in registry")
    return REGISTRY[name]


class BaseModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]):
        raise NotImplementedError

    @abstractmethod
    def setup_from_config(cls, config):
        raise NotImplementedError
