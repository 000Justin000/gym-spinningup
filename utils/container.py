from module import BaseModule, register_module, get_module
from typing import Dict, List, Any, Optional, Union
from torch import Tensor
import torch.nn as nn


@register_module
class ResidualContainer(BaseModule):

    def __init__(self, module, in_key: str, out_key: str, weight: float = 1.0):
        super().__init__()
        self.module = module
        self.in_key = in_key
        self.out_key = out_key
        self.weight = weight

    def forward(self, batch: Dict[str, Tensor]):
        x = batch[self.in_key]
        batch[self.out_key] = x + self.weight * self.module(x)
        return batch

    @classmethod
    def setup_from_config(cls, config):
        module_name, module_config = config.pop("module")
        module = get_module(module_name).setup_from_config(module_config)
        return cls(module=module, **config)


@register_module
class SequentialContainer(BaseModule):

    def __init__(self, module_list: List[BaseModule]):
        super().__init__()
        self.module_list = nn.ModuleList(module_list)

    def forward(self, batch: Dict[str, Tensor]):
        for module in self.module_list:
            batch = module(batch)
        return batch

    @classmethod
    def setup_from_config(cls, config):
        module_list = []
        for module_name, module_config in config["module_list"]:
            module_list.append(get_module(module_name).setup_from_config(module_config))
        return cls(module_list=module_list)
