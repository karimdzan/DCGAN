from dataclasses import dataclass, field
import torch
from typing import List, Tuple


@dataclass
class GeneratorConfig:
    in_channels: int = 100
    out_channels: int = 3
    hidden_dim: int = 64
    kernel_size: int = 4 
    stride: int = 2
    padding: int = 1


@dataclass
class DiscriminatorConfig:
    in_channels: int = 3
    out_channels: int = 1
    hidden_dim: int = 64
    kernel_size: int = 4 
    stride: int = 2
    padding: int = 1


generator_config = GeneratorConfig()
discriminator_config = DiscriminatorConfig()