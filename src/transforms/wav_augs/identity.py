import torch.nn as nn
from torch import Tensor


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Tensor) -> Tensor:
        return data
