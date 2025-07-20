import torch_audiomentations
from torch import Tensor, nn


class PitchShifting(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
