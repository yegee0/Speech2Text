import torchaudio
from torch import Tensor, nn


class SpeedChange(nn.Module):
    def __init__(self, speed_factor: float = 1.0, sample_rate: int = 16000):
        super().__init__()
        self.speed_factor = speed_factor
        self.sample_rate = sample_rate

    def forward(self, data: Tensor) -> Tensor:
        resampled_data = torchaudio.functional.resample(
            data, self.sample_rate, int(self.sample_rate * self.speed_factor)
        )
        return resampled_data
