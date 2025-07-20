import random
from typing import Dict, List

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from .wav_augs import ColoredNoise, Gain, Identity, PitchShifting, SpeedChange


class RandomTransform(nn.Module):
    def __init__(self, transforms: List[Dict]):
        super().__init__()
        self.transforms = nn.ModuleList()
        self.probs = []

        for item in transforms:
            transform_cfg = item["transform_cfg"]
            prob = item["p"]

            if isinstance(transform_cfg, (DictConfig, ListConfig, dict)):
                transform = instantiate(transform_cfg)
            else:
                transform = transform_cfg

            self.transforms.append(transform)
            self.probs.append(prob)

        total_prob = sum(self.probs)
        assert (
            abs(total_prob - 1.0) < 1e-6
        ), f"Probs must sum to 1, but they sum to {total_prob}"

    def forward(self, data: Tensor) -> Tensor:
        transform = random.choices(self.transforms, self.probs)[0]
        return transform(data)
