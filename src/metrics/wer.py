from typing import List
import numpy as np
import torch
from torch import Tensor
from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder

class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamWERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ) -> float:
        lengths = log_probs_length.cpu().numpy()
        log_probs_cpu = log_probs.detach().cpu().numpy()

        def get_pred_text(log_prob_vec, length) -> str:
            if hasattr(self.text_encoder, "ctc_beam_search"):
                return self.text_encoder.ctc_beam_search(
                    log_prob_vec[:length], self.beam_size
                )
            return self.text_encoder.ctc_decode(log_prob_vec[:length])

        wers = [
            calc_wer(
                CTCTextEncoder.normalize_text(target_text),
                get_pred_text(log_prob_vec, length),
            )
            for log_prob_vec, length, target_text in zip(log_probs_cpu, lengths, text)
        ]
        return np.mean(wers)


class LMBeamWERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, logits: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ) -> float:
        lengths = log_probs_length.cpu().numpy()
        logits_cpu = logits.detach().cpu().numpy()

        def get_pred_text(logits_vec, length) -> str:
            if hasattr(self.text_encoder, "lm_ctc_beam_search"):
                return self.text_encoder.lm_ctc_beam_search(
                    logits_vec[:length], self.beam_size
                )
            return self.text_encoder.ctc_decode(logits_vec[:length])

        wers = [
            calc_wer(
                CTCTextEncoder.normalize_text(target_text),
                get_pred_text(logits_vec, length),
            )
            for logits_vec, length, target_text in zip(logits_cpu, lengths, text)
        ]
        return np.mean(wers)
