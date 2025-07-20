import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            self.log_spectrogram(**batch)
        else:
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self,
        text,
        logits: torch.tensor,
        log_probs: torch.tensor,
        log_probs_length: torch.tensor,
        audio_path,
        audio: torch.tensor,
        examples_to_log=5,
        **batch
    ):
        indices = random.sample(range(len(text)), min(examples_to_log, len(text)))

        texts = [text[i] for i in indices]
        logits = logits[indices].detach().cpu().numpy()
        log_probas = log_probs[indices].detach().cpu().numpy()
        log_probs_lengths = log_probs_length[indices].detach().cpu().numpy()
        audio_paths = [audio_path[i] for i in indices]
        audios = audio[indices].squeeze().numpy()

        argmax_inds = log_probas.argmax(-1)
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_lengths)
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        tuples = list(
            zip(
                argmax_texts,
                # bs_texts,
                # lm_texts,
                texts,
                argmax_texts_raw,
                audio_paths,
                audios,
            )
        )

        rows = {}
        for (
            pred_argmax,
            target,
            raw_pred,
            audio_path,
            audio_aug,
        ) in tuples:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred_argmax) * 100
            cer = calc_cer(target, pred_argmax) * 100
            rows[Path(audio_path).name] = {
                "audio_augmented": self.writer.wandb.Audio(audio_aug, sample_rate=16000),
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred_argmax,
                "wer_argmax": wer,
                "cer_argmax": cer,
            }

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
