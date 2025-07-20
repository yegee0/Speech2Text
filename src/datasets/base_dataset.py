import logging
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from src.text_encoder import CTCTextEncoder

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        index,
        text_encoder=None,
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        max_text_length=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.text_encoder = text_encoder
        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)
        text = data_dict["text"]
        text_encoded = self.text_encoder.encode(text)

        instance_data: dict = {"audio": audio}
        instance_data = self.preprocess_data(instance_data)  # transforming audio first

        spectrogram = self.get_spectrogram(
            instance_data["audio"]
        )  # calculating spectrogram of the TRANSFORMED audio
        spectrogram = torch.log(
            spectrogram + 1e-5
        )  # for beautiful pictures when logging

        instance_data.update(
            {
                "spectrogram": spectrogram,
                "text": text,
                "text_encoded": text_encoded,
                "audio_path": audio_path,
            }
        )
        return instance_data

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def get_spectrogram(self, audio):
        return self.instance_transforms["get_spectrogram"](audio)

    def preprocess_data(self, instance_data):
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name == "get_spectrogram":
                    continue  # skip special key
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
        max_audio_length,
        max_text_length,
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["audio_len"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                np.array(
                    [len(CTCTextEncoder.normalize_text(el["text"])) for el in index]
                )
                >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records  from dataset"
            )

        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - object ground-truth transcription."
            )
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - length of the audio."
            )

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
