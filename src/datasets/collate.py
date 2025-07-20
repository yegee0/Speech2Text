import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    text_encoded_lengths = []
    spectrogram_lengths = []
    spectrograms = []
    text_encodeds = []
    texts = []
    audios = []
    audio_paths = []

    for item in dataset_items:
        text_encoded_lengths.append(item["text_encoded"].shape[1])
        spectrogram_lengths.append(item["spectrogram"].shape[2])

        spectrogram = item["spectrogram"].squeeze(0).transpose(0, -1)
        # spectrogram = torch.log(spectrogram + 1e-7)
        spectrograms.append(spectrogram)

        text_encoded = item["text_encoded"].squeeze(0).transpose(0, -1)
        text_encodeds.append(text_encoded)
        texts.append(item["text"])

        audio = item["audio"].squeeze(0)
        audios.append(audio)
        audio_paths.append(item["audio_path"])

    text_encoded_lengths = torch.tensor(text_encoded_lengths, dtype=torch.long)
    spectrogram_lengths = torch.tensor(spectrogram_lengths, dtype=torch.long)

    spectrograms = pad_sequence(
        spectrograms, batch_first=True, padding_value=0.0
    ).transpose(1, -1)
    text_encodeds = pad_sequence(
        text_encodeds, batch_first=True, padding_value=0
    ).transpose(1, -1)
    audios = pad_sequence(audios, batch_first=True, padding_value=0.0).transpose(1, -1)

    result_batch = {
        "text_encoded_length": text_encoded_lengths,
        "spectrogram_length": spectrogram_lengths,
        "spectrogram": spectrograms,
        "text_encoded": text_encodeds,
        "text": texts,
        "audio": audios,
        "audio_path": audio_paths,
    }

    return result_batch
