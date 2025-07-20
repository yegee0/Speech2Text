import torch
from torch import nn
from torch.nn import Sequential

class NormGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            dropout=dropout_rate,
            batch_first=False,
            bidirectional=True,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, inputs, hs=None):
        x, hs = self.gru(inputs, hs)
        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)
        batch_size, time_steps = x.shape[1], x.shape[0]
        x = x.view(batch_size * time_steps, -1)
        x = self.batch_norm(x)
        x = x.view(time_steps, batch_size, -1).contiguous()

        return x, hs


class DeepSpeech2(nn.Module):
    def __init__(self, n_feats, n_tokens, num_rnn_layers, hidden_size, rnn_dropout):
        super().__init__()
        self.conv_params = {
            "conv1": {"padding": (20, 5), "kernel_size": (41, 11), "stride": (2, 2)},
            "conv2": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 2)},
            "conv3": {"padding": (10, 5), "kernel_size": (21, 11), "stride": (2, 1)},
        }

        self.conv = Sequential(
            nn.Conv2d(1, 32, **self.conv_params["conv1"]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, **self.conv_params["conv2"]),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 96, **self.conv_params["conv3"]),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        rnn_input_dim = self.calc_rnn_input_size(n_feats)
        rnn_input_dim *= 96

        self.rnns = Sequential(
            *[
                NormGRU(
                    (hidden_size if i > 0 else rnn_input_dim), hidden_size, rnn_dropout
                )
                for i in range(num_rnn_layers)
            ]
        )

        self.fc = nn.Linear(hidden_size, n_tokens)

    def calc_rnn_input_size(self, n_feats):
        size = n_feats
        for conv_param in self.conv_params.values():
            size = (
                size + 2 * conv_param["padding"][0] - conv_param["kernel_size"][0]
            ) // conv_param["stride"][0] + 1
        return size

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = self.conv(spectrogram.unsqueeze(1))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = (
            x.transpose(1, 2).transpose(0, 1).contiguous()
        )  # n_tokens x batch_size x (n_channels * n_freqs)

        h = None
        for rnn in self.rnns:
            x, h = rnn(x, h)

        time_steps, batch_size = x.shape[0], x.shape[1]
        x = x.view(time_steps * batch_size, -1)
        logits = self.fc(x)
        logits = logits.view(time_steps, batch_size, -1).transpose(0, 1)

        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return {
            "logits": logits,
            "log_probs": log_probs,
            "log_probs_length": self.transform_input_lengths(spectrogram_length),
        }

    def transform_input_lengths(self, input_lengths):
        for conv_param in self.conv_params.values():
            input_lengths = (
                input_lengths
                + 2 * conv_param["padding"][1]
                - conv_param["kernel_size"][1]
            ) // conv_param["stride"][1] + 1
        return input_lengths

    def __str__(self):
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result_str = super().__str__()
        result_str += f"\nAll parameters: {all_params}"
        result_str += f"\nTrainable parameters: {trainable_params}"
        return result_str
