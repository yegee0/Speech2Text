import re
from collections import defaultdict
from string import ascii_lowercase
import numpy as np
import torch
from pyctcdecode import build_ctcdecoder

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        lm_path: str = None,
        unigrams_path: str = None,
        *args,
        **kwargs,
    ):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        if lm_path is not None:
            print(f"""
                  --------------------
                  LM path: {lm_path}
                  --------------------
                  """)

            assert unigrams_path is not None, "LM and unigrams should be provided"

            print(f"""
                  --------------------
                  Unigrams path: {unigrams_path}
                  --------------------
                  """)

            unigrams = []
            with open(unigrams_path, "r") as file:
                for line in file:
                    word = line.split()[0]
                    unigrams.append(word)

            self.lm_model = build_ctcdecoder(
                labels=[self.EMPTY_TOK] + list(self.alphabet),
                kenlm_model_path=lm_path,
                unigrams=unigrams,
                alpha=0.6,
                beta=0.2,
            )

        else:
            print("LM path is not provided, guess we're running without LM")

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        empty_ind = self.char2ind[self.EMPTY_TOK]
        last_char_ind = empty_ind
        for ind in inds:
            if ind == last_char_ind:
                continue
            else:
                if ind != empty_ind:
                    decoded.append(self.ind2char[ind])
            last_char_ind = ind
        return "".join(decoded)

    def ctc_beam_search(self, log_probs: np.ndarray, beam_size: int):
        time_dim, char_dim = log_probs.shape
        if char_dim > len(self.vocab):
            raise Exception(
                f"log_probs has shape {log_probs.shape}, char_dim ({char_dim}) > len(self.vocab) ({len(self.vocab)})"
            )

        dp = {("", self.EMPTY_TOK): 1.0}

        def extend_path_and_merge(dp, next_token_probs: np.array, ind2char: dict):
            new_dp = defaultdict(float)
            for ind, next_token_prob in enumerate(next_token_probs):
                cur_char = ind2char[ind]
                for (prefix, last_char), v in dp.items():
                    if cur_char == last_char:
                        new_prefix = prefix
                    else:
                        if cur_char != self.EMPTY_TOK:
                            new_prefix = prefix + cur_char
                        else:
                            new_prefix = prefix
                    new_dp[(new_prefix, cur_char)] += (
                        v * next_token_prob
                    ) 
            return new_dp

        def truncate_paths(dp, beam_size):
            return dict(
                sorted(list(dp.items()), key=lambda x: x[1], reverse=True)[:beam_size]
            )

        for probs in np.exp(log_probs):
            dp = extend_path_and_merge(
                dp=dp, next_token_probs=probs, ind2char=self.ind2char
            )
            dp = truncate_paths(dp, beam_size)

        dp = [(prefix, proba) for (prefix, _), proba in dp.items()]

        return dp[0][
            0
        ] 

    def lm_ctc_beam_search(self, logits: np.ndarray, beam_size: int = 20):
        return self.lm_model.decode(logits=logits, beam_width=beam_size)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
