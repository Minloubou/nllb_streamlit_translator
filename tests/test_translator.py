from unittest.mock import MagicMock

import torch

from src.inference.translator import Translator


class FakeTokenizer:
    unk_token_id = 0
    src_lang = None

    def convert_tokens_to_ids(self, token):
        return {"fra_Latn": 1, "eng_Latn": 2}.get(token, 0)

    def __call__(self, text, return_tensors, truncation, max_length):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def batch_decode(self, tokens, skip_special_tokens):
        return ["Hello world"]


class FakeModel:
    def generate(self, **kwargs):
        return torch.tensor([[2, 4, 5]])


def test_translate():
    config = {
        "languages": {
            "French": "fra_Latn",
            "English": "eng_Latn",
        },
        "generation": {
            "max_input_length": 128,
            "max_new_tokens": 64,
            "num_beams": 1,
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "early_stopping": True,
        },
    }

    translator = Translator(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        config=config,
    )

    result = translator.translate(
        "Bonjour le monde",
        source_language="French",
        target_language="English",
    )

    assert result == "Hello world"
