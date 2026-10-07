from __future__ import annotations

from typing import Any

import torch


class Translator:
    def __init__(
        self,
        model,
        tokenizer,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def _language_token(self, language_name: str) -> str:
        languages = self.config["languages"]

        if language_name not in languages:
            raise ValueError(
                f"Unknown language '{language_name}'. "
                "Add it to configs/config.yaml."
            )

        return languages[language_name]

    def _target_token_id(self, target_token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(target_token)

        if token_id is None:
            raise ValueError(
                f"Tokenizer does not recognize target token '{target_token}'."
            )

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)

        if unk_token_id is not None and token_id == unk_token_id:
            raise ValueError(
                f"'{target_token}' maps to the tokenizer's unknown token. "
                "Check the language token in configs/config.yaml."
            )

        return int(token_id)

    @torch.inference_mode()
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        text = text.strip()

        if not text:
            raise ValueError("Input text is empty.")

        source_token = self._language_token(source_language)
        target_token = self._language_token(target_language)

        # NLLB tokenizers use src_lang to insert the appropriate source-language
        # special token during tokenization.
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = source_token

        generation_config = self.config["generation"]

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(generation_config["max_input_length"]),
        )

        encoded = {
            key: value.to("cpu")
            for key, value in encoded.items()
        }

        target_token_id = self._target_token_id(target_token)

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=target_token_id,
            max_new_tokens=int(generation_config["max_new_tokens"]),
            num_beams=int(generation_config["num_beams"]),
            length_penalty=float(generation_config["length_penalty"]),
            repetition_penalty=float(generation_config["repetition_penalty"]),
            early_stopping=bool(generation_config["early_stopping"]),
        )

        translation = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        return translation.strip()
