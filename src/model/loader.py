from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.model.quantization import quantize_model_int8


def _resolve_model_reference(
    model_config: dict[str, Any],
) -> str:

    source = model_config["source"].lower().strip()
    path_or_repo = model_config["path_or_repo"]

    if source == "local":
        model_path = Path(path_or_repo)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model folder not found: {model_path}"
            )

        return str(model_path)

    if source == "huggingface":
        return path_or_repo

    raise ValueError(
        "model.source must be 'local' or 'huggingface'."
    )


def _get_huggingface_token() -> str | None:
    """
    Retrieve the Hugging Face token stored
    securely in Streamlit Secrets.
    """

    try:
        return st.secrets["HF_TOKEN"]

    except (KeyError, FileNotFoundError):
        return None


def load_translation_components(
    config: dict[str, Any],
):

    model_config = config["model"]

    model_reference = _resolve_model_reference(
        model_config
    )

    source = model_config["source"].lower().strip()

    token = None

    if source == "huggingface":
        token = _get_huggingface_token()

        if token is None:
            raise RuntimeError(
                "The Hugging Face model is private but "
                "HF_TOKEN was not found in Streamlit Secrets."
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_reference,
        token=token,
        local_files_only=(source == "local"),
        trust_remote_code=bool(
            model_config.get(
                "trust_remote_code",
                False,
            )
        ),
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_reference,
        token=token,
        local_files_only=(source == "local"),
        trust_remote_code=bool(
            model_config.get(
                "trust_remote_code",
                False,
            )
        ),
        low_cpu_mem_usage=True,
    )

    model.eval()

    if bool(
        model_config.get(
            "quantize_int8",
            True,
        )
    ):
        model = quantize_model_int8(model)

    return model, tokenizer