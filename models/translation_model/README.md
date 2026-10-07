# Put your model here for local execution

This directory is intentionally ignored by Git except for this README.

Place the Hugging Face `save_pretrained()` output for BOTH the model and tokenizer
inside this directory.

Typical files may include:

- `config.json`
- `generation_config.json`
- `model.safetensors` or sharded `model-xxxxx-of-xxxxx.safetensors`
- `model.safetensors.index.json` if sharded
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `sentencepiece.bpe.model` or the tokenizer files produced by your training code

The exact list depends on how your NLLB model/tokenizer were saved.

For Streamlit Community Cloud, do NOT normally commit a 600M/660M model directly
to regular Git history. Prefer hosting the model in a Hugging Face repository and
switching `model.source` to `huggingface` in `configs/config.yaml`.
