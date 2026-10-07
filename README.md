# Neural Machine Translation App

Interactive Streamlit deployment for a fine-tuned NLLB-style sequence-to-sequence
translation model.

## Features

- Hugging Face `AutoTokenizer` + `AutoModelForSeq2SeqLM`
- Local model folder or Hugging Face Hub repository
- CPU inference
- Optional dynamic INT8 quantization of `torch.nn.Linear` layers
- NLLB source/target language-token handling
- YAML configuration
- Streamlit model caching
- Unit tests with `pytest`
- Ready for Streamlit Community Cloud

## Project structure

```text
.
├── app.py
├── configs/
│   └── config.yaml
├── models/
│   └── translation_model/
├── src/
│   ├── inference/
│   │   └── translator.py
│   ├── model/
│   │   ├── loader.py
│   │   └── quantization.py
│   └── utils/
│       └── config.py
├── tests/
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── requirements-dev.txt
```

## 1. Prepare your model

Your model should be saved in Hugging Face format.

Example:

```python
model.save_pretrained("models/translation_model")
tokenizer.save_pretrained("models/translation_model")
```

For local execution, copy that folder into:

```text
models/translation_model/
```

Then keep:

```yaml
model:
  source: "local"
  path_or_repo: "models/translation_model"
  quantize_int8: true
  local_files_only: true
```

## 2. Configure languages

Edit `configs/config.yaml`.

For standard NLLB languages:

```yaml
languages:
  French: "fra_Latn"
  English: "eng_Latn"
```

For your fine-tuned low-resource languages, use the EXACT language tokens expected
by your tokenizer.

## 3. Install

Python 3.12 is a practical deployment target for Streamlit Community Cloud.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

## 4. Run locally

From the repository root:

```bash
streamlit run app.py
```

## 5. Run tests

```bash
pytest -q
```

## 6. Deploy without committing the large model to GitHub

For a ~600M/660M model, the recommended deployment layout is:

```text
GitHub repository -> Streamlit Community Cloud
                         |
                         v
                  Hugging Face model repo
```

Upload your model/tokenizer to a Hugging Face model repository, then change:

```yaml
model:
  source: "huggingface"
  path_or_repo: "YOUR_USERNAME/YOUR_MODEL_REPOSITORY"
  quantize_int8: true
  local_files_only: false
```

If the Hugging Face repository is public, no token is required.

If the repository is private, authentication must be added through Streamlit
Secrets rather than hardcoding a token in the repository.

## 7. INT8 quantization

The project uses PyTorch dynamic quantization:

```python
quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

This is a CPU-oriented optimization. The original floating-point model is loaded
first and then transformed in memory. Therefore, peak startup memory can still be
higher than the final quantized model size.

Always compare translation quality before and after quantization.

## Important deployment note

A 600M/660M sequence-to-sequence model is large for a free CPU hosting service.
Whether it fits depends on the memory available to the Streamlit instance and the
exact checkpoint format. If the application exceeds the platform's memory limit,
the next optimization step should be benchmarked rather than guessed (for example,
a smaller/distilled checkpoint, ONNX Runtime quantization, or another inference
backend).
