from __future__ import annotations

import torch
from torch import nn


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Apply PyTorch dynamic INT8 quantization to Linear layers.

    Dynamic quantization is designed for CPU inference and replaces eligible
    floating-point Linear layers with dynamically quantized implementations.

    Notes
    -----
    - The model is first loaded in floating point, then quantized in memory.
    - This is intended primarily for CPU deployment.
    - Always benchmark translation quality after quantization.
    """
    model = model.to("cpu")
    model.eval()

    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        # Compatibility fallback for older PyTorch versions.
        from torch.quantization import quantize_dynamic

    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )

    quantized_model.eval()
    return quantized_model
