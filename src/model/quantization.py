from __future__ import annotations

import torch
from torch import nn


def quantize_model_int8(model: nn.Module) -> nn.Module:
    model = model.to("cpu")
    model.eval()

    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic

    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )

    quantized_model.eval()

    return quantized_model