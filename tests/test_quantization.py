import torch
from torch import nn

from src.model.quantization import quantize_model_int8


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


def test_dynamic_quantization_runs():
    model = TinyModel().eval()
    quantized_model = quantize_model_int8(model)

    x = torch.randn(2, 8)

    with torch.inference_mode():
        output = quantized_model(x)

    assert output.shape == (2, 4)
