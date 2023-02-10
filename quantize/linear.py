import torch
import torch.nn as nn

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=4):
        super().__init__()
        self.bits = bits
        self.register_buffer('weight', torch.randn(out_features, in_features))
        self.register_buffer('scale', torch.ones(out_features, 1))

    def forward(self, x):
        # Simulated 4-bit quantization logic
        q_weight = torch.round(self.weight / self.scale)
        q_weight = torch.clamp(q_weight, - (2**(self.bits-1)), 2**(self.bits-1) - 1)
        return nn.functional.linear(x, q_weight * self.scale)
