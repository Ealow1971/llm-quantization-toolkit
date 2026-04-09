import torch
import torch.nn as nn

class Quantizer:
    def __init__(self, bits=4):
        self.bits = bits

    def quantize_tensor(self, tensor):
        scale = tensor.abs().max() / (2**(self.bits-1) - 1)
        quantized = (tensor / scale).round().clamp(-(2**(self.bits-1)), 2**(self.bits-1)-1)
        return quantized, scale

if __name__ == "__main__":
    q = Quantizer()
    t = torch.randn(100, 100)
    qt, s = q.quantize_tensor(t)
    print(f"Quantized tensor shape: {qt.shape}")
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP









