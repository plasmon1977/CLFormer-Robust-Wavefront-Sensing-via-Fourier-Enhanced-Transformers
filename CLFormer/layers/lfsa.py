from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, LayerNorm_type: str = "WithBias"):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LFSA(nn.Module):
    def __init__(
        self,
        dim: int,
        bias: bool = False,
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(
            dim * 6, dim * 6, kernel_size=3, stride=1,
            padding=1, groups=dim * 6, bias=bias
        )
        self.norm = LayerNorm(dim * 2, LayerNorm_type="WithBias")
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.to_hidden(x)
        hidden_dw = self.to_hidden_dw(hidden)
        q, k, v = hidden_dw.chunk(3, dim=1)

        q_patch = rearrange(
            q,
            "b c (h p1) (w p2) -> b c h w p1 p2",
            p1=self.patch_size, p2=self.patch_size,
        )
        k_patch = rearrange(
            k,
            "b c (h p1) (w p2) -> b c h w p1 p2",
            p1=self.patch_size, p2=self.patch_size,
        )

        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out_fft = q_fft * k_fft
        out = torch.fft.irfft2(out_fft, s=(self.patch_size, self.patch_size))

        out = rearrange(
            out,
            "b c h w p1 p2 -> b c (h p1) (w p2)",
            p1=self.patch_size, p2=self.patch_size,
        )

        out_norm = self.norm(out)
        output = v * out_norm
        output = self.project_out(output)
        return output
