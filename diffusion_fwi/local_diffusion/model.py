"""
Conditional diffusion U-Net для FWI (локальный baseline).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
  def __init__(self, dim: int) -> None:
    super().__init__()
    self.dim = dim

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    half = self.dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionUNet(nn.Module):
  """Простой U-Net: concat(noisy_vp, seismic) -> predicted noise."""

  def __init__(self, base_ch: int = 32) -> None:
    super().__init__()
    ch = base_ch

    def block(c_in, c_out):
      return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.ReLU(inplace=True),
      )

    self.enc = block(2, ch)
    self.mid = block(ch, ch)
    self.dec = block(ch, ch)
    self.out = nn.Conv2d(ch, 1, 1)

  def forward(self, x_noisy: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    _ = t
    x = torch.cat([x_noisy, cond], dim=1)
    x = self.enc(x)
    x = self.mid(x)
    x = self.dec(x)
    return self.out(x)


class DiffusionSchedule:
  def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
    self.timesteps = timesteps
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    self.alphas_cumprod = torch.cumprod(alphas, dim=0)

  def to(self, device: torch.device) -> "DiffusionSchedule":
    self.alphas_cumprod = self.alphas_cumprod.to(device)
    return self

  def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    a = self.alphas_cumprod[t][:, None, None, None]
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

  @torch.no_grad()
  def sample(self, model: DiffusionUNet, cond: torch.Tensor) -> torch.Tensor:
    device = cond.device
    b, _, h, w = cond.shape
    x = torch.randn(b, 1, h, w, device=device)
    for step in reversed(range(self.timesteps)):
      t = torch.full((b,), step, device=device, dtype=torch.long)
      eps = model(x, cond, t)
      alpha = self.alphas_cumprod[step]
      alpha_prev = self.alphas_cumprod[step - 1] if step > 0 else torch.tensor(1.0, device=device)
      beta = 1 - alpha / alpha_prev if step > 0 else 1 - alpha
      x = (1 / torch.sqrt(1 - beta)) * (x - beta / torch.sqrt(1 - alpha) * eps)
      if step > 0:
        x = x + torch.sqrt(beta) * torch.randn_like(x)
    return x
