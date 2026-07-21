"""
PINN для обратной задачи FWI.

Архитектура:
  - P-Net: (x, z, t) → u(x, z, t)  — волновое поле
  - V-Net: (x, z)     → v(x, z)    — скорость P-волны
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.config import PINN_ACTIVATION, PINN_HIDDEN_DIMS


def _get_activation(name: str) -> nn.Module:
  mapping = {
    "relu": nn.ReLU(inplace=True),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
    "sin": _Sine(),
  }
  if name not in mapping:
    raise ValueError(f"Unknown activation: {name}")
  return mapping[name]


class _Sine(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


class FourierFeatures(nn.Module):
  """Fourier feature mapping для лучшего обучения высоких частот."""

  def __init__(self, in_dim: int, n_freq: int = 32, scale: float = 1.0) -> None:
    super().__init__()
    B = torch.randn(in_dim, n_freq) * scale
    self.register_buffer("B", B)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    proj = 2.0 * torch.pi * x @ self.B
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _build_mlp(in_dim: int, out_dim: int, hidden: Tuple[int, ...], activation: str) -> nn.Sequential:
  layers: list[nn.Module] = []
  d = in_dim
  for h in hidden:
    layers += [nn.Linear(d, h), _get_activation(activation)]
    d = h
  layers.append(nn.Linear(d, out_dim))
  return nn.Sequential(*layers)


class FWI_PINN(nn.Module):
  """
  Physics-Informed Neural Network для FWI.

  forward(coords) → (u, v) при coords [N, 3] = (x, z, t).
  Для скорости используются только (x, z).
  """

  def __init__(
    self,
    hidden_dims: Tuple[int, ...] = PINN_HIDDEN_DIMS,
    activation: str = PINN_ACTIVATION,
    vp_min: float = 1500.0,
    vp_max: float = 4500.0,
    use_fourier: bool = True,
    n_fourier: int = 32,
  ) -> None:
    super().__init__()
    self.vp_min = vp_min
    self.vp_max = vp_max

    self.ff_p = FourierFeatures(3, n_freq=n_fourier) if use_fourier else nn.Identity()
    self.ff_v = FourierFeatures(2, n_freq=n_fourier) if use_fourier else nn.Identity()

    p_in = n_fourier * 2 if use_fourier else 3
    v_in = n_fourier * 2 if use_fourier else 2

    self.p_net = _build_mlp(p_in, 1, hidden_dims, activation)
    self.v_net = _build_mlp(v_in, 1, hidden_dims, activation)

  def predict_velocity(self, xz: torch.Tensor) -> torch.Tensor:
    """xz: [N, 2] → v [N, 1] в [vp_min, vp_max]."""
    feat = self.ff_v(xz) if isinstance(self.ff_v, FourierFeatures) else xz
    raw = self.v_net(feat)
    return self.vp_min + (self.vp_max - self.vp_min) * torch.sigmoid(raw)

  def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """coords [N, 3] → (u, v) каждый [N, 1]."""
    xz = coords[:, :2]
    v = self.predict_velocity(xz)

    feat = self.ff_p(coords) if isinstance(self.ff_p, FourierFeatures) else coords
    u = self.p_net(feat)
    return u, v
