"""
Физические функции потерь для PINN-FWI.

Акустическое волновое уравнение (2D):

    (1 / v(x,z)²) · ∂²u/∂t²  −  ∂²u/∂x²  −  ∂²u/∂z²  =  f(x,z,t)

Эквивалентная невязка:

    r = (1/v²)·u_tt − u_xx − u_zz − f
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def ricker_wavelet(
  t: torch.Tensor,
  f0: float = 15.0,
  t0: Optional[float] = None,
) -> torch.Tensor:
  """Ricker wavelet f(t) = (1 - 2π²f₀²(t-t₀)²) · exp(-π²f₀²(t-t₀)²)."""
  if t0 is None:
    t0 = 1.5 / f0
  tau = t - t0
  pf2 = (torch.pi * f0) ** 2
  return (1.0 - 2.0 * pf2 * tau ** 2) * torch.exp(-pf2 * tau ** 2)


def source_term(
  coords: torch.Tensor,
  x_src: float,
  z_src: float,
  f0: float = 15.0,
  sigma: float = 2.0,
) -> torch.Tensor:
  """
  Источник f(x,z,t) = wavelet(t) · exp(-r²/σ²), r² = (x-x_src)² + (z-z_src)².
  coords: [N, 3] — (x, z, t) в физических единицах.
  """
  x, z, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
  r2 = (x - x_src) ** 2 + (z - z_src) ** 2
  spatial = torch.exp(-r2 / (sigma ** 2))
  return ricker_wavelet(t, f0=f0) * spatial


def acoustic_wave_residual(
  u: torch.Tensor,
  v: torch.Tensor,
  coords: torch.Tensor,
  source: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """
  Невязка PDE: (1/v²)·u_tt − u_xx − u_zz − f.

  u, v: [N, 1]; coords: [N, 3] с requires_grad=True.
  """
  if not coords.requires_grad:
    coords = coords.clone().detach().requires_grad_(True)

  grads = torch.autograd.grad(
    u, coords, grad_outputs=torch.ones_like(u),
    create_graph=True, retain_graph=True,
  )[0]
  u_x, u_z, u_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]

  u_xx = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
  u_zz = torch.autograd.grad(u_z, coords, grad_outputs=torch.ones_like(u_z), create_graph=True, retain_graph=True)[0][:, 1:2]
  u_tt = torch.autograd.grad(u_t, coords, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True)[0][:, 2:3]

  v_safe = torch.clamp(v, min=300.0)
  inv_v2 = 1.0 / (v_safe ** 2)
  f = source if source is not None else torch.zeros_like(u)

  residual = inv_v2 * u_tt - u_xx - u_zz - f
  return residual


def open_boundary_residual(
  u: torch.Tensor,
  v: torch.Tensor,
  coords: torch.Tensor,
  boundary: str,
) -> torch.Tensor:
  """Первый порядок: u_t ± c·u_n = 0 на открытых границах."""
  if not coords.requires_grad:
    coords = coords.clone().detach().requires_grad_(True)

  grads = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
  u_x, u_z, u_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]
  c = torch.clamp(v, min=300.0)

  if boundary == "x_min":
    return u_t - c * u_x
  if boundary == "x_max":
    return u_t + c * u_x
  if boundary == "z_min":
    return u_t - c * u_z
  if boundary == "z_max":
    return u_t + c * u_z
  raise ValueError(f"Unknown boundary: {boundary}")


def velocity_smoothness(v: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
  """TV-подобная гладкость скорости: ||∇v||²."""
  if not coords.requires_grad:
    coords = coords.clone().detach().requires_grad_(True)
  grads = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
  return grads[:, 0:1] ** 2 + grads[:, 1:2] ** 2
