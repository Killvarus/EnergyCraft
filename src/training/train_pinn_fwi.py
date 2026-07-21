"""
Обучение PINN для FWI на одном пространственном окне.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
  DEVICE, GRAD_CLIP, LR, LOG_INTERVAL, LOSS_FN, MAX_EPOCHS, N_EPOCH, OPTIMIZER,
  PATIENCE, PINN_DX, PINN_DZ, PINN_DT, PINN_LAMBDA_BC, PINN_LAMBDA_DATA,
  PINN_LAMBDA_PHYS, PINN_LAMBDA_VP, PINN_N_BOUNDARY, PINN_N_COLLOC,
  PINN_N_RECEIVERS, PINN_SOURCE_FREQ, WEIGHT_DECAY,
)
from src.models.pinn_fwi import FWI_PINN
from src.models.physics import (
  acoustic_wave_residual,
  open_boundary_residual,
  source_term,
  velocity_smoothness,
)
from src.training.early_stopping import EarlyStopConfig, EarlyStopState, relative_improvement, update_early_stop
from src.training.tensorboard import TrainingTracker

logger = logging.getLogger("pinn_fwi")


def _optimizer(params, lr: float) -> optim.Optimizer:
  opt = OPTIMIZER.lower()
  if opt == "adam":
    return optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
  if opt == "adamw":
    return optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
  raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")


def _criterion() -> nn.Module:
  return nn.MSELoss() if LOSS_FN.lower() == "mse" else nn.L1Loss()


def _sample_collocation(
  x0: float, x1: float, z0: float, z1: float, t0: float, t1: float,
  n: int, device: torch.device,
) -> torch.Tensor:
  x = x0 + torch.rand(n, 1, device=device) * (x1 - x0)
  z = z0 + torch.rand(n, 1, device=device) * (z1 - z0)
  t = t0 + torch.rand(n, 1, device=device) * (t1 - t0)
  return torch.cat([x, z, t], dim=1).requires_grad_(True)


def _sample_boundary(
  x0: float, x1: float, z0: float, z1: float, t0: float, t1: float,
  n_per_side: int, device: torch.device,
) -> Dict[str, torch.Tensor]:
  def side(boundary: str) -> torch.Tensor:
    n = n_per_side
    t = t0 + torch.rand(n, 1, device=device) * (t1 - t0)
    if boundary == "x_min":
      x, z = torch.full((n, 1), x0, device=device), z0 + torch.rand(n, 1, device=device) * (z1 - z0)
    elif boundary == "x_max":
      x, z = torch.full((n, 1), x1, device=device), z0 + torch.rand(n, 1, device=device) * (z1 - z0)
    elif boundary == "z_min":
      x, z = x0 + torch.rand(n, 1, device=device) * (x1 - x0), torch.full((n, 1), z0, device=device)
    else:
      x, z = x0 + torch.rand(n, 1, device=device) * (x1 - x0), torch.full((n, 1), z1, device=device)
    return torch.cat([x, z, t], dim=1).requires_grad_(True)

  return {b: side(b) for b in ("x_min", "x_max", "z_min", "z_max")}


def _sample_receivers(
  observed: np.ndarray,
  x_start: float,
  dx: float,
  dz: float,
  dt: float,
  n: int,
  device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Случайная подвыборка точек наблюдения из окна observed [T, W].
  Возвращает coords [N, 3] и values [N, 1].
  """
  t_size, w_size = observed.shape
  t_idx = np.random.randint(0, t_size, size=n)
  x_idx = np.random.randint(0, w_size, size=n)

  x = x_start + x_idx.astype(np.float32) * dx
  z = t_idx.astype(np.float32) * dz
  t = t_idx.astype(np.float32) * dt
  vals = observed[t_idx, x_idx].astype(np.float32)

  coords = torch.tensor(np.stack([x, z, t], axis=1), dtype=torch.float32, device=device)
  values = torch.tensor(vals[:, None], dtype=torch.float32, device=device)
  return coords.requires_grad_(True), values


def train_pinn_window(
  model: FWI_PINN,
  observed_window: np.ndarray,
  x_start: float,
  x_extent: float,
  z_extent: float,
  max_epochs: int = MAX_EPOCHS,
  lr: float = LR,
  rel_patience: float = PATIENCE,
  n_epoch: int = N_EPOCH,
  vp_min: float = 1500.0,
  vp_max: float = 4500.0,
  device: torch.device = DEVICE,
  tb_run_name: str | None = None,
) -> Tuple[FWI_PINN, Dict]:
  """
  Обучает PINN на одном окне сейсмики.

  observed_window: [T, W] — нормализованная амплитуда.
  x_start: начало окна по X [м].
  """
  model = model.to(device)
  criterion = _criterion()
  optimizer = _optimizer(model.parameters(), lr=lr)

  x0, x1 = x_start, x_start + x_extent
  z0, z1 = 0.0, z_extent
  t0, t1 = 0.0, observed_window.shape[0] * PINN_DT
  x_src = x0 + x_extent * 0.5
  z_src = z0 + 5.0

  losses: List[float] = []
  best_state = copy.deepcopy(model.state_dict())
  es_cfg = EarlyStopConfig(patience=rel_patience, n_epoch=n_epoch, max_epochs=max_epochs)
  es = EarlyStopState()

  t_start = time.time()
  n_side = max(PINN_N_BOUNDARY // 4, 1)
  n_params = sum(p.numel() for p in model.parameters())

  logger.info(
    "PINN window train | params=%d | max_epochs=%s | lr=%.5f | x=[%.1f,%.1f] z=[%.1f,%.1f] | "
    "rel_patience=%.2e | n_epoch=%d | lambda_data=%.3f phys=%.3f bc=%.3f vp=%.3f | log_every=%d",
    n_params, "unlimited" if max_epochs <= 0 else str(max_epochs), lr, x0, x1, z0, z1,
    rel_patience, n_epoch,
    PINN_LAMBDA_DATA, PINN_LAMBDA_PHYS, PINN_LAMBDA_BC, PINN_LAMBDA_VP, LOG_INTERVAL,
  )

  last_components: Dict[str, float] = {}
  epoch = 0
  run_name = tb_run_name or TrainingTracker.make_run_name("pinn_window")
  tracker = TrainingTracker(run_name)

  try:
    while True:
      if max_epochs > 0 and epoch >= max_epochs:
        logger.info("PINN stop: reached max_epochs=%d", max_epochs)
        break

      model.train()
      optimizer.zero_grad(set_to_none=True)

      col = _sample_collocation(x0, x1, z0, z1, t0, t1, PINN_N_COLLOC, device)
      u_c, v_c = model(col)
      src = source_term(col, x_src, z_src, f0=PINN_SOURCE_FREQ)
      phys = acoustic_wave_residual(u_c, v_c, col, source=src)
      loss_phys = torch.mean(phys ** 2)

      rec_coords, rec_vals = _sample_receivers(
        observed_window, x_start, PINN_DX, PINN_DZ, PINN_DT, PINN_N_RECEIVERS, device,
      )
      u_r, _ = model(rec_coords)
      loss_data = criterion(u_r, rec_vals)

      bc_losses = []
      for bname, bcoords in _sample_boundary(x0, x1, z0, z1, t0, t1, n_side, device).items():
        u_b, v_b = model(bcoords)
        bc_losses.append(torch.mean(open_boundary_residual(u_b, v_b, bcoords, bname) ** 2))
      loss_bc = torch.stack(bc_losses).mean()

      xz_grid = _sample_collocation(x0, x1, z0, z1, t0, t0, PINN_N_COLLOC // 4, device)[:, :2].requires_grad_(True)
      v_s = model.predict_velocity(xz_grid)
      loss_vp = torch.mean(velocity_smoothness(v_s, xz_grid))

      total = (
        PINN_LAMBDA_DATA * loss_data
        + PINN_LAMBDA_PHYS * loss_phys
        + PINN_LAMBDA_BC * loss_bc
        + PINN_LAMBDA_VP * loss_vp
      )
      total.backward()

      if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
      optimizer.step()

      loss_val = float(total.item())
      losses.append(loss_val)
      last_components = {
        "data": float(loss_data),
        "phys": float(loss_phys),
        "bc": float(loss_bc),
        "vp": float(loss_vp),
      }

      rel_imp = relative_improvement(es.best_loss, loss_val)
      es = update_early_stop(es, loss_val, epoch, es_cfg)
      improved = rel_imp >= rel_patience
      if es.best_epoch == epoch:
        best_state = copy.deepcopy(model.state_dict())

      epoch_num = epoch + 1
      tracker.scalar("epoch/total_loss", loss_val, epoch_num)
      tracker.scalar("epoch/data_loss", last_components["data"], epoch_num)
      tracker.scalar("epoch/phys_loss", last_components["phys"], epoch_num)
      tracker.scalar("epoch/bc_loss", last_components["bc"], epoch_num)
      tracker.scalar("epoch/vp_loss", last_components["vp"], epoch_num)
      tracker.scalar("epoch/best_loss", es.best_loss, epoch_num)
      if rel_imp != float("inf") and rel_imp != float("-inf"):
        tracker.scalar("early_stopping/relative_improvement", rel_imp, epoch_num)
      tracker.scalar("early_stopping/no_improve_epochs", es.bad_epochs, epoch_num)

      if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
        logger.info(
          "PINN epoch %4d | total=%.4e | data=%.4e | phys=%.4e | bc=%.4e | vp=%.4e | "
          "best=%.4e | rel_imp=%.2e | no_improve=%d/%d%s",
          epoch + 1, loss_val,
          last_components["data"], last_components["phys"], last_components["bc"], last_components["vp"],
          es.best_loss, rel_imp, es.bad_epochs, n_epoch, " *" if improved else "",
        )

      if es.stopped:
        logger.info(
          "PINN early stop epoch %d | %s | best_loss=%.4e",
          epoch + 1, es.stop_reason, es.best_loss,
        )
        break

      epoch += 1
  finally:
    tracker.close()

  model.load_state_dict(best_state)
  history = {
    "losses": losses,
    "best_loss": es.best_loss,
    "best_epoch": es.best_epoch,
    "early_stop_reason": es.stop_reason,
    "time": time.time() - t_start,
    "vp_range": [vp_min, vp_max],
    "last_components": last_components,
  }
  logger.info(
    "PINN window done | epochs=%d | best_epoch=%d | best_loss=%.4e | time=%.1fs",
    len(losses), es.best_epoch + 1, es.best_loss, history["time"],
  )
  return model, history


@torch.no_grad()
def predict_velocity_grid(
  model: FWI_PINN,
  x_start: float,
  width: int,
  height: int,
  dx: float = PINN_DX,
  dz: float = PINN_DZ,
  device: torch.device = DEVICE,
) -> np.ndarray:
  """Предсказывает v(x,z) на сетке окна [height, width]."""
  model.eval()
  xs = x_start + np.arange(width, dtype=np.float32) * dx
  zs = np.arange(height, dtype=np.float32) * dz
  xx, zz = np.meshgrid(xs, zs, indexing="xy")
  xz = torch.tensor(np.stack([xx.ravel(), zz.ravel()], axis=1), dtype=torch.float32, device=device)
  v = model.predict_velocity(xz).cpu().numpy().reshape(height, width)
  return v.astype(np.float32)
