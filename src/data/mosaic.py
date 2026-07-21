"""
Построение мозаики сейсмики и нарезка окон для обучения.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.config import HEIGHT, WIDTH, VP_DX_METERS


@dataclass
class ShotGather:
  shot_id: int
  source_x_model: float
  receiver_x_model: np.ndarray
  traces_tx: np.ndarray  # [T, n_rec]


def build_model_x(nx: int, dx: float = VP_DX_METERS) -> np.ndarray:
  return np.arange(nx, dtype=np.float64) * dx


def resize_velocity(vp_zx: np.ndarray, target_t: int = HEIGHT) -> np.ndarray:
  """Приводит [nz, nx] к [target_t, nx]."""
  v_t = torch.from_numpy(vp_zx).unsqueeze(0).unsqueeze(0)
  v_t = F.interpolate(v_t, size=(target_t, vp_zx.shape[1]), mode="bilinear", align_corners=False)
  return v_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def shots_from_numpy(
  traces: np.ndarray,
  headers: Dict[str, np.ndarray],
) -> List[ShotGather]:
  """Собирает shot gathers из NumPy + заголовков."""
  shot_id = headers["shot_id"]
  gx = headers.get("group_x_m", headers["group_x"].astype(np.float64))
  sx = headers.get("source_x_m", headers["source_x"].astype(np.float64))

  shots: List[ShotGather] = []
  for sh in np.unique(shot_id):
    idx = np.where(shot_id == sh)[0]
    gx_sh = gx[idx]
    sx_sh = sx[idx]
    tr = traces[idx].T.astype(np.float32)  # [T, n_rec]

    order = np.argsort(gx_sh)
    shots.append(ShotGather(
      shot_id=int(sh),
      source_x_model=float(np.median(sx_sh)),
      receiver_x_model=gx_sh[order].astype(np.float64),
      traces_tx=tr[:, order],
    ))

  shots.sort(key=lambda s: s.shot_id)
  return shots


def build_interpolated_mosaic(
  shots: List[ShotGather],
  model_x: np.ndarray,
  t_size: int = HEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
  """Интерполирует shot gathers на сетку model_x → [T, nx]."""
  nx = len(model_x)
  x_sum = np.zeros((t_size, nx), dtype=np.float64)
  x_cnt = np.zeros(nx, dtype=np.float64)

  for s in shots:
    gx = s.receiver_x_model.astype(np.float64)
    tr = s.traces_tx.astype(np.float32)

    if tr.shape[0] != t_size:
      t = torch.from_numpy(tr).unsqueeze(0).unsqueeze(0)
      t = F.interpolate(t, size=(t_size, tr.shape[1]), mode="bilinear", align_corners=False)
      tr = t.squeeze(0).squeeze(0).cpu().numpy()

    inside = (model_x >= gx[0]) & (model_x <= gx[-1])
    if not np.any(inside):
      continue

    x_target = model_x[inside]
    interp = np.vstack([np.interp(x_target, gx, tr[it]) for it in range(t_size)])
    x_sum[:, inside] += interp
    x_cnt[inside] += 1.0

  covered = x_cnt > 0
  mosaic = np.zeros((t_size, nx), dtype=np.float32)
  mosaic[:, covered] = (x_sum[:, covered] / x_cnt[covered]).astype(np.float32)
  return mosaic, covered.astype(np.float32)


def extract_windows(
  x_mosaic: np.ndarray,
  y_full: np.ndarray,
  coverage_mask: np.ndarray,
  window_w: int = WIDTH,
  stride: int | None = None,
  min_cov: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Нарезает мозаику на окна [N, 1, T, W]."""
  stride = stride or max(1, window_w // 4)
  _, nx = x_mosaic.shape
  starts = list(range(0, nx - window_w + 1, stride))

  x_list: List[np.ndarray] = []
  y_list: List[np.ndarray] = []
  meta: List[Dict] = []

  for s in starts:
    e = s + window_w
    cov = float(np.mean(coverage_mask[s:e]))
    if cov < min_cov:
      continue
    x_list.append(x_mosaic[:, s:e])
    y_list.append(y_full[:, s:e])
    meta.append({"x_start_idx": int(s), "x_end_idx": int(e - 1), "coverage": cov})

  if not x_list:
    raise RuntimeError("Не удалось сформировать окна с достаточным покрытием")

  x_arr = np.stack(x_list, axis=0)[:, None, :, :].astype(np.float32)
  y_arr = np.stack(y_list, axis=0)[:, None, :, :].astype(np.float32)
  return x_arr, y_arr, np.array(meta, dtype=object)


def spatial_split_indices(n: int, seed: int = 42, mode: str = "spatial") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  idx = np.arange(n)
  if mode == "random":
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
  elif mode != "spatial":
    raise ValueError(f"Unknown split mode: {mode}")

  n_train = int(round(n * 0.7))
  n_val = int(round(n * 0.2))
  n_test = n - n_train - n_val
  if min(n_train, n_val, n_test) <= 0:
    raise ValueError(f"Too few windows ({n}) for 70/20/10 split")

  return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def stitch_windows(
  windows: np.ndarray,
  meta: np.ndarray,
  full_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
  stitched_sum = np.zeros(full_shape, dtype=np.float64)
  stitched_count = np.zeros(full_shape, dtype=np.float64)

  for win, item in zip(windows, meta):
    start = int(item["x_start_idx"])
    end = int(item["x_end_idx"]) + 1
    stitched_sum[:, start:end] += win
    stitched_count[:, start:end] += 1.0

  mask = stitched_count > 0
  stitched = np.full(full_shape, np.nan, dtype=np.float32)
  stitched[mask] = (stitched_sum[mask] / stitched_count[mask]).astype(np.float32)
  return stitched, mask
