"""
Объединение всех пар «сейсмика + модель Vp» в единый обучающий набор.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
  DATASET_PAIRS,
  HEIGHT,
  SPLIT_RANDOM_STATE,
  USE_ALL_DATASETS,
  VP_DX_METERS,
  WIDTH,
)
from src.data.mosaic import (
  build_interpolated_mosaic,
  build_model_x,
  extract_windows,
  resize_velocity,
  shots_from_numpy,
  spatial_split_indices,
)
from src.data.numpy_store import load_seismic_with_headers, load_velocity_model

logger = logging.getLogger("fwi.data.combined")


def get_dataset_pairs(families: Optional[List[str]] = None) -> List[dict]:
  """Возвращает список пар для обучения. families: ['zoloto','domanic'] или None = все."""
  pairs = DATASET_PAIRS
  if families:
    pairs = [p for p in pairs if p.get("family") in families]
  return pairs


def prepare_mosaic_windows_for_pair(
  pair: dict,
  target_t: int = HEIGHT,
  window_w: int = WIDTH,
  window_stride: int | None = None,
  min_cov: float = 0.5,
  vp_dx_meters: float = VP_DX_METERS,
) -> Dict[str, object]:
  """Строит окна для одной пары seismic + velocity_model."""
  seismic_key = pair["seismic"]
  vp_key = pair["velocity_model"]
  pair_id = pair["id"]

  logger.info(
    "Pair '%s' | seismic=%s | vp_model=%s | window=%dx%d",
    pair_id, seismic_key, vp_key, target_t, window_w,
  )

  seismic, headers, seismic_meta = load_seismic_with_headers(dataset_key=seismic_key)
  vp_raw, vp_meta = load_velocity_model(dataset_key=vp_key)
  Y_full = resize_velocity(vp_raw, target_t=target_t)
  model_x = build_model_x(Y_full.shape[1], dx=vp_dx_meters)

  shots = shots_from_numpy(seismic, headers)
  logger.info(
    "  shots=%d | Vp=%s | X: 0..%.1f m (%d traces)",
    len(shots), Y_full.shape, float(model_x[-1]), len(model_x),
  )

  X_mosaic, coverage_mask = build_interpolated_mosaic(shots, model_x, t_size=target_t)
  cov = float(np.mean(coverage_mask))
  logger.info("  mosaic coverage=%.4f", cov)

  stride = window_stride or max(1, window_w // 4)
  try:
    X_w, Y_w, meta = extract_windows(
      X_mosaic, Y_full, coverage_mask,
      window_w=window_w, stride=stride, min_cov=min_cov,
    )
  except RuntimeError:
    logger.warning("  pair %s: min_cov=%.2f failed, retry with 0", pair_id, min_cov)
    X_w, Y_w, meta = extract_windows(
      X_mosaic, Y_full, coverage_mask,
      window_w=window_w, stride=stride, min_cov=0.0,
    )

  enriched_meta = []
  for i, m in enumerate(meta):
    item = dict(m)
    item["dataset_id"] = pair_id
    item["family"] = pair.get("family", pair_id)
    item["seismic_dataset"] = seismic_key
    item["velocity_model"] = vp_key
    item["source_window_idx"] = int(i)
    item["x_start_m"] = float(model_x[int(item["x_start_idx"])])
    enriched_meta.append(item)

  logger.info("  windows=%d | stride=%d", len(X_w), stride)

  return {
    "pair_id": pair_id,
    "X_mosaic": X_mosaic,
    "Y_full": Y_full,
    "model_x": model_x,
    "coverage_mask": coverage_mask,
    "shots": shots,
    "X_w": X_w,
    "Y_w": Y_w,
    "meta": np.array(enriched_meta, dtype=object),
    "seismic_meta": seismic_meta,
    "vp_meta": vp_meta,
  }


def prepare_combined_mosaic_windows(
  pairs: Optional[List[dict]] = None,
  target_t: int = HEIGHT,
  window_w: int = WIDTH,
  window_stride: int | None = None,
  min_cov: float = 0.5,
) -> Dict[str, object]:
  """
  Объединяет окна из всех пар в единые массивы X_w, Y_w, meta.

  Возвращает также sources: dict[pair_id] -> pack одного источника.
  """
  pairs = pairs or get_dataset_pairs()
  if not pairs:
    raise RuntimeError("Нет пар датасетов для объединения")

  logger.info("=" * 50)
  logger.info("Combined dataset | %d pair(s)", len(pairs))

  x_parts: List[np.ndarray] = []
  y_parts: List[np.ndarray] = []
  meta_parts: List[np.ndarray] = []
  sources: Dict[str, dict] = {}

  for pair in pairs:
    pack = prepare_mosaic_windows_for_pair(
      pair, target_t=target_t, window_w=window_w,
      window_stride=window_stride, min_cov=min_cov,
      vp_dx_meters=pair.get("vp_dx_meters", VP_DX_METERS),
    )
    sources[pair["id"]] = pack
    x_parts.append(pack["X_w"])
    y_parts.append(pack["Y_w"])
    meta_parts.append(pack["meta"])

  X_w = np.concatenate(x_parts, axis=0)
  Y_w = np.concatenate(y_parts, axis=0)
  meta = np.concatenate(meta_parts, axis=0)

  vp_positive = Y_w[Y_w > 0]
  y_vmin = float(vp_positive.min()) if vp_positive.size else 0.0
  y_vmax = float(Y_w.max())

  counts = {}
  for m in meta:
    counts[m["dataset_id"]] = counts.get(m["dataset_id"], 0) + 1

  logger.info(
    "Combined total: %d windows | Vp range [%.1f, %.1f] m/s | per source: %s",
    len(X_w), y_vmin, y_vmax, counts,
  )

  return {
    "X_w": X_w,
    "Y_w": Y_w,
    "meta": meta,
    "sources": sources,
    "pairs": pairs,
    "y_vmin": y_vmin,
    "y_vmax": y_vmax,
    "n_windows_per_source": counts,
  }


def combined_split_indices(
  meta: np.ndarray,
  mode: str = "spatial",
  seed: int = SPLIT_RANDOM_STATE,
  train_ratio: float = 0.7,
  val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Train/val/test split по объединённому набору.

  spatial: внутри каждого dataset_id — пространственный split, затем объединение.
  random:  shuffle всех окон вместе.
  """
  n = len(meta)
  if n < 3:
    raise ValueError(f"Too few windows for split: {n}")

  if mode == "random":
    return spatial_split_indices(n, seed=seed, mode="random")

  by_source: Dict[str, List[int]] = {}
  for i, m in enumerate(meta):
    by_source.setdefault(m["dataset_id"], []).append(i)

  train_idx: List[int] = []
  val_idx: List[int] = []
  test_idx: List[int] = []

  for source_id, indices in sorted(by_source.items()):
    idx = np.array(indices)
    n_s = len(idx)
    tr, va, te = spatial_split_indices(n_s, seed=seed, mode="spatial")
    train_idx.extend(idx[tr].tolist())
    val_idx.extend(idx[va].tolist())
    test_idx.extend(idx[te].tolist())
    logger.info(
      "Split '%s' (%s): train=%d val=%d test=%d",
      source_id, mode, len(tr), len(va), len(te),
    )

  return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def resolve_training_pairs(
  use_all: Optional[bool] = None,
  family: Optional[str] = None,
  single_pair: Optional[dict] = None,
) -> List[dict]:
  """
  Определяет список пар для текущего запуска.

  use_all=True  -> все DATASET_PAIRS
  family='zoloto' -> только пары zoloto (обе сейсмики)
  single_pair   -> одна явная пара
  """
  if single_pair is not None:
    return [single_pair]
  if use_all if use_all is not None else USE_ALL_DATASETS:
    if family:
      return get_dataset_pairs(families=[family])
    return get_dataset_pairs()
  from src.config import ACTIVE_SEISMIC_DATASET, ACTIVE_VELOCITY_MODEL
  return [{
    "id": "active",
    "seismic": ACTIVE_SEISMIC_DATASET,
    "velocity_model": ACTIVE_VELOCITY_MODEL,
    "family": family or "active",
  }]
