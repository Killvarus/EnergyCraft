"""
Загрузка данных, нормализация, построение DataLoader.
"""
from __future__ import annotations

from src.data.combined import (
  combined_split_indices,
  prepare_combined_mosaic_windows,
  prepare_mosaic_windows_for_pair,
  resolve_training_pairs,
)
from src.data.mosaic import (
  ShotGather,
  build_interpolated_mosaic,
  build_model_x,
  extract_windows,
  resize_velocity,
  shots_from_numpy,
  spatial_split_indices,
  stitch_windows,
)
from src.data.numpy_store import (
  load_array,
  load_manifest,
  load_seismic_with_headers,
  load_velocity_model,
)
from src.data.sgy_converter import convert_all_sgy_in_directory, convert_sgy_file

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
  VP_MIN, VP_MAX, HEIGHT, WIDTH, MLP_BATCH_SIZE,
  SPLIT_TRAIN_RATIO, SPLIT_VAL_RATIO, SPLIT_TEST_RATIO, SPLIT_RANDOM_STATE,
)

logger = logging.getLogger("fwi.data")


def normalize_amplitude(X: np.ndarray) -> np.ndarray:
  max_abs = np.max(np.abs(X))
  return X if max_abs == 0 else X / max_abs


def minmax_normalize(Y: np.ndarray, vmin: float = VP_MIN, vmax: float = VP_MAX) -> np.ndarray:
  return (Y - vmin) / (vmax - vmin)


def minmax_denormalize(Y_norm: np.ndarray, vmin: float = VP_MIN, vmax: float = VP_MAX) -> np.ndarray:
  return Y_norm * (vmax - vmin) + vmin


def build_data_loaders(X: np.ndarray, Y: np.ndarray, batch_size: int = MLP_BATCH_SIZE) -> DataLoader:
  dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
  return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_train_val_test_loaders(
  X: np.ndarray,
  Y: np.ndarray,
  batch_size: int = MLP_BATCH_SIZE,
  train_ratio: float = SPLIT_TRAIN_RATIO,
  val_ratio: float = SPLIT_VAL_RATIO,
  test_ratio: float = SPLIT_TEST_RATIO,
  random_state: int = SPLIT_RANDOM_STATE,
) -> Dict[str, object]:
  total = train_ratio + val_ratio + test_ratio
  if not np.isclose(total, 1.0):
    raise ValueError(f"Split ratios must sum to 1.0, got {total}")

  n_samples = len(X)
  rng = np.random.default_rng(random_state)
  perm = rng.permutation(n_samples)
  X_shuffled, Y_shuffled = X[perm], Y[perm]
  idx = np.arange(n_samples)

  train_val_idx, test_idx = train_test_split(idx, test_size=test_ratio, random_state=random_state, shuffle=False)
  val_ratio_in_train_val = val_ratio / (train_ratio + val_ratio)
  train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio_in_train_val, random_state=random_state, shuffle=False)

  X_train, Y_train = X_shuffled[train_idx], Y_shuffled[train_idx]
  X_val, Y_val = X_shuffled[val_idx], Y_shuffled[val_idx]
  X_test, Y_test = X_shuffled[test_idx], Y_shuffled[test_idx]

  train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()), batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float()), batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float()), batch_size=batch_size, shuffle=False)

  return {
    "train_loader": train_loader,
    "val_loader": val_loader,
    "test_loader": test_loader,
    "X_train": X_train, "Y_train": Y_train,
    "X_val": X_val, "Y_val": Y_val,
    "X_test": X_test, "Y_test": Y_test,
    "train_idx": perm[train_idx],
    "val_idx": perm[val_idx],
    "test_idx": perm[test_idx],
  }


def prepare_training_windows(
  target_t: int = HEIGHT,
  window_w: int = WIDTH,
  window_stride: int | None = None,
  min_cov: float = 0.5,
  use_all: bool | None = None,
  family: str | None = None,
  seismic_dataset_key: str | None = None,
) -> Dict[str, object]:
  """
  Единая точка входа: все пары или одна пара / один пресет.

  use_all=True (default из USE_ALL_DATASETS) — все сейсмики + модели.
  family='zoloto'|'domanic' — только семейство.
  seismic_dataset_key — одна конкретная сейсмика (с парной Vp из DATASET_PAIRS).
  """
  from src.config import ACTIVE_VELOCITY_MODEL, DATASET_PAIRS, USE_ALL_DATASETS
  from src.data.combined import (
    prepare_combined_mosaic_windows,
    prepare_mosaic_windows_for_pair,
    resolve_training_pairs,
  )

  stride = window_stride or max(1, window_w // 4)

  if seismic_dataset_key:
    match = [p for p in DATASET_PAIRS if p["seismic"] == seismic_dataset_key]
    if not match:
      pair = {
        "id": "custom",
        "seismic": seismic_dataset_key,
        "velocity_model": ACTIVE_VELOCITY_MODEL,
        "family": "custom",
      }
    else:
      pair = match[0]
    pack = prepare_mosaic_windows_for_pair(
      pair, target_t=target_t, window_w=window_w, window_stride=stride, min_cov=min_cov,
    )
    vp_pos = pack["Y_w"][pack["Y_w"] > 0]
    return {
      **pack,
      "sources": {pair["id"]: pack},
      "pairs": [pair],
      "y_vmin": float(vp_pos.min()) if vp_pos.size else 0.0,
      "y_vmax": float(pack["Y_w"].max()),
      "n_windows_per_source": {pair["id"]: len(pack["X_w"])},
      "combined": False,
    }

  pairs = resolve_training_pairs(use_all=use_all, family=family)
  if len(pairs) == 1:
    pack = prepare_mosaic_windows_for_pair(
      pairs[0], target_t=target_t, window_w=window_w, window_stride=stride, min_cov=min_cov,
    )
    vp_pos = pack["Y_w"][pack["Y_w"] > 0]
    return {
      **pack,
      "sources": {pairs[0]["id"]: pack},
      "pairs": pairs,
      "y_vmin": float(vp_pos.min()) if vp_pos.size else 0.0,
      "y_vmax": float(pack["Y_w"].max()),
      "n_windows_per_source": {pairs[0]["id"]: len(pack["X_w"])},
      "combined": False,
    }

  combined = prepare_combined_mosaic_windows(
    pairs=pairs, target_t=target_t, window_w=window_w, window_stride=stride, min_cov=min_cov,
  )
  combined["combined"] = True
  return combined


def prepare_mosaic_windows(
  target_t: int = HEIGHT,
  window_w: int = WIDTH,
  window_stride: int | None = None,
  min_cov: float = 0.5,
  seismic_dataset_key: str | None = None,
) -> Dict[str, object]:
  """Обратная совместимость → prepare_training_windows."""
  return prepare_training_windows(
    target_t=target_t,
    window_w=window_w,
    window_stride=window_stride,
    min_cov=min_cov,
    seismic_dataset_key=seismic_dataset_key,
    use_all=False,
  )
