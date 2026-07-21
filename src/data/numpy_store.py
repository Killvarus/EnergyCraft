"""
Загрузка предобработанных NumPy-датасетов из data/processed/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.config import ACTIVE_SEISMIC_DATASET, ACTIVE_VELOCITY_MODEL, PROCESSED_DATA_DIR

logger = logging.getLogger("fwi.data.numpy_store")


def load_manifest(processed_dir: Optional[Path] = None) -> Dict[str, Any]:
  processed_dir = Path(processed_dir or PROCESSED_DATA_DIR)
  manifest_path = processed_dir / "manifest.json"
  if not manifest_path.exists():
    raise FileNotFoundError(
      f"Manifest not found: {manifest_path}. Run: python scripts/convert_sgy_to_numpy.py"
    )
  with open(manifest_path, encoding="utf-8") as f:
    return json.load(f)


def _resolve_dataset_key(manifest: Dict[str, Any], role: str) -> str:
  """role: seismic | seismic_alt | velocity_model"""
  if role == "seismic":
    return ACTIVE_SEISMIC_DATASET
  if role == "velocity_model":
    return ACTIVE_VELOCITY_MODEL
  active = manifest.get("active", {})
  if role not in active:
    raise KeyError(f"Role '{role}' not in manifest.active. Available: {list(active)}")
  return active[role]


def load_array(
  role: str,
  processed_dir: Optional[Path] = None,
  dataset_key: Optional[str] = None,
) -> np.ndarray:
  """Загружает массив по роли (seismic | velocity | velocity_model)."""
  manifest = load_manifest(processed_dir)
  key = dataset_key or _resolve_dataset_key(manifest, role)
  info = manifest["datasets"][key]
  path = Path(info["array_path"])
  if not path.exists():
    raise FileNotFoundError(f"Array not found: {path}")
  data = np.load(path)
  logger.info("Loaded %s | role=%s | shape=%s", key, role, data.shape)
  return data.astype(np.float32)


def load_seismic_with_headers(
  processed_dir: Optional[Path] = None,
  dataset_key: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
  """Возвращает (traces[n_traces, n_samples], headers, meta)."""
  manifest = load_manifest(processed_dir)
  key = dataset_key or ACTIVE_SEISMIC_DATASET
  if key not in manifest["datasets"]:
    raise KeyError(f"Dataset '{key}' not in manifest. Available: {list(manifest['datasets'])}")

  info = manifest["datasets"][key]
  if info.get("kind") != "seismic":
    logger.warning("Dataset %s has kind=%s (expected seismic)", key, info.get("kind"))

  data = np.load(info["array_path"]).astype(np.float32)
  headers: Dict[str, np.ndarray] = {}
  if info.get("headers_path"):
    with np.load(info["headers_path"]) as hp:
      headers = {k: hp[k] for k in hp.files}
  else:
    logger.warning("No headers for %s — re-run convert_sgy_to_numpy.py", key)

  logger.info(
    "Loaded seismic '%s' | shape=%s | shots=%s | dt_ms=%s",
    key, data.shape,
    info.get("extra", {}).get("n_shots", "?"),
    info.get("sample_interval_ms", "?"),
  )
  return data, headers, info


def load_velocity_model(
  processed_dir: Optional[Path] = None,
  dataset_key: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
  """Загружает модель Vp [nz, nx] (трассы → X, отсчёты → Z)."""
  manifest = load_manifest(processed_dir)
  key = dataset_key or ACTIVE_VELOCITY_MODEL
  info = manifest["datasets"][key]
  raw = np.load(info["array_path"]).astype(np.float32)
  vp = raw.T
  logger.info("Loaded velocity model '%s' | shape_zx=%s", key, vp.shape)
  return vp, info
