"""
Конвертация SEG-Y → NumPy (obspy для данных, segyio для заголовков).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import obspy

logger = logging.getLogger("fwi.data.sgy_converter")

try:
    import segyio
    from segyio import TraceField
except ImportError:  # pragma: no cover
    segyio = None
    TraceField = None


@dataclass
class SegyDatasetInfo:
  """Метаданные одного сконвертированного SEG-Y файла."""

  source_file: str
  array_path: str
  shape: List[int]
  dtype: str
  n_traces: int
  n_samples: int
  sample_interval_ms: float
  kind: str  # "seismic" | "velocity" | "velocity_model"
  headers_path: Optional[str] = None
  extra: Optional[Dict[str, Any]] = None


def _read_trace_data(sgy_path: Path) -> tuple[np.ndarray, float]:
  """Читает все трассы в массив [n_traces, n_samples] через obspy."""
  stream = obspy.read(str(sgy_path), format="SEGY")
  n_traces = len(stream)
  n_samples = len(stream[0].data)
  data = np.zeros((n_traces, n_samples), dtype=np.float32)
  for i, trace in enumerate(stream):
    data[i, :] = trace.data
  dt_ms = float(stream[0].stats.delta * 1000.0)
  return data, dt_ms


def _read_segy_headers(sgy_path: Path) -> Dict[str, np.ndarray]:
  """Извлекает ключевые поля заголовков трасс через segyio."""
  if segyio is None:
    raise ImportError("segyio required for header extraction: pip install segyio")

  with segyio.open(str(sgy_path), "r", ignore_geometry=True) as f:
    n = f.tracecount
    shot_id = np.array([f.header[i][TraceField.FieldRecord] for i in range(n)], dtype=np.int64)
    source_x = np.array([f.header[i][TraceField.SourceX] for i in range(n)], dtype=np.int64)
    group_x = np.array([f.header[i][TraceField.GroupX] for i in range(n)], dtype=np.int64)
    scalco = np.array([f.header[i][TraceField.SourceGroupScalar] for i in range(n)], dtype=np.int64)

  return {
    "shot_id": shot_id,
    "source_x": source_x,
    "group_x": group_x,
    "scalco": scalco,
  }


def _apply_scalco(raw: np.ndarray, scalco: np.ndarray) -> np.ndarray:
  raw = raw.astype(np.float64)
  scalco = scalco.astype(np.float64)
  out = raw.copy()
  pos = scalco > 0
  neg = scalco < 0
  out[pos] = raw[pos] * scalco[pos]
  out[neg] = raw[neg] / np.abs(scalco[neg])
  return out


def _infer_kind(filename: str) -> str:
  """Определяет тип SEG-Y. Явные переопределения — для известных датасетов."""
  name = filename.lower()
  overrides = {
    # Zoloto
    "vp_zoloto_50sm-50sm.sgy": "seismic",
    "vp_zoloto_101shot_501rec_1000ms.sgy": "seismic",
    "vp_zoloto_50sm-50sm_model.sgy": "velocity_model",
    # Domanic
    "domanic_vp_2-2.sgy": "seismic",
    "domanic_vp_2-2_model.sgy": "velocity_model",
  }
  if name in overrides:
    return overrides[name]
  if "shot" in name or "seismic" in name or "record" in name:
    return "seismic"
  if name.endswith("_model.sgy") or "_model." in name:
    return "velocity_model"
  return "seismic"


def convert_sgy_file(
  sgy_path: Path,
  output_dir: Path,
  kind: Optional[str] = None,
) -> SegyDatasetInfo:
  """
  Конвертирует один SEG-Y файл в NumPy + JSON-метаданные.

  Для сейсмики дополнительно сохраняет заголовки (shot_id, координаты).
  """
  sgy_path = Path(sgy_path)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  stem = sgy_path.stem
  kind = kind or _infer_kind(sgy_path.name)

  data, dt_ms = _read_trace_data(sgy_path)
  array_path = output_dir / f"{stem}.npy"
  np.save(array_path, data)

  headers_path: Optional[str] = None
  extra: Dict[str, Any] = {"sample_interval_ms": dt_ms}

  if kind == "seismic":
    headers = _read_segy_headers(sgy_path)
    headers["group_x_m"] = _apply_scalco(headers["group_x"], headers["scalco"]).astype(np.float64)
    headers["source_x_m"] = _apply_scalco(headers["source_x"], headers["scalco"]).astype(np.float64)

    hp = output_dir / f"{stem}_headers.npz"
    np.savez_compressed(hp, **headers)
    headers_path = str(hp)

    unique_shots, counts = np.unique(headers["shot_id"], return_counts=True)
    extra.update({
      "n_shots": int(len(unique_shots)),
      "shots_min": int(unique_shots.min()),
      "shots_max": int(unique_shots.max()),
      "traces_per_shot_min": int(counts.min()),
      "traces_per_shot_max": int(counts.max()),
    })

  info = SegyDatasetInfo(
    source_file=str(sgy_path.resolve()),
    array_path=str(array_path.resolve()),
    shape=list(data.shape),
    dtype=str(data.dtype),
    n_traces=int(data.shape[0]),
    n_samples=int(data.shape[1]),
    sample_interval_ms=dt_ms,
    kind=kind,
    headers_path=headers_path,
    extra=extra,
  )

  meta_path = output_dir / f"{stem}_meta.json"
  with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(asdict(info), f, indent=2, ensure_ascii=False)

  logger.info("Converted %s -> %s | shape=%s | kind=%s", sgy_path.name, array_path.name, data.shape, kind)
  return info


def convert_all_sgy_in_directory(
  data_dir: Path,
  processed_dir: Path,
  pattern: str = "*.sgy",
) -> Dict[str, Any]:
  """Конвертирует все SEG-Y из data_dir в processed_dir."""
  from src.config import ACTIVE_DATASET, DATASET_PRESETS, apply_dataset_preset

  apply_dataset_preset(ACTIVE_DATASET)

  data_dir = Path(data_dir)
  processed_dir = Path(processed_dir)
  processed_dir.mkdir(parents=True, exist_ok=True)

  datasets: Dict[str, Any] = {}
  for sgy_path in sorted(data_dir.glob(pattern)):
    info = convert_sgy_file(sgy_path, processed_dir)
    datasets[sgy_path.stem] = asdict(info)

  manifest = {
    "data_dir": str(data_dir.resolve()),
    "processed_dir": str(processed_dir.resolve()),
    "datasets": datasets,
    "dataset_families": {
      "zoloto": {
        "velocity_model": "Vp_zoloto_50sm-50sm_MODEL",
        "seismic_variants": [
          "Vp_zoloto_101shot_501rec_1000ms",
          "Vp_zoloto_50sm-50sm",
        ],
        "note": "Две сейсмики к одной модели Vp, разная синтетическая генерация.",
      },
      "domanic": {
        "velocity_model": "Domanic_Vp_2-2_MODEL",
        "seismic": "Domanic_Vp_2-2",
        "note": "Domanic_Vp_2-2 — сейсмограммы; Domanic_Vp_2-2_MODEL — скоростная модель.",
      },
    },
    "active": {
      "dataset_preset": ACTIVE_DATASET,
      "seismic": DATASET_PRESETS[ACTIVE_DATASET]["seismic"],
      "seismic_alt": DATASET_PRESETS[ACTIVE_DATASET].get("seismic_alt"),
      "velocity_model": DATASET_PRESETS[ACTIVE_DATASET]["velocity_model"],
    },
    "presets": DATASET_PRESETS,
  }

  manifest_path = processed_dir / "manifest.json"
  with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

  logger.info("Manifest saved: %s (%d datasets)", manifest_path, len(datasets))
  return manifest
