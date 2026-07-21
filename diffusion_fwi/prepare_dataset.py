"""
Подготовка датасета в формате, совместимом с NVIDIA PhysicsNeMo diffusion_fwi.

PhysicsNeMo ожидает E-FWI .npz файлы с полями velocity и seismic observations.
Этот скрипт создаёт упрощённый аналог из окон EnergyCraft.

Запуск:
    python diffusion_fwi/prepare_dataset.py
    python diffusion_fwi/prepare_dataset.py --output diffusion_fwi/data/energycraft_windows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
  sys.path.insert(0, str(ROOT.parent))

from src.config import HEIGHT, WIDTH, setup_directories
from src.data import minmax_normalize, normalize_amplitude, prepare_training_windows


def prepare_efwi_style_dataset(output_dir: Path) -> Path:
  setup_directories()
  output_dir = Path(output_dir)
  samples_dir = output_dir / "samples"
  samples_dir.mkdir(parents=True, exist_ok=True)

  pack = prepare_training_windows(target_t=HEIGHT, window_w=WIDTH)
  X_w, Y_w = pack["X_w"], pack["Y_w"]
  y_vmin, y_vmax = pack["y_vmin"], pack["y_vmax"]

  X_norm = normalize_amplitude(X_w)
  Y_norm = minmax_normalize(Y_w, vmin=y_vmin, vmax=y_vmax)

  for i in range(len(X_w)):
    np.savez_compressed(
      samples_dir / f"sample_{i:05d}.npz",
      seismic=X_norm[i, 0].astype(np.float32),
      vp=Y_norm[i, 0].astype(np.float32),
      vp_physical=Y_w[i, 0].astype(np.float32),
    )

  stats = {
    "n_samples": int(len(X_w)),
    "seismic_shape": list(X_w.shape[2:]),
    "vp_shape": list(Y_w.shape[2:]),
    "vp_min": y_vmin,
    "vp_max": y_vmax,
    "format": "energycraft_window_v1",
    "physicsnemo_compatible_fields": ["seismic", "vp"],
  }
  with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

  print(f"Prepared {len(X_w)} samples in {samples_dir}")
  return output_dir


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, default=ROOT / "data" / "energycraft_windows")
  args = parser.parse_args()
  prepare_efwi_style_dataset(args.output)
