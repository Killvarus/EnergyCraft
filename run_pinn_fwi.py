"""
PINN для обратной задачи FWI на тестовых окнах (все источники данных).

Запуск:
    python run_pinn_fwi.py
    python run_pinn_fwi.py --max-windows 2 --max-epochs 100
    python run_pinn_fwi.py --single
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

if sys.stdout.encoding != "utf-8":
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.config import (
  ACTIVE_DATASET, ACTIVE_SEISMIC_DATASET, DEVICE, HEIGHT, LOGS_DIR, OUTPUT_DIR,
  PINN_DX, PINN_DZ, PINN_MAX_EPOCHS, SPLIT_MODE, SPLIT_RANDOM_STATE, USE_ALL_DATASETS,
  WIDTH, apply_dataset_preset, setup_directories,
)
from src.data import minmax_denormalize, normalize_amplitude, prepare_training_windows
from src.data.combined import combined_split_indices
from src.data.mosaic import stitch_windows
from src.evaluation import MetricsCalculator
from src.models import FWI_PINN
from src.training.train_pinn_fwi import predict_velocity_grid, train_pinn_window
from src.training.tensorboard import TrainingTracker, tensorboard_root
from src.utils import setup_logger
from src.visualization import plot_learning_curve, plot_prediction_slice


def run_pinn_fwi(
  max_windows: int = 5,
  max_epochs: int | None = None,
  seismic_dataset: str | None = None,
  dataset: str | None = None,
  single: bool = False,
) -> None:
  setup_directories()
  logger = setup_logger("pinn_fwi", LOGS_DIR / "pinn_fwi.log")
  epochs_limit = PINN_MAX_EPOCHS if max_epochs is None else max_epochs

  use_all = USE_ALL_DATASETS and not single and not seismic_dataset
  family = dataset if dataset in ("zoloto", "domanic") else None
  if single:
    apply_dataset_preset(dataset or ACTIVE_DATASET)
    use_all = False

  logger.info("=" * 60)
  logger.info(
    "PINN FWI | use_all=%s | family=%s | max_windows=%d | max_epochs=%s",
    use_all, family, max_windows, "unlimited" if epochs_limit <= 0 else epochs_limit,
  )

  pack = prepare_training_windows(
    target_t=HEIGHT, window_w=WIDTH,
    use_all=use_all, family=family,
    seismic_dataset_key=seismic_dataset,
  )
  X_w, Y_w, meta = pack["X_w"], pack["Y_w"], pack["meta"]
  y_vmin, y_vmax = pack["y_vmin"], pack["y_vmax"]
  sources = pack.get("sources", {})

  _, _, te_idx = combined_split_indices(meta, mode=SPLIT_MODE, seed=SPLIT_RANDOM_STATE)
  test_windows = te_idx[:max_windows]
  logger.info("PINN test windows: %s (of %d)", list(test_windows), len(X_w))

  X_norm = normalize_amplitude(X_w)
  z_extent = HEIGHT * PINN_DZ
  per_window: list[dict] = []

  for i, wi in enumerate(test_windows):
    m = meta[wi]
    source_id = m["dataset_id"]
    x_start = float(m.get("x_start_m", m["x_start_idx"] * PINN_DX))
    x_extent = WIDTH * PINN_DX
    logger.info("PINN %d/%d | source=%s | x_start=%.1f", i + 1, len(test_windows), source_id, x_start)

    model = FWI_PINN(vp_min=y_vmin, vp_max=y_vmax)
    tb_name = TrainingTracker.make_run_name("pinn", f"{source_id}_w{i}")
    model, history = train_pinn_window(
      model, observed_window=X_norm[wi, 0],
      x_start=x_start, x_extent=x_extent, z_extent=z_extent,
      max_epochs=epochs_limit, vp_min=y_vmin, vp_max=y_vmax, device=DEVICE,
      tb_run_name=tb_name,
    )
    vp_pred = predict_velocity_grid(model, x_start=x_start, width=WIDTH, height=HEIGHT, device=DEVICE)
    y_true = Y_w[wi, 0]
    metrics = MetricsCalculator.calculate_all(y_true, vp_pred)
    per_window.append({"window": i, "source": source_id, **metrics, "best_loss": history["best_loss"]})
    logger.info("  R2=%.4f | RMSE=%.1f | loss=%.4e", metrics["r2"], metrics["rmse"], history["best_loss"])
    plot_prediction_slice(y_true, vp_pred, name=f"pinn_{source_id}_w{i}",
                          title=f"PINN | {source_id} | R2={metrics['r2']:.3f}")
    plot_learning_curve(history["losses"], None, len(history["losses"]) - 1,
                        name=f"pinn_loss_{source_id}_w{i}", title=f"PINN loss {source_id}")

  per_source_stitched: dict = {}
  by_source: dict = defaultdict(list)
  for i, wi in enumerate(test_windows):
    by_source[meta[wi]["dataset_id"]].append(i)

  for source_id, local_idxs in by_source.items():
    if source_id not in sources:
      continue
    src_pack = sources[source_id]
    preds = [per_window[li] for li in local_idxs]
    _ = preds
    wi_list = [test_windows[li] for li in local_idxs]
    # Re-predict not stored - skip stitched for PINN multi-window per source for now
    logger.info("PINN source '%s': %d windows evaluated", source_id, len(local_idxs))

  summary = {
    "pipeline": "pinn_fwi_combined" if pack.get("combined") else "pinn_fwi_single",
    "equation": "(1/v^2)*u_tt - u_xx - u_zz = f",
    "use_all_datasets": use_all,
    "pairs": [p["id"] for p in pack.get("pairs", [])],
    "n_windows_per_source": pack.get("n_windows_per_source", {}),
    "n_test_windows": len(test_windows),
    "epochs_per_window": epochs_limit if epochs_limit > 0 else "unlimited",
    "per_window": per_window,
  }
  tag = "combined" if pack.get("combined") else SPLIT_MODE
  out = OUTPUT_DIR / f"summary_pinn_fwi_{tag}.json"
  with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

  logger.info("PINN FWI done | windows=%d | tensorboard=%s", len(test_windows), tensorboard_root())
  print(f"Summary: {out}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max-windows", type=int, default=5)
  parser.add_argument("--max-epochs", type=int, default=None,
                      help="Верхний лимит эпох на окно; 0 = только early stopping")
  parser.add_argument("--epochs", type=int, default=None,
                      help="(устар.) то же, что --max-epochs")
  parser.add_argument("--seismic", type=str, default=None, help="Одна конкретная сейсмика")
  parser.add_argument("--dataset", choices=["zoloto", "domanic"], default=None)
  parser.add_argument("--single", action="store_true")
  args = parser.parse_args()
  run_pinn_fwi(
    max_windows=args.max_windows,
    max_epochs=args.max_epochs if args.max_epochs is not None else args.epochs,
    seismic_dataset=args.seismic, dataset=args.dataset, single=args.single,
  )
