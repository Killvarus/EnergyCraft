"""
CNN baseline (encoder-decoder) на окнах мозаики.

По умолчанию обучается на ВСЕХ парах сейсмика+Vp (USE_ALL_DATASETS=True).

Запуск:
    python run_cnn_baseline.py
    python run_cnn_baseline.py --dataset zoloto    # только Zoloto (2 сейсмики)
    python run_cnn_baseline.py --dataset domanic
    python run_cnn_baseline.py --single            # одна пара из ACTIVE_DATASET
"""
from __future__ import annotations

import copy
import json
import random
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if sys.stdout.encoding != "utf-8":
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.config import (
  ACTIVE_DATASET, BASELINE_ARCH, CNN_BASE_CHANNELS, DEVICE, HEIGHT, LOGS_DIR,
  MAX_EPOCHS, MLP_BATCH_SIZE, MLP_N_ITERATIONS, MLP_RANDOM_STATE, MLP_RANDOM_STATE_STEP,
  MLP_SELECTION_METRIC, OUTPUT_DIR, SPLIT_MODE, SPLIT_RANDOM_STATE, USE_ALL_DATASETS,
  WIDTH, apply_dataset_preset, setup_directories,
)
from src.data import (
  minmax_denormalize, minmax_normalize, normalize_amplitude, prepare_training_windows,
)
from src.data.combined import combined_split_indices
from src.data.mosaic import stitch_windows
from src.evaluation import MetricsCalculator
from src.models import FWICNN, FWIMLP
from src.training import save_checkpoint, train_mlp
from src.training.tensorboard import TrainingTracker, tensorboard_root
from src.utils import setup_logger
from src.visualization import plot_learning_curve, plot_prediction_slice

WINDOW_STRIDE = WIDTH // 4


def _set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _build_model() -> torch.nn.Module:
  if BASELINE_ARCH == "cnn":
    return FWICNN(base_channels=CNN_BASE_CHANNELS)
  return FWIMLP(height=HEIGHT, width=WIDTH)


def run_cnn_baseline(
  dataset: str | None = None,
  single: bool = False,
) -> None:
  setup_directories()
  logger = setup_logger("cnn_baseline", LOGS_DIR / "cnn_baseline.log")

  use_all = USE_ALL_DATASETS and not single
  family = dataset if dataset in ("zoloto", "domanic") else None
  if single or (dataset and dataset not in ("zoloto", "domanic")):
    apply_dataset_preset(dataset or ACTIVE_DATASET)
    use_all = False

  logger.info("=" * 60)
  logger.info(
    "CNN BASELINE | use_all=%s | family=%s | arch=%s | window=%dx%d",
    use_all, family, BASELINE_ARCH, HEIGHT, WIDTH,
  )

  pack = prepare_training_windows(
    target_t=HEIGHT,
    window_w=WIDTH,
    window_stride=WINDOW_STRIDE,
    use_all=use_all,
    family=family,
  )
  X_w, Y_w, meta = pack["X_w"], pack["Y_w"], pack["meta"]
  y_vmin, y_vmax = pack["y_vmin"], pack["y_vmax"]
  sources = pack.get("sources", {})

  logger.info(
    "Training pool | windows=%d | sources=%s | Vp [%.1f, %.1f] m/s",
    len(X_w), pack.get("n_windows_per_source", {}), y_vmin, y_vmax,
  )

  tr_idx, va_idx, te_idx = combined_split_indices(meta, mode=SPLIT_MODE, seed=SPLIT_RANDOM_STATE)
  logger.info(
    "Split (%s) | train=%d | val=%d | test=%d",
    SPLIT_MODE, len(tr_idx), len(va_idx), len(te_idx),
  )

  X_norm = normalize_amplitude(X_w)
  Y_norm = minmax_normalize(Y_w, vmin=y_vmin, vmax=y_vmax)

  def make_loader(x, y, shuffle):
    return DataLoader(
      TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float()),
      batch_size=MLP_BATCH_SIZE, shuffle=shuffle,
    )

  train_loader = make_loader(X_norm[tr_idx], Y_norm[tr_idx], True)
  val_loader = make_loader(X_norm[va_idx], Y_norm[va_idx], False)

  best_model_state = None
  best_metrics: Dict = {}
  best_history: Dict = {}
  best_iter = -1
  best_score = None
  iteration_results: List[Dict] = []

  for iter_idx in range(1, MLP_N_ITERATIONS + 1):
    seed = MLP_RANDOM_STATE + (iter_idx - 1) * MLP_RANDOM_STATE_STEP
    _set_seed(seed)
    model = _build_model()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Iteration %d/%d | seed=%d | params=%d", iter_idx, MLP_N_ITERATIONS, seed, n_params)
    tb_name = TrainingTracker.make_run_name("cnn", f"iter{iter_idx}_seed{seed}")
    _, history = train_mlp(
      model, train_loader, val_loader=val_loader,
      max_epochs=MAX_EPOCHS, random_state=seed, tb_run_name=tb_name,
    )
    val_score = float(history["best_val_loss"])

    model.eval()
    with torch.no_grad():
      pred = model(torch.from_numpy(X_norm[te_idx]).float().to(DEVICE)).cpu().numpy()

    y_true = minmax_denormalize(Y_norm[te_idx, 0], vmin=y_vmin, vmax=y_vmax)
    y_pred = minmax_denormalize(pred[:, 0], vmin=y_vmin, vmax=y_vmax)
    metrics = MetricsCalculator.calculate_all(y_true, y_pred)

    iteration_results.append({
      "iteration": iter_idx, "seed": seed,
      "best_val_loss": val_score, "best_epoch": history["best_epoch"] + 1,
      **{k: float(v) for k, v in metrics.items()},
    })
    logger.info(
      "Iteration %d | best_val=%.6e (ep %d) | test R2=%.4f | RMSE=%.1f",
      iter_idx, val_score, history["best_epoch"] + 1,
      metrics.get("r2", 0), metrics.get("rmse", 0),
    )

    if best_score is None or val_score < best_score:
      best_score = val_score
      best_iter = iter_idx
      best_model_state = copy.deepcopy(model.state_dict())
      best_history = history
      best_metrics = metrics

  assert best_model_state is not None
  best_model = _build_model()
  best_model.load_state_dict(best_model_state)
  save_checkpoint(best_model, "cnn_baseline_best", metrics=best_metrics, iteration=best_iter)

  plot_learning_curve(
    best_history["train_losses"], best_history.get("val_losses"),
    best_history["best_epoch"], name="cnn_baseline",
  )

  best_model.eval()
  with torch.no_grad():
    test_pred = best_model(torch.from_numpy(X_norm[te_idx]).float().to(DEVICE)).cpu().numpy()
  test_phys = minmax_denormalize(test_pred[:, 0], vmin=y_vmin, vmax=y_vmax)

  per_source_metrics: Dict[str, dict] = {}
  by_source: Dict[str, List[int]] = defaultdict(list)
  for k, wi in enumerate(te_idx):
    by_source[meta[wi]["dataset_id"]].append(k)

  for source_id, local_keys in by_source.items():
    y_t = np.stack([minmax_denormalize(Y_norm[te_idx[ki], 0], vmin=y_vmin, vmax=y_vmax) for ki in local_keys])
    y_p = np.stack([test_phys[ki] for ki in local_keys])
    sm = MetricsCalculator.calculate_all(y_t, y_p)
    per_source_metrics[source_id] = sm
    logger.info(
      "Test source '%s' | n=%d | R2=%.4f | RMSE=%.1f",
      source_id, len(local_keys), sm["r2"], sm["rmse"],
    )

    for ki in local_keys[:3]:
      wi = te_idx[ki]
      m = meta[wi]
      plot_prediction_slice(
        minmax_denormalize(Y_norm[wi, 0], vmin=y_vmin, vmax=y_vmax),
        test_phys[ki],
        name=f"cnn_test_{m['dataset_id']}_w{ki}",
        title=f"CNN test | {m['dataset_id']} | x={m.get('x_start_idx', '?')}",
      )

    if source_id in sources:
      src_pack = sources[source_id]
      src_te = [te_idx[ki] for ki in local_keys]
      src_meta = meta[src_te]
      src_pred = np.stack([test_phys[ki] for ki in local_keys])
      stitched_pred, stitched_mask = stitch_windows(src_pred, src_meta, src_pack["Y_full"].shape)
      stitched_true = np.where(stitched_mask, src_pack["Y_full"], np.nan)
      if np.any(stitched_mask):
        pm = MetricsCalculator.calculate_all(stitched_true[stitched_mask], stitched_pred[stitched_mask])
        per_source_metrics[f"{source_id}_stitched"] = pm
        plot_prediction_slice(
          np.nan_to_num(stitched_true, nan=y_vmin),
          np.nan_to_num(stitched_pred, nan=y_vmin),
          name=f"cnn_stitched_{source_id}_{SPLIT_MODE}",
          title=f"CNN stitched TEST | {source_id}",
        )

  summary = {
    "pipeline": "cnn_combined" if pack.get("combined") else "cnn_single",
    "model": BASELINE_ARCH,
    "use_all_datasets": use_all,
    "pairs": [p["id"] for p in pack.get("pairs", [])],
    "n_windows_per_source": pack.get("n_windows_per_source", {}),
    "split_mode": SPLIT_MODE,
    "n_windows": int(len(X_w)),
    "best_iteration": best_iter,
    "best_val_loss": best_score,
    "test_metrics_overall": best_metrics,
    "test_selection_metric": MLP_SELECTION_METRIC,
    "test_metrics_per_source": per_source_metrics,
    "iterations": iteration_results,
    "vp_norm_range": {"vmin": y_vmin, "vmax": y_vmax},
  }
  tag = "combined" if pack.get("combined") else SPLIT_MODE
  out = OUTPUT_DIR / f"summary_cnn_baseline_{tag}.json"
  with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

  logger.info(
    "CNN done | best_iter=%d | overall R2=%.4f | sources=%d | tensorboard=%s",
    best_iter, best_metrics.get("r2", 0), len(per_source_metrics), tensorboard_root(),
  )
  print(f"Summary: {out}")


if __name__ == "__main__":
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("--dataset", choices=["zoloto", "domanic"], default=None,
                 help="Только семейство zoloto (2 сейсмики) или domanic")
  p.add_argument("--single", action="store_true",
                 help="Только одна пара из ACTIVE_DATASET (без объединения)")
  args = p.parse_args()
  run_cnn_baseline(dataset=args.dataset, single=args.single)
