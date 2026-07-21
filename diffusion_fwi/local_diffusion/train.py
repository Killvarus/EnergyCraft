"""
Обучение локальной diffusion model на окнах EnergyCraft.

Запуск:
    python diffusion_fwi/local_diffusion/train.py
    python diffusion_fwi/local_diffusion/train.py --max-epochs 100 --batch-size 4
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from src.config import (
  DEVICE, DIFFUSION_BASE_CHANNELS, DIFFUSION_BATCH_SIZE, DIFFUSION_LR,
  DIFFUSION_MAX_EPOCHS, DIFFUSION_TIMESTEPS, HEIGHT, LOGS_DIR, LOG_INTERVAL,
  N_EPOCH, OUTPUT_DIR, PATIENCE, SPLIT_MODE, SPLIT_RANDOM_STATE, WIDTH,
  setup_directories,
)
from src.data import minmax_normalize, normalize_amplitude, prepare_training_windows
from src.data.combined import combined_split_indices
from src.training.early_stopping import EarlyStopConfig, EarlyStopState, relative_improvement, update_early_stop
from src.training.tensorboard import TrainingTracker, tensorboard_root
from src.utils import setup_logger
from diffusion_fwi.local_diffusion.model import DiffusionSchedule, DiffusionUNet


@torch.no_grad()
def _eval_diffusion(
  model: DiffusionUNet,
  schedule: DiffusionSchedule,
  loader: DataLoader,
  loss_fn: torch.nn.Module,
) -> float:
  model.eval()
  total = 0.0
  n = 0
  for x_cond, y0 in loader:
    x_cond, y0 = x_cond.to(DEVICE), y0.to(DEVICE)
    t = torch.randint(0, schedule.timesteps, (y0.size(0),), device=DEVICE)
    noise = torch.randn_like(y0)
    y_noisy = schedule.q_sample(y0, t, noise)
    pred = model(y_noisy, x_cond, t)
    loss = loss_fn(pred, noise)
    bs = y0.size(0)
    total += loss.item() * bs
    n += bs
  return total / max(n, 1)


def train_local_diffusion(
  max_epochs: int | None = None,
  batch_size: int | None = None,
  lr: float | None = None,
) -> None:
  setup_directories()
  logger = setup_logger("diffusion_local", LOGS_DIR / "diffusion_local.log")
  epoch_limit = DIFFUSION_MAX_EPOCHS if max_epochs is None else max_epochs
  batch_size = batch_size or DIFFUSION_BATCH_SIZE
  lr = lr or DIFFUSION_LR

  logger.info(
    "Local diffusion train | max_epochs=%s | batch=%d | lr=%.2e | rel_patience=%.2e | n_epoch=%d",
    "unlimited" if epoch_limit <= 0 else epoch_limit,
    batch_size, lr, PATIENCE, N_EPOCH,
  )
  pack = prepare_training_windows(target_t=HEIGHT, window_w=WIDTH)
  X_w, Y_w = pack["X_w"], pack["Y_w"]
  y_vmin, y_vmax = pack["y_vmin"], pack["y_vmax"]
  meta = pack["meta"]

  tr_idx, va_idx, te_idx = combined_split_indices(meta, mode=SPLIT_MODE, seed=SPLIT_RANDOM_STATE)
  X_norm = normalize_amplitude(X_w)
  Y_norm = minmax_normalize(Y_w, vmin=y_vmin, vmax=y_vmax)

  train_loader = DataLoader(
    TensorDataset(
      torch.from_numpy(X_norm[tr_idx]).float(),
      torch.from_numpy(Y_norm[tr_idx]).float(),
    ),
    batch_size=batch_size, shuffle=True,
  )
  val_loader = DataLoader(
    TensorDataset(
      torch.from_numpy(X_norm[va_idx]).float(),
      torch.from_numpy(Y_norm[va_idx]).float(),
    ),
    batch_size=batch_size, shuffle=False,
  )

  model = DiffusionUNet(base_ch=DIFFUSION_BASE_CHANNELS).to(DEVICE)
  schedule = DiffusionSchedule(timesteps=DIFFUSION_TIMESTEPS).to(DEVICE)
  n_params = sum(p.numel() for p in model.parameters())
  logger.info(
    "Model params=%d | timesteps=%d | train=%d | val=%d",
    n_params, DIFFUSION_TIMESTEPS, len(tr_idx), len(va_idx),
  )
  opt = torch.optim.AdamW(model.parameters(), lr=lr)
  loss_fn = torch.nn.MSELoss()

  best_state = copy.deepcopy(model.state_dict())
  es_cfg = EarlyStopConfig(patience=PATIENCE, n_epoch=N_EPOCH, max_epochs=epoch_limit)
  es = EarlyStopState()
  epoch = 0
  tracker = TrainingTracker(TrainingTracker.make_run_name("diffusion_local"))

  try:
    while True:
      if epoch_limit > 0 and epoch >= epoch_limit:
        logger.info("Diffusion stop: reached max_epochs=%d", epoch_limit)
        break

      model.train()
      train_total = 0.0
      train_n = 0
      for x_cond, y0 in train_loader:
        x_cond, y0 = x_cond.to(DEVICE), y0.to(DEVICE)
        t = torch.randint(0, schedule.timesteps, (y0.size(0),), device=DEVICE)
        noise = torch.randn_like(y0)
        y_noisy = schedule.q_sample(y0, t, noise)
        pred = model(y_noisy, x_cond, t)
        loss = loss_fn(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_total += loss.item() * y0.size(0)
        train_n += y0.size(0)

      train_loss = train_total / max(train_n, 1)
      val_loss = _eval_diffusion(model, schedule, val_loader, loss_fn)
      rel_imp = relative_improvement(es.best_loss, val_loss)
      es = update_early_stop(es, val_loss, epoch, es_cfg)
      if es.best_epoch == epoch:
        best_state = copy.deepcopy(model.state_dict())

      epoch_num = epoch + 1
      tracker.scalar("epoch/train_loss", train_loss, epoch_num)
      tracker.scalar("epoch/val_loss", val_loss, epoch_num)
      tracker.scalar("epoch/best_val_loss", es.best_loss, epoch_num)
      if rel_imp != float("inf") and rel_imp != float("-inf"):
        tracker.scalar("early_stopping/relative_improvement", rel_imp, epoch_num)
      tracker.scalar("early_stopping/no_improve_epochs", es.bad_epochs, epoch_num)

      if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
        logger.info(
          "Epoch %4d | train=%.6f | val=%.6f | best_val=%.6f (ep %d) | "
          "rel_imp=%.2e | no_improve=%d/%d",
          epoch + 1, train_loss, val_loss, es.best_loss, es.best_epoch + 1,
          rel_imp, es.bad_epochs, N_EPOCH,
        )

      if es.stopped:
        logger.info(
          "Diffusion early stop at epoch %d | %s | best_val=%.6f",
          epoch + 1, es.stop_reason, es.best_loss,
        )
        break

      epoch += 1
  finally:
    tracker.close()

  model.load_state_dict(best_state)
  logger.info("TensorBoard: tensorboard --logdir %s", tensorboard_root())

  ckpt_dir = OUTPUT_DIR / "models"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  ckpt_path = ckpt_dir / "local_diffusion_fwi.pt"
  torch.save({
    "model": model.state_dict(),
    "y_vmin": y_vmin,
    "y_vmax": y_vmax,
    "best_val_loss": es.best_loss,
    "best_epoch": es.best_epoch + 1,
  }, ckpt_path)

  model.eval()
  with torch.no_grad():
    x_test = torch.from_numpy(X_norm[te_idx[:3]]).float().to(DEVICE)
    samples = schedule.sample(model, x_test).cpu().numpy()

  summary = {
    "checkpoint": str(ckpt_path),
    "epochs_trained": epoch + 1 if es.stopped else epoch,
    "best_val_loss": float(es.best_loss),
    "best_epoch": es.best_epoch + 1,
    "n_train": int(len(tr_idx)),
    "n_val": int(len(va_idx)),
    "n_test_samples": int(len(samples)),
    "note": "Local diffusion baseline inspired by NVIDIA PhysicsNeMo diffusion_fwi",
  }
  out = OUTPUT_DIR / "summary_local_diffusion.json"
  with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  logger.info("Saved checkpoint: %s", ckpt_path)
  logger.info("Summary: %s", out)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--max-epochs", type=int, default=None,
                 help="Верхний лимит эпох; 0 = только early stopping")
  p.add_argument("--epochs", type=int, default=None, help="(устар.) то же, что --max-epochs")
  p.add_argument("--batch-size", type=int, default=4)
  p.add_argument("--lr", type=float, default=1e-4)
  args = p.parse_args()
  max_ep = args.max_epochs if args.max_epochs is not None else args.epochs
  train_local_diffusion(max_epochs=max_ep, batch_size=args.batch_size, lr=args.lr)
