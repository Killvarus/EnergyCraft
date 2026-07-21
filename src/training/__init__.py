"""
Циклы обучения с ранней остановкой по val_loss.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

from src.config import (
    LR, MOMENTUM, WEIGHT_DECAY, GRAD_CLIP,
    PATIENCE, N_EPOCH, MAX_EPOCHS, DEVICE,
    OPTIMIZER, LOSS_FN,
    SCHEDULER_TYPE, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA,
    LOG_INTERVAL, TENSORBOARD_LOG_BATCH_EVERY,
    MODELS_DIR,
)
from src.training.early_stopping import EarlyStopConfig, EarlyStopState, relative_improvement, update_early_stop
from src.training.tensorboard import TrainingTracker

logger = logging.getLogger("cnn_baseline")


def _build_optimizer(params, lr: float) -> optim.Optimizer:
    opt = OPTIMIZER.lower()
    if opt == "adam":
        return optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    elif opt == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    elif opt == "sgd":
        return optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    elif opt == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {OPTIMIZER}")


def _build_criterion() -> nn.Module:
    if LOSS_FN.lower() == "mse":
        return nn.MSELoss()
    elif LOSS_FN.lower() == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Неизвестная функция потерь: {LOSS_FN}")


def _build_scheduler(optimizer: optim.Optimizer, max_epochs: int) -> Optional[object]:
    st = SCHEDULER_TYPE.lower()
    if st == "none":
        return None
    elif st == "step":
        return StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    elif st == "cosine":
        t_max = max_epochs if max_epochs > 0 else 10_000
        return CosineAnnealingLR(optimizer, T_max=t_max)
    elif st == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_GAMMA,
                                 patience=max(N_EPOCH // 2, 1), min_lr=1e-8)
    else:
        raise ValueError(f"Неизвестный scheduler: {SCHEDULER_TYPE}")


@torch.no_grad()
def _eval_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LR,
    rel_patience: float = PATIENCE,
    n_epoch: int = N_EPOCH,
    device: torch.device = DEVICE,
    log_interval: int = LOG_INTERVAL,
    random_state: Optional[int] = None,
    tracker: Optional[TrainingTracker] = None,
    tb_run_name: Optional[str] = None,
) -> Tuple[List[float], Dict]:
    """
    Обучение CNN/MLP с ранней остановкой по val_loss.

    max_epochs=0 — без верхнего лимита эпох (только early stopping).
    rel_patience (PATIENCE) — мин. относительное улучшение val_loss.
    n_epoch (N_EPOCH) — сколько эпох подряд без такого улучшения до остановки.
    """
    if val_loader is None:
        raise ValueError("val_loader обязателен: обучение останавливается по val_loss")

    owns_tracker = tracker is None
    if tracker is None:
        run_name = tb_run_name or TrainingTracker.make_run_name(model.__class__.__name__.lower())
        tracker = TrainingTracker(run_name)

    model.to(device)
    criterion = _build_criterion()
    optimizer = _build_optimizer(model.parameters(), lr=lr)
    scheduler = _build_scheduler(optimizer, max_epochs)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_state = copy.deepcopy(model.state_dict())
    es_cfg = EarlyStopConfig(patience=rel_patience, n_epoch=n_epoch, max_epochs=max_epochs)
    es = EarlyStopState()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Training start | arch=%s | params=%d | optimizer=%s | lr=%.5f | "
        "max_epochs=%s | train=%d | val=%d | rel_patience=%.2e | n_epoch=%d | device=%s",
        model.__class__.__name__, n_params, OPTIMIZER.upper(), lr,
        "unlimited" if max_epochs <= 0 else str(max_epochs),
        len(train_loader.dataset), len(val_loader.dataset),
        rel_patience, n_epoch, device,
    )

    t_start = time.time()
    epoch = 0
    global_step = 0

    try:
        while True:
            if max_epochs > 0 and epoch >= max_epochs:
                logger.info("Stop: reached MAX_EPOCHS=%d", max_epochs)
                break

            model.train()
            epoch_train_loss = 0.0
            n_train_batches = len(train_loader)
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader, start=1):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                epoch_train_loss += loss.item() * x_batch.size(0)
                global_step += 1

                if (
                    batch_idx == 1
                    or batch_idx % TENSORBOARD_LOG_BATCH_EVERY == 0
                    or batch_idx == n_train_batches
                ):
                    tracker.scalar("batch/loss", loss.item(), global_step)
                    tracker.scalar("time/elapsed_minutes", (time.time() - t_start) / 60.0, global_step)

                if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == n_train_batches:
                    logger.info(
                        "Epoch %d train | batch %d/%d | batch_loss=%.6e | elapsed=%.1f min",
                        epoch + 1, batch_idx, n_train_batches, loss.item(),
                        (time.time() - t_start) / 60.0,
                    )

            epoch_train_loss /= len(train_loader.dataset)
            logger.info("Epoch %d validation started | batches=%d", epoch + 1, len(val_loader))
            epoch_val_loss = _eval_loader(model, val_loader, criterion, device)

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()

            rel_imp = relative_improvement(es.best_loss, epoch_val_loss)
            es = update_early_stop(es, epoch_val_loss, epoch, es_cfg)
            if es.best_epoch == epoch:
                best_state = copy.deepcopy(model.state_dict())

            epoch_num = epoch + 1
            lr_now = optimizer.param_groups[0]["lr"]
            tracker.scalar("epoch/train_loss", epoch_train_loss, epoch_num)
            tracker.scalar("epoch/val_loss", epoch_val_loss, epoch_num)
            tracker.scalar("epoch/best_val_loss", es.best_loss, epoch_num)
            tracker.scalar("early_stopping/no_improve_epochs", es.bad_epochs, epoch_num)
            tracker.scalar("optimizer/learning_rate", lr_now, epoch_num)
            if rel_imp != float("inf") and rel_imp != float("-inf"):
                tracker.scalar("early_stopping/relative_improvement", rel_imp, epoch_num)

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                logger.info(
                    "Epoch %4d | train=%.6e | val=%.6e | best_val=%.6e (ep %d) | "
                    "rel_imp=%.2e | no_improve=%d/%d | lr=%.2e",
                    epoch + 1, epoch_train_loss, epoch_val_loss,
                    es.best_loss, es.best_epoch + 1, rel_imp, es.bad_epochs, n_epoch, lr_now,
                )

            if es.stopped:
                logger.info(
                    "Early stop at epoch %d | %s | best_epoch=%d | best_val=%.6e",
                    epoch + 1, es.stop_reason, es.best_epoch + 1, es.best_loss,
                )
                break

            epoch += 1
    finally:
        if owns_tracker:
            tracker.close()

    model.load_state_dict(best_state)
    elapsed = time.time() - t_start

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": es.best_epoch,
        "best_val_loss": es.best_loss,
        "stopped_epoch": len(train_losses),
        "early_stop_reason": es.stop_reason,
        "total_time": elapsed,
        "optimizer": OPTIMIZER,
        "lr": lr,
        "rel_patience": rel_patience,
        "n_epoch": n_epoch,
        "device": str(device),
    }

    logger.info(
        "Training done | epochs=%d | best_epoch=%d | best_val=%.6e | time=%.1fs",
        len(train_losses), es.best_epoch + 1, es.best_loss, elapsed,
    )
    return train_losses, history


def save_checkpoint(model: nn.Module, name: str, **extra_meta) -> str:
    save_path = MODELS_DIR / f"{name}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    checkpoint.update(extra_meta)
    torch.save(checkpoint, save_path)
    logger.info("Чекпоинт сохранён: %s", save_path)
    return str(save_path)
