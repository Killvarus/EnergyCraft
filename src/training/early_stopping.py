"""
Ранняя остановка: относительный порог PATIENCE за N_EPOCH эпох без улучшения.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def relative_improvement(best: float, current: float) -> float:
    """Относительное улучшение loss: (best - current) / |best|."""
    if not np.isfinite(best):
        return float("inf")
    denom = max(abs(best), 1e-12)
    return (best - current) / denom


@dataclass
class EarlyStopState:
    best_loss: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0
    stopped: bool = False
    stop_reason: str = ""


@dataclass
class EarlyStopConfig:
    """PATIENCE — мин. относительное улучшение; N_EPOCH — эпох без него до стопа."""

    patience: float = 0.003
    n_epoch: int = 15
    max_epochs: int = 0


def update_early_stop(
    state: EarlyStopState,
    current_loss: float,
    epoch: int,
    cfg: EarlyStopConfig,
) -> EarlyStopState:
    """
    Улучшение засчитывается, если относительное снижение loss >= cfg.patience.
    """
    rel = relative_improvement(state.best_loss, current_loss)

    if rel >= cfg.patience:
        state.best_loss = current_loss
        state.best_epoch = epoch
        state.bad_epochs = 0
    else:
        state.bad_epochs += 1
        if current_loss < state.best_loss:
            state.best_loss = current_loss
            state.best_epoch = epoch

    if state.bad_epochs >= cfg.n_epoch:
        state.stopped = True
        state.stop_reason = (
            f"no relative improvement >= {cfg.patience:.2e} for {cfg.n_epoch} epochs"
        )

    return state
