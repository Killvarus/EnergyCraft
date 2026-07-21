"""Встроенное логирование TensorBoard для циклов обучения."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.config import OUTPUT_DIR, TENSORBOARD_DIR, TENSORBOARD_ENABLED


class TrainingTracker:
    """Обёртка над SummaryWriter; безопасно отключается через config."""

    def __init__(
        self,
        run_name: str,
        enabled: Optional[bool] = None,
        log_dir=None,
    ) -> None:
        self.enabled = TENSORBOARD_ENABLED if enabled is None else enabled
        self.writer = None
        if not self.enabled:
            return

        from torch.utils.tensorboard import SummaryWriter

        base = log_dir or TENSORBOARD_DIR
        base.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(base / run_name))

    @staticmethod
    def make_run_name(prefix: str, suffix: str = "") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tail = f"_{suffix}" if suffix else ""
        return f"{prefix}_{stamp}{tail}"

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


def tensorboard_root() -> str:
    """Путь к корню логов для команды tensorboard --logdir."""
    return str(TENSORBOARD_DIR or (OUTPUT_DIR / "tensorboard"))
