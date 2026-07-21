"""
Построение графиков для FWI.

Графики:
- Кривая обучения (train/val loss + best epoch)
- Сравнительная heatmap моделей
- Срез скоростной модели (истина / предсказание / остаток)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import PLOTS_DIR


# ---------------------------------------------------------------------------
# Стиль
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def plot_learning_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    best_epoch: int,
    name: str = "model",
    title: str = "Кривая обучения",
) -> str:
    """
    Кривая обучения с вертикальной best-epoch-линией.

    Параметры
    ----------
    train_losses : train loss по эпохам
    val_losses   : val loss по эпохам (опционально)
    best_epoch   : индекс лучшей эпохи (0-based)
    name         : имя для имени файла
    title        : заголовок графика

    Возвращает
    ----------
    Путь к сохранённому PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    ax.axvline(x=best_epoch + 1, color="r", linestyle="--", label=f"Best epoch: {best_epoch + 1}")

    ax.set_xlabel("Эпохи")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = PLOTS_DIR / f"{name}_learning_curve.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    return str(save_path)


def plot_comparison_heatmap(
    metrics: Dict[str, Dict[str, float]],
    name: str = "comparison",
    title: str = "Сравнение моделей",
) -> str:
    """
    Seaborn heatmap: строки — модели, колонки — метрики.

    Параметры
    ----------
    metrics : словарь вида {"MLP": {"r2": ..., "mse": ...}, "PINN": {...}}
    name    : имя для имени файла
    title   : заголовок графика

    Возвращает
    ----------
    Путь к сохранённому PNG.
    """
    import pandas as pd

    df = pd.DataFrame(metrics).T

    fig, ax = plt.subplots(figsize=(10, max(2, len(metrics) * 1.2)))
    sns.heatmap(
        df, annot=True, fmt=".4e", cmap="RdYlGn",
        ax=ax, cbar_kws={"label": "Значение"}, linewidths=0.5,
    )
    ax.set_title(title)
    plt.tight_layout()

    save_path = PLOTS_DIR / f"{name}_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    return str(save_path)


def plot_prediction_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "model",
    title: str = "Скоростная модель",
) -> str:
    """
    Три панели: истина / предсказание / остаток.

    Возвращает
    ----------
    Путь к сохранённому PNG.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())

    im0 = axes[0].imshow(y_true, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Истинная скорость Vp")
    axes[0].set_xlabel("X (трассы)")
    axes[0].set_ylabel("Z (время)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(y_pred, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Предсказанная скорость Vp")
    axes[1].set_xlabel("X (трассы)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    residual = y_true - y_pred
    vlim = np.abs(residual).max()
    im2 = axes[2].imshow(residual, aspect="auto", cmap="seismic", vmin=-vlim, vmax=vlim)
    axes[2].set_title("Остаток (True − Pred)")
    axes[2].set_xlabel("X (трассы)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(title)
    plt.tight_layout()

    save_path = PLOTS_DIR / f"{name}_prediction.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    return str(save_path)
