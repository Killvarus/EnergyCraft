"""
Анализ обучающих данных FWI.

Что делает скрипт:
1) Загружает X (сейсмограммы) и Y (скоростные модели)
2) Выводит статистику (форма, min/max/mean/std)
3) Для каждой сейсмограммы строит корреляции без агрегации признаков:
   - X vs X (между трассами)
   - X vs Y (между колонками X и Y)
4) Сохраняет heatmap и CSV по каждому sample

Запуск:
    python analyze_data.py
"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.config import (
    SEGY_PATH,
    VP_SEGY_PATH,
    OUTPUT_DIR,
    PLOTS_DIR,
    LOGS_DIR,
    NUM_SEISMOGRAMS,
    HEIGHT,
    WIDTH,
    setup_directories,
)
from src.utils import setup_logger
from src.data import (
    load_segy_to_tensor,
    load_velocity_from_segy,
    normalize_amplitude,
    minmax_normalize,
)


def _describe_array(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _save_heatmap(matrix: pd.DataFrame, title: str, filepath, fmt: str = ".2f") -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap="coolwarm", center=0, annot=False, fmt=fmt, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _corr_xx_xy_per_sample(x_2d: np.ndarray, y_2d: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Без агрегации признаков:
    - X vs X: корреляции между колонками (трассами) сейсмограммы
    - X vs Y: корреляции между колонками X и колонками Y

    x_2d/y_2d: [H, W]
    """
    x_cols = [f"x_{j:03d}" for j in range(x_2d.shape[1])]
    y_cols = [f"y_{j:03d}" for j in range(y_2d.shape[1])]

    df_x = pd.DataFrame(x_2d, columns=x_cols)
    df_y = pd.DataFrame(y_2d, columns=y_cols)

    corr_xx = df_x.corr(numeric_only=True)

    df_xy = pd.concat([df_x, df_y], axis=1)
    corr_all = df_xy.corr(numeric_only=True)
    corr_xy = corr_all.loc[x_cols, y_cols]

    # Если какие-то колонки константны, pandas даёт NaN — для heatmap заменяем на 0
    corr_xx = corr_xx.fillna(0.0)
    corr_xy = corr_xy.fillna(0.0)

    return corr_xx, corr_xy


def main() -> None:
    setup_directories()
    logger = setup_logger("fwi.data.analysis", LOGS_DIR / "data_analysis.log")

    logger.info("=" * 60)
    logger.info("DATA ANALYSIS START")
    logger.info("=" * 60)

    # 1) Загрузка
    X = load_segy_to_tensor(SEGY_PATH, num_seismic=NUM_SEISMOGRAMS)
    X_norm = normalize_amplitude(X)

    vp_model = load_velocity_from_segy(VP_SEGY_PATH, target_shape=(HEIGHT, WIDTH))
    y_vmin = float(np.min(vp_model))
    y_vmax = float(np.max(vp_model))

    Y = np.stack([vp_model] * NUM_SEISMOGRAMS, axis=0)
    Y = np.expand_dims(Y, axis=1)
    Y_norm = minmax_normalize(Y, vmin=y_vmin, vmax=y_vmax)

    # 2) Статистика массивов
    stats = {
        "X_raw": _describe_array(X),
        "X_norm": _describe_array(X_norm),
        "Y_raw": _describe_array(Y),
        "Y_norm": _describe_array(Y_norm),
    }

    logger.info("X_raw:  %s", stats["X_raw"])
    logger.info("X_norm: %s", stats["X_norm"])
    logger.info("Y_raw:  %s", stats["Y_raw"])
    logger.info("Y_norm: %s", stats["Y_norm"])

    # 3) Корреляции БЕЗ агрегации признаков: отдельно для каждой сейсмограммы
    sns.set_theme(style="whitegrid", context="notebook")

    sample_outputs = []
    for i in range(X_norm.shape[0]):
        x_i = X_norm[i, 0]  # [H, W]
        y_i = Y_norm[i, 0]  # [H, W]

        corr_xx_i, corr_xy_i = _corr_xx_xy_per_sample(x_i, y_i)

        corr_xx_path = OUTPUT_DIR / f"data_corr_xx_sample_{i:02d}.csv"
        corr_xy_path = OUTPUT_DIR / f"data_corr_xy_sample_{i:02d}.csv"
        corr_xx_i.to_csv(corr_xx_path, encoding="utf-8")
        corr_xy_i.to_csv(corr_xy_path, encoding="utf-8")

        p1 = PLOTS_DIR / f"data_corr_inputs_sample_{i:02d}.png"
        p2 = PLOTS_DIR / f"data_corr_input_target_sample_{i:02d}.png"
        _save_heatmap(corr_xx_i, f"Sample {i:02d}: Корреляция входных признаков (X vs X)", p1)
        _save_heatmap(corr_xy_i, f"Sample {i:02d}: Корреляция X vs Y", p2)

        sample_outputs.append({
            "sample_index": i,
            "corr_xx_csv": str(corr_xx_path),
            "corr_xy_csv": str(corr_xy_path),
            "corr_xx_plot": str(p1),
            "corr_xy_plot": str(p2),
        })

    summary = {
        "n_samples": int(X.shape[0]),
        "stats": stats,
        "sample_outputs": sample_outputs,
    }

    summary_path = OUTPUT_DIR / "data_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Сформировано sample-графиков: %d", len(sample_outputs))
    logger.info("Сводка: %s", summary_path)
    logger.info("DATA ANALYSIS DONE")


if __name__ == "__main__":
    main()
