"""
Аудит геометрического соответствия: сейсмограммы (input.sgy) ↔ модель скорости (Vp_*.sgy).

Ничего не внедряет в обучение. Только анализ и артефакты в outputs/.

Что сохраняет:
- outputs/geometry_shot_coverage.csv
- outputs/geometry_model_x_axis.csv
- outputs/geometry_coverage_summary.json
- outputs/plots/geometry_shot_coverage.png
- outputs/plots/geometry_overlap_count.png

Запуск:
    python audit_geometry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from segyio import TraceField

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.config import SEGY_PATH, VP_SEGY_PATH, OUTPUT_DIR, PLOTS_DIR, LOGS_DIR, setup_directories
from src.utils import setup_logger


def _apply_scalar(raw: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    """SEG-Y scalar application rule for coordinates.

    scalar > 0: value * scalar
    scalar < 0: value / abs(scalar)
    scalar == 0: value
    """
    raw = raw.astype(np.float64)
    scalar = scalar.astype(np.float64)

    out = raw.copy()
    pos = scalar > 0
    neg = scalar < 0

    out[pos] = raw[pos] * scalar[pos]
    out[neg] = raw[neg] / np.abs(scalar[neg])
    return out


def _read_input_geometry(segy_path: str) -> pd.DataFrame:
    """Читает shot-геометрию из input.sgy (по FieldRecord)."""
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        n = f.tracecount

        shot_id = np.array([f.header[i][TraceField.FieldRecord] for i in range(n)], dtype=np.int64)
        src_x_raw = np.array([f.header[i][TraceField.SourceX] for i in range(n)], dtype=np.float64)
        rec_x_raw = np.array([f.header[i][TraceField.GroupX] for i in range(n)], dtype=np.float64)
        scalar = np.array([f.header[i][TraceField.SourceGroupScalar] for i in range(n)], dtype=np.int64)

        # Переводим в физические координаты
        src_x = _apply_scalar(src_x_raw, scalar)
        rec_x = _apply_scalar(rec_x_raw, scalar)

        # Время из samples для справки
        time_samples = np.asarray(f.samples, dtype=np.float64)

    rows: List[Dict] = []
    for sh in np.unique(shot_id):
        idx = np.where(shot_id == sh)[0]
        rows.append({
            "shot_id": int(sh),
            "n_traces": int(len(idx)),
            "source_x": float(np.median(src_x[idx])),
            "receiver_x_min": float(np.min(rec_x[idx])),
            "receiver_x_max": float(np.max(rec_x[idx])),
            "receiver_x_span": float(np.max(rec_x[idx]) - np.min(rec_x[idx])),
            "scalar_median": int(np.median(scalar[idx])),
        })

    df = pd.DataFrame(rows).sort_values("shot_id").reset_index(drop=True)

    # Доп. справка по оси времени
    t_min = float(np.min(time_samples)) if time_samples.size else None
    t_max = float(np.max(time_samples)) if time_samples.size else None
    dt = float(np.median(np.diff(time_samples))) if time_samples.size > 1 else None
    df.attrs["time_axis"] = {
        "n_samples": int(time_samples.size),
        "t_min": t_min,
        "t_max": t_max,
        "dt_median": dt,
    }
    return df


def _read_vp_x_axis(vp_segy_path: str) -> pd.DataFrame:
    """Читает X-ось модели скорости по trace headers Vp-файла."""
    with segyio.open(vp_segy_path, "r", ignore_geometry=True) as f:
        n = f.tracecount
        gx_raw = np.array([f.header[i][TraceField.GroupX] for i in range(n)], dtype=np.float64)
        scalar = np.array([f.header[i][TraceField.SourceGroupScalar] for i in range(n)], dtype=np.int64)

        gx = _apply_scalar(gx_raw, scalar)

        # Если координаты немонотонные, сортируем индексы по X
        order = np.argsort(gx)
        gx_sorted = gx[order]

    return pd.DataFrame({
        "model_trace_idx": np.arange(len(gx_sorted), dtype=np.int64),
        "model_x": gx_sorted.astype(np.float64),
    })


def _map_shots_to_model(shot_df: pd.DataFrame, model_x_df: pd.DataFrame) -> pd.DataFrame:
    """Сопоставляет X-интервал приёмников каждого shot к индексам X модели."""
    model_x = model_x_df["model_x"].to_numpy()
    model_x_min = float(np.min(model_x))
    model_x_max = float(np.max(model_x))

    mapped_rows: List[Dict] = []
    for _, r in shot_df.iterrows():
        rx_min = float(r["receiver_x_min"])
        rx_max = float(r["receiver_x_max"])

        # Пересечение с диапазоном модели
        ov_min = max(rx_min, model_x_min)
        ov_max = min(rx_max, model_x_max)
        has_overlap = ov_max > ov_min

        if has_overlap:
            idx_min = int(np.searchsorted(model_x, ov_min, side="left"))
            idx_max = int(np.searchsorted(model_x, ov_max, side="right") - 1)
            idx_min = max(0, min(idx_min, len(model_x) - 1))
            idx_max = max(0, min(idx_max, len(model_x) - 1))
            model_count = int(idx_max - idx_min + 1)
            overlap_width = float(ov_max - ov_min)
        else:
            idx_min = -1
            idx_max = -1
            model_count = 0
            overlap_width = 0.0

        mapped_rows.append({
            **r.to_dict(),
            "model_x_min": model_x_min,
            "model_x_max": model_x_max,
            "overlap_x_min": float(ov_min) if has_overlap else np.nan,
            "overlap_x_max": float(ov_max) if has_overlap else np.nan,
            "overlap_width": overlap_width,
            "has_overlap": bool(has_overlap),
            "model_ix_min": int(idx_min),
            "model_ix_max": int(idx_max),
            "model_ix_count": int(model_count),
        })

    return pd.DataFrame(mapped_rows)


def _plot_shot_coverage(mapped_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))

    y_positions = np.arange(len(mapped_df))
    for i, (_, r) in enumerate(mapped_df.iterrows()):
        # Полный spread приёмников (синий)
        ax.plot([r["receiver_x_min"], r["receiver_x_max"]], [i, i], color="tab:blue", lw=2)
        # Пересечение с моделью (красный)
        if bool(r["has_overlap"]):
            ax.plot([r["overlap_x_min"], r["overlap_x_max"]], [i, i], color="tab:red", lw=4)
        # Позиция источника
        ax.scatter([r["source_x"]], [i], color="black", s=15, zorder=3)

    model_x_min = float(mapped_df["model_x_min"].iloc[0])
    model_x_max = float(mapped_df["model_x_max"].iloc[0])
    ax.axvline(model_x_min, color="tab:green", ls="--", lw=1.5, label="Model X min/max")
    ax.axvline(model_x_max, color="tab:green", ls="--", lw=1.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"shot {int(s)}" for s in mapped_df["shot_id"].tolist()])
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Shot")
    ax.set_title("Shot receiver coverage vs model X extent")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_overlap_count(mapped_df: pd.DataFrame, model_x_df: pd.DataFrame, out_path: Path) -> None:
    n = len(model_x_df)
    count = np.zeros(n, dtype=np.int64)

    for _, r in mapped_df.iterrows():
        if not bool(r["has_overlap"]):
            continue
        a = int(r["model_ix_min"])
        b = int(r["model_ix_max"])
        if a >= 0 and b >= 0 and b >= a:
            count[a:b + 1] += 1

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(model_x_df["model_x"].to_numpy(), count, color="tab:purple", lw=1.5)
    ax.set_xlabel("Model X coordinate")
    ax.set_ylabel("Number of covering shots")
    ax.set_title("Shot overlap count along model X")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    setup_directories()
    logger = setup_logger("fwi.geometry.audit", LOGS_DIR / "geometry_audit.log")

    logger.info("=" * 70)
    logger.info("GEOMETRY AUDIT START")
    logger.info("SEGY_PATH: %s", SEGY_PATH)
    logger.info("VP_SEGY_PATH: %s", VP_SEGY_PATH)
    logger.info("=" * 70)

    shot_df = _read_input_geometry(SEGY_PATH)
    model_x_df = _read_vp_x_axis(VP_SEGY_PATH)
    mapped_df = _map_shots_to_model(shot_df, model_x_df)

    # Сохранения
    out_cov_csv = OUTPUT_DIR / "geometry_shot_coverage.csv"
    out_model_csv = OUTPUT_DIR / "geometry_model_x_axis.csv"
    out_cov_plot = PLOTS_DIR / "geometry_shot_coverage.png"
    out_count_plot = PLOTS_DIR / "geometry_overlap_count.png"

    mapped_df.to_csv(out_cov_csv, index=False, encoding="utf-8")
    model_x_df.to_csv(out_model_csv, index=False, encoding="utf-8")

    _plot_shot_coverage(mapped_df, out_cov_plot)
    _plot_overlap_count(mapped_df, model_x_df, out_count_plot)

    time_axis = shot_df.attrs.get("time_axis", {})
    summary = {
        "n_shots": int(len(mapped_df)),
        "shots_with_overlap": int(mapped_df["has_overlap"].sum()),
        "shots_without_overlap": int((~mapped_df["has_overlap"]).sum()),
        "model_x_min": float(model_x_df["model_x"].min()),
        "model_x_max": float(model_x_df["model_x"].max()),
        "receiver_x_min_global": float(mapped_df["receiver_x_min"].min()),
        "receiver_x_max_global": float(mapped_df["receiver_x_max"].max()),
        "time_axis_input": time_axis,
        "artifacts": {
            "shot_coverage_csv": str(out_cov_csv),
            "model_x_csv": str(out_model_csv),
            "shot_coverage_plot": str(out_cov_plot),
            "overlap_count_plot": str(out_count_plot),
        },
    }

    out_summary = OUTPUT_DIR / "geometry_coverage_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved: %s", out_cov_csv)
    logger.info("Saved: %s", out_model_csv)
    logger.info("Saved: %s", out_cov_plot)
    logger.info("Saved: %s", out_count_plot)
    logger.info("Saved: %s", out_summary)
    logger.info("Shots with overlap: %d / %d", summary["shots_with_overlap"], summary["n_shots"])
    logger.info("GEOMETRY AUDIT DONE")


if __name__ == "__main__":
    main()
