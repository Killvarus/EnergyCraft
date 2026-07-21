"""
Расширенный аудит согласованности координат модели и SEG-Y для подготовки данных FWI.

Не внедряет ничего в обучение. Только анализ, таблицы, графики и рекомендации.

Артефакты:
- outputs/coord_audit_segy_trace_stats.csv
- outputs/coord_audit_shot_stats.csv
- outputs/coord_audit_model_stats.json
- outputs/coord_audit_hypotheses.json
- outputs/coord_audit_window_recommendations.csv
- outputs/coord_audit_final_report.md
- outputs/plots/coord_audit_graph1_raw_shots.png
- outputs/plots/coord_audit_graph2_transformed_shots.png
- outputs/plots/coord_audit_graph3_ranges_overlay.png
- outputs/plots/coord_audit_graph4_coverage_map.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from segyio import TraceField

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.config import SEGY_PATH, VP_SEGY_PATH, OUTPUT_DIR, PLOTS_DIR, LOGS_DIR, setup_directories
from src.utils import setup_logger


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _apply_scalar(raw: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.float64)
    scalar = scalar.astype(np.float64)
    out = raw.copy()
    pos = scalar > 0
    neg = scalar < 0
    out[pos] = raw[pos] * scalar[pos]
    out[neg] = raw[neg] / np.abs(scalar[neg])
    return out


def _safe_header_array(f: segyio.SegyFile, field_name: str, n: int) -> np.ndarray | None:
    if not hasattr(TraceField, field_name):
        return None
    key = getattr(TraceField, field_name)
    try:
        return np.array([f.header[i][key] for i in range(n)], dtype=np.int64)
    except Exception:
        return None


def _first_valid(*arrays: np.ndarray | None) -> np.ndarray | None:
    for arr in arrays:
        if arr is None:
            continue
        if np.any(arr != 0):
            return arr
    return None


def _range_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "unique": int(len(np.unique(arr))),
    }


def _coverage_ratio(interval_min: float, interval_max: float, model_min: float, model_max: float) -> float:
    ov_min = max(interval_min, model_min)
    ov_max = min(interval_max, model_max)
    if ov_max <= ov_min:
        return 0.0
    model_len = model_max - model_min
    return float((ov_max - ov_min) / model_len) if model_len > 0 else 0.0


# -----------------------------------------------------------------------------
# 1-2. SEG-Y coordinate analysis + statistics
# -----------------------------------------------------------------------------
def analyze_segy_geometry(segy_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        n = f.tracecount
        samples = np.asarray(f.samples, dtype=np.float64)

        field_record = _safe_header_array(f, "FieldRecord", n)
        trace_number = _safe_header_array(f, "TraceNumber", n)
        scalco = _safe_header_array(f, "SourceGroupScalar", n)
        if scalco is None:
            scalco = np.zeros(n, dtype=np.int64)

        source_x_raw = _safe_header_array(f, "SourceX", n)
        source_y_raw = _safe_header_array(f, "SourceY", n)
        group_x_raw = _safe_header_array(f, "GroupX", n)
        group_y_raw = _safe_header_array(f, "GroupY", n)
        shotpoint_raw = _safe_header_array(f, "ShotPoint", n)
        cdp_raw = _safe_header_array(f, "CDP", n)

        source_x = _apply_scalar(source_x_raw, scalco) if source_x_raw is not None else np.zeros(n)
        source_y = _apply_scalar(source_y_raw, scalco) if source_y_raw is not None else np.zeros(n)
        group_x = _apply_scalar(group_x_raw, scalco) if group_x_raw is not None else np.zeros(n)
        group_y = _apply_scalar(group_y_raw, scalco) if group_y_raw is not None else np.zeros(n)

        trace_df = pd.DataFrame({
            "trace_index": np.arange(n, dtype=np.int64),
            "field_record": field_record if field_record is not None else np.zeros(n, dtype=np.int64),
            "trace_number": trace_number if trace_number is not None else np.zeros(n, dtype=np.int64),
            "scalco": scalco,
            "source_x_raw": source_x_raw if source_x_raw is not None else np.zeros(n, dtype=np.int64),
            "source_y_raw": source_y_raw if source_y_raw is not None else np.zeros(n, dtype=np.int64),
            "group_x_raw": group_x_raw if group_x_raw is not None else np.zeros(n, dtype=np.int64),
            "group_y_raw": group_y_raw if group_y_raw is not None else np.zeros(n, dtype=np.int64),
            "source_x": source_x,
            "source_y": source_y,
            "group_x": group_x,
            "group_y": group_y,
            "shotpoint": shotpoint_raw if shotpoint_raw is not None else np.zeros(n, dtype=np.int64),
            "cdp": cdp_raw if cdp_raw is not None else np.zeros(n, dtype=np.int64),
        })

    # shot-level stats
    rows: List[Dict] = []
    for shot in sorted(trace_df["field_record"].unique()):
        sub = trace_df[trace_df["field_record"] == shot]
        gx = np.sort(sub["group_x"].to_numpy(dtype=float))
        src = sub["source_x"].to_numpy(dtype=float)
        shot_step = np.nan
        if len(gx) > 1:
            dif = np.diff(gx)
            dif = dif[np.abs(dif) > 0]
            shot_step = float(np.mean(dif)) if len(dif) else 0.0
        rows.append({
            "shot_id": int(shot),
            "n_receivers": int(len(sub)),
            "source_x": float(np.median(src)),
            "source_y": float(np.median(sub["source_y"])),
            "receiver_x_min": float(np.min(sub["group_x"])),
            "receiver_x_max": float(np.max(sub["group_x"])),
            "receiver_y_min": float(np.min(sub["group_y"])),
            "receiver_y_max": float(np.max(sub["group_y"])),
            "receiver_dx_mean": shot_step,
            "scalco_median": int(np.median(sub["scalco"])),
        })
    shot_df = pd.DataFrame(rows)

    # global SEG-Y stats
    src_unique = np.sort(shot_df["source_x"].unique())
    src_steps = np.diff(src_unique)
    src_steps = src_steps[np.abs(src_steps) > 0]

    summary = {
        "source_x": _range_stats(trace_df["source_x"].to_numpy()),
        "source_y": _range_stats(trace_df["source_y"].to_numpy()),
        "group_x": _range_stats(trace_df["group_x"].to_numpy()),
        "group_y": _range_stats(trace_df["group_y"].to_numpy()),
        "shotpoint": _range_stats(trace_df["shotpoint"].to_numpy()),
        "cdp": _range_stats(trace_df["cdp"].to_numpy()),
        "scalco_unique": {str(int(k)): int(v) for k, v in zip(*np.unique(trace_df["scalco"], return_counts=True))},
        "n_unique_sources": int(len(src_unique)),
        "receivers_per_shot_min": int(shot_df["n_receivers"].min()),
        "receivers_per_shot_max": int(shot_df["n_receivers"].max()),
        "receivers_per_shot_mean": float(shot_df["n_receivers"].mean()),
        "receiver_dx_mean": float(np.nanmean(shot_df["receiver_dx_mean"])),
        "source_dx_mean": float(np.mean(src_steps)) if len(src_steps) else None,
        "time_axis": {
            "n_samples": int(len(samples)),
            "tmin": float(samples.min()) if len(samples) else None,
            "tmax": float(samples.max()) if len(samples) else None,
            "dt": float(np.median(np.diff(samples))) if len(samples) > 1 else None,
        },
    }

    return trace_df, shot_df, summary


# -----------------------------------------------------------------------------
# 3. Velocity model analysis
# -----------------------------------------------------------------------------
def analyze_velocity_model(vp_path: str) -> Dict:
    with segyio.open(vp_path, "r", ignore_geometry=True) as f:
        data = f.trace.raw[:]
        n_traces = f.tracecount
        n_samples = f.samples.size
        samples = np.asarray(f.samples, dtype=np.float64)

        scalco = _safe_header_array(f, "SourceGroupScalar", n_traces)
        if scalco is None:
            scalco = np.zeros(n_traces, dtype=np.int64)

        group_x_raw = _safe_header_array(f, "GroupX", n_traces)
        group_y_raw = _safe_header_array(f, "GroupY", n_traces)
        cdp_x_raw = _safe_header_array(f, "CDP_X", n_traces)
        cdp_y_raw = _safe_header_array(f, "CDP_Y", n_traces)

        group_x = _apply_scalar(group_x_raw, scalco) if group_x_raw is not None else None
        group_y = _apply_scalar(group_y_raw, scalco) if group_y_raw is not None else None
        cdp_x = _apply_scalar(cdp_x_raw, scalco) if cdp_x_raw is not None else None
        cdp_y = _apply_scalar(cdp_y_raw, scalco) if cdp_y_raw is not None else None

        x_axis = _first_valid(group_x, cdp_x)
        y_axis = _first_valid(group_y, cdp_y)

        # model grid params
        nx = int(n_traces)
        nz = int(n_samples)
        if x_axis is not None:
            x_sorted = np.sort(np.unique(x_axis))
            dx = float(np.median(np.diff(x_sorted))) if len(x_sorted) > 1 else None
            xmin = float(x_sorted.min())
            xmax = float(x_sorted.max())
        else:
            dx = None
            xmin = None
            xmax = None

        if len(samples) > 1:
            dz = float(np.median(np.diff(samples)))
            zmin = float(samples.min())
            zmax = float(samples.max())
        else:
            dz = None
            zmin = float(samples.min()) if len(samples) else None
            zmax = float(samples.max()) if len(samples) else None

    return {
        "data_shape": list(data.shape),
        "value_min": float(data.min()),
        "value_max": float(data.max()),
        "value_mean": float(data.mean()),
        "value_std": float(data.std()),
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "x_axis_source": "GroupX" if group_x is not None and np.any(group_x != 0) else ("CDP_X" if cdp_x is not None and np.any(cdp_x != 0) else None),
        "y_axis_source": "GroupY" if group_y is not None and np.any(group_y != 0) else ("CDP_Y" if cdp_y is not None and np.any(cdp_y != 0) else None),
        "group_x_stats": _range_stats(group_x) if group_x is not None else None,
        "group_y_stats": _range_stats(group_y) if group_y is not None else None,
        "cdp_x_stats": _range_stats(cdp_x) if cdp_x is not None else None,
        "cdp_y_stats": _range_stats(cdp_y) if cdp_y is not None else None,
        "scalco_unique": {str(int(k)): int(v) for k, v in zip(*np.unique(scalco, return_counts=True))},
    }


# -----------------------------------------------------------------------------
# 4. Hypothesis checks A-D
# -----------------------------------------------------------------------------
def _transform_identity(x: np.ndarray) -> np.ndarray:
    return x.copy()


def _transform_offset(x: np.ndarray, offset: float) -> np.ndarray:
    return x + offset


def _transform_scale(x: np.ndarray, scale: float) -> np.ndarray:
    return x * scale


def _transform_scale_offset(x: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return x * scale + offset


def evaluate_hypotheses(shot_df: pd.DataFrame, model_stats: Dict) -> Tuple[Dict, str, Dict]:
    model_xmin = float(model_stats["xmin"])
    model_xmax = float(model_stats["xmax"])
    model_len = model_xmax - model_xmin

    shot_x = shot_df["source_x"].to_numpy(dtype=float)
    rec_min = shot_df["receiver_x_min"].to_numpy(dtype=float)
    rec_max = shot_df["receiver_x_max"].to_numpy(dtype=float)

    shot_xmin = float(np.min(shot_x))
    shot_xmax = float(np.max(shot_x))
    shot_len = shot_xmax - shot_xmin

    def score_ranges(src: np.ndarray, rmin: np.ndarray, rmax: np.ndarray) -> Dict:
        src_min = float(np.min(src))
        src_max = float(np.max(src))
        rec_global_min = float(np.min(rmin))
        rec_global_max = float(np.max(rmax))
        within_receiver = int(np.sum((rmin >= model_xmin) & (rmax <= model_xmax)))
        partial_overlap = int(np.sum((rmax > model_xmin) & (rmin < model_xmax)))
        src_inside = int(np.sum((src >= model_xmin) & (src <= model_xmax)))
        src_match_err = abs(src_min - model_xmin) + abs(src_max - model_xmax)
        rec_cover_ratio = _coverage_ratio(rec_global_min, rec_global_max, model_xmin, model_xmax)
        return {
            "src_min": src_min,
            "src_max": src_max,
            "rec_min": rec_global_min,
            "rec_max": rec_global_max,
            "within_receiver_count": within_receiver,
            "partial_overlap_count": partial_overlap,
            "source_inside_count": src_inside,
            "source_range_match_error": float(src_match_err),
            "receiver_cover_ratio": float(rec_cover_ratio),
        }

    hypotheses = {}

    # A: already matched
    hypotheses["A_identity"] = {
        "formula": "X_model = X_segy",
        "params": {},
        "result": score_ranges(shot_x, rec_min, rec_max),
    }

    # B: constant offset by shot xmin
    offset_b = model_xmin - shot_xmin
    hypotheses["B_offset"] = {
        "formula": f"X_model = X_segy + {offset_b:.6f}",
        "params": {"offset": float(offset_b)},
        "result": score_ranges(_transform_offset(shot_x, offset_b), _transform_offset(rec_min, offset_b), _transform_offset(rec_max, offset_b)),
    }

    # C: pure scale by shot length
    scale_c = (model_len / shot_len) if shot_len != 0 else 1.0
    hypotheses["C_scale"] = {
        "formula": f"X_model = X_segy * {scale_c:.6f}",
        "params": {"scale": float(scale_c)},
        "result": score_ranges(_transform_scale(shot_x, scale_c), _transform_scale(rec_min, scale_c), _transform_scale(rec_max, scale_c)),
    }

    # D: local profile coords -> global via + model xmin
    offset_d = model_xmin
    hypotheses["D_local_plus_model_xmin"] = {
        "formula": f"X_model = X_segy + {offset_d:.6f}",
        "params": {"offset": float(offset_d)},
        "result": score_ranges(_transform_offset(shot_x, offset_d), _transform_offset(rec_min, offset_d), _transform_offset(rec_max, offset_d)),
    }

    # affine fit from receiver min/max ranges
    scale_aff = (model_len / (float(np.max(rec_max)) - float(np.min(rec_min)))) if (float(np.max(rec_max)) - float(np.min(rec_min))) != 0 else 1.0
    offset_aff = model_xmin - float(np.min(rec_min)) * scale_aff
    hypotheses["E_scale_offset"] = {
        "formula": f"X_model = X_segy * {scale_aff:.6f} + {offset_aff:.6f}",
        "params": {"scale": float(scale_aff), "offset": float(offset_aff)},
        "result": score_ranges(
            _transform_scale_offset(shot_x, scale_aff, offset_aff),
            _transform_scale_offset(rec_min, scale_aff, offset_aff),
            _transform_scale_offset(rec_max, scale_aff, offset_aff),
        ),
    }

    # heuristic best: maximize partial overlap, then receiver cover, then min error
    def rank(item: Dict) -> Tuple:
        r = item["result"]
        return (
            r["partial_overlap_count"],
            r["within_receiver_count"],
            r["receiver_cover_ratio"],
            -r["source_range_match_error"],
        )

    best_name = max(hypotheses.keys(), key=lambda k: rank(hypotheses[k]))
    best = hypotheses[best_name]
    return hypotheses, best_name, best


# -----------------------------------------------------------------------------
# 5-6. Diagnostic plots and coverage
# -----------------------------------------------------------------------------
def _apply_hypothesis_to_shots(shot_df: pd.DataFrame, best_name: str, best: Dict) -> pd.DataFrame:
    df = shot_df.copy()
    params = best.get("params", {})
    scale = float(params.get("scale", 1.0))
    offset = float(params.get("offset", 0.0))

    for col in ["source_x", "receiver_x_min", "receiver_x_max"]:
        df[f"{col}_transformed"] = df[col].to_numpy(dtype=float) * scale + offset
    return df


def plot_graph1_raw_shots(shot_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(shot_df))
    for i, (_, r) in enumerate(shot_df.iterrows()):
        ax.plot([r["receiver_x_min"], r["receiver_x_max"]], [i, i], color="tab:blue", lw=2)
        ax.scatter([r["source_x"]], [i], color="black", s=18)
    ax.set_title("Graph 1: raw SEG-Y coordinates by shot")
    ax.set_xlabel("SEG-Y real X after SCALCO")
    ax.set_ylabel("Shot index")
    ax.set_yticks(y)
    ax.set_yticklabels([str(int(s)) for s in shot_df["shot_id"].tolist()])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_graph2_transformed_shots(shot_df_t: pd.DataFrame, model_stats: Dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(shot_df_t))
    for i, (_, r) in enumerate(shot_df_t.iterrows()):
        ax.plot([r["receiver_x_min_transformed"], r["receiver_x_max_transformed"]], [i, i], color="tab:orange", lw=2)
        ax.scatter([r["source_x_transformed"]], [i], color="black", s=18)
    ax.axvline(model_stats["xmin"], color="tab:green", ls="--")
    ax.axvline(model_stats["xmax"], color="tab:green", ls="--")
    ax.set_title("Graph 2: transformed shot geometry")
    ax.set_xlabel("Transformed X")
    ax.set_ylabel("Shot index")
    ax.set_yticks(y)
    ax.set_yticklabels([str(int(s)) for s in shot_df_t["shot_id"].tolist()])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_graph3_ranges_overlay(shot_df: pd.DataFrame, shot_df_t: pd.DataFrame, model_stats: Dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.5))
    model_xmin, model_xmax = model_stats["xmin"], model_stats["xmax"]
    ax.plot([model_xmin, model_xmax], [2, 2], color="tab:green", lw=6, label="Model X range")
    ax.plot([shot_df["source_x"].min(), shot_df["source_x"].max()], [1, 1], color="black", lw=4, label="Shot X raw")
    ax.plot([shot_df["receiver_x_min"].min(), shot_df["receiver_x_max"].max()], [0, 0], color="tab:blue", lw=4, label="Receiver X raw")
    ax.plot([shot_df_t["source_x_transformed"].min(), shot_df_t["source_x_transformed"].max()], [1.4, 1.4], color="gray", lw=3, label="Shot X transformed")
    ax.plot([shot_df_t["receiver_x_min_transformed"].min(), shot_df_t["receiver_x_max_transformed"].max()], [0.4, 0.4], color="tab:orange", lw=3, label="Receiver X transformed")
    ax.set_yticks([0, 0.4, 1, 1.4, 2])
    ax.set_yticklabels(["Receiver raw", "Receiver tr", "Shot raw", "Shot tr", "Model"])
    ax.set_title("Graph 3: overlay of model / shot / receiver ranges")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_graph4_coverage_map(shot_df_t: pd.DataFrame, model_stats: Dict, out_path: Path) -> pd.DataFrame:
    x0, x1 = float(model_stats["xmin"]), float(model_stats["xmax"])
    dx = float(model_stats["dx"]) if model_stats.get("dx") else (x1 - x0) / max(1, int(model_stats["nx"]) - 1)
    nx = int(model_stats["nx"])
    x_axis = np.linspace(x0, x1, nx)
    count = np.zeros(nx, dtype=np.int64)

    for _, r in shot_df_t.iterrows():
        a = max(float(r["receiver_x_min_transformed"]), x0)
        b = min(float(r["receiver_x_max_transformed"]), x1)
        if b <= a:
            continue
        ia = int(np.searchsorted(x_axis, a, side="left"))
        ib = int(np.searchsorted(x_axis, b, side="right") - 1)
        ia = max(0, min(ia, nx - 1))
        ib = max(0, min(ib, nx - 1))
        if ib >= ia:
            count[ia:ib + 1] += 1

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(x_axis, count, color="tab:purple", lw=1.5)
    ax.axvline(x0, color="tab:green", ls="--")
    ax.axvline(x1, color="tab:green", ls="--")
    ax.set_title("Graph 4: coverage map along model X")
    ax.set_xlabel("Model X")
    ax.set_ylabel("Number of covering shots")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return pd.DataFrame({"x": x_axis, "coverage_count": count})


# -----------------------------------------------------------------------------
# 7-8. Recommendations for train/val/test and windows
# -----------------------------------------------------------------------------
def recommend_windows(coverage_df: pd.DataFrame, shot_df_t: pd.DataFrame, window_size: float) -> pd.DataFrame:
    x = coverage_df["x"].to_numpy(dtype=float)
    c = coverage_df["coverage_count"].to_numpy(dtype=int)
    x0, x1 = float(x.min()), float(x.max())

    rows: List[Dict] = []
    start = x0
    while start < x1:
        end = min(start + window_size, x1)
        mask = (x >= start) & (x < end if end < x1 else x <= end)
        mean_cov = float(np.mean(c[mask])) if np.any(mask) else 0.0

        shots = shot_df_t[
            (shot_df_t["receiver_x_max_transformed"] > start) &
            (shot_df_t["receiver_x_min_transformed"] < end)
        ]["shot_id"].tolist()

        traces_est = int(sum(
            shot_df_t[
                (shot_df_t["shot_id"] == s)
            ]["n_receivers"].iloc[0] for s in shots
        )) if shots else 0

        rows.append({
            "xmin": float(start),
            "xmax": float(end),
            "shots": ",".join(str(int(s)) for s in shots),
            "n_shots": int(len(shots)),
            "n_traces_est": traces_est,
            "coverage_mean": mean_cov,
            "has_sufficient_coverage": bool(mean_cov > 0),
        })
        start = end

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 9. Final report
# -----------------------------------------------------------------------------
def build_final_report(model_stats: Dict, segy_summary: Dict, hypotheses: Dict, best_name: str, best: Dict,
                       shot_df_t: pd.DataFrame, coverage_df: pd.DataFrame, window_df: pd.DataFrame) -> str:
    shots_inside = int(np.sum((shot_df_t["source_x_transformed"] >= model_stats["xmin"]) & (shot_df_t["source_x_transformed"] <= model_stats["xmax"])))
    recv_outside = int(np.sum((shot_df_t["receiver_x_min_transformed"] < model_stats["xmin"]) | (shot_df_t["receiver_x_max_transformed"] > model_stats["xmax"])))
    unused_ratio = float(np.mean(coverage_df["coverage_count"].to_numpy() == 0))

    preferred_split = "по пространственным окнам модели" if unused_ratio < 0.5 else "по shot"

    lines = [
        "# Итоговый отчет по аудиту согласованности координат",
        "",
        "## Координатная система модели",
        f"- xmin = {model_stats['xmin']}",
        f"- xmax = {model_stats['xmax']}",
        f"- dx = {model_stats['dx']}",
        f"- nx = {model_stats['nx']}",
        f"- zmin = {model_stats['zmin']}",
        f"- zmax = {model_stats['zmax']}",
        f"- dz = {model_stats['dz']}",
        f"- nz = {model_stats['nz']}",
        "",
        "## Координатная система SEG-Y",
        f"- SourceX min/max = {segy_summary['source_x']['min']} / {segy_summary['source_x']['max']}",
        f"- GroupX min/max = {segy_summary['group_x']['min']} / {segy_summary['group_x']['max']}",
        f"- SourceY min/max = {segy_summary['source_y']['min']} / {segy_summary['source_y']['max']}",
        f"- GroupY min/max = {segy_summary['group_y']['min']} / {segy_summary['group_y']['max']}",
        f"- SCALCO = {segy_summary['scalco_unique']}",
        "",
        "## Результат проверки",
        f"- Лучшая гипотеза: {best_name}",
        f"- Формула: {best['formula']}",
        f"- Source inside model count: {best['result']['source_inside_count']}",
        f"- Receiver partial overlap count: {best['result']['partial_overlap_count']}",
        f"- Receiver fully within model count: {best['result']['within_receiver_count']}",
        f"- Receiver cover ratio: {best['result']['receiver_cover_ratio']}",
        "",
        "## Финальное заключение",
        f"- Можно ли напрямую сопоставлять сейсмограммы и модель: {'да' if best_name == 'A_identity' else 'нет'}",
        f"- Какое преобразование необходимо: {best['formula']}",
        f"- Предпочтительный вариант разбиения: {preferred_split}",
        "",
        "## Рекомендации по train/validation/test",
        "### Вариант 1 — по shot",
        "- Плюсы: простая реализация, нет утечки между одинаковыми shot.",
        "- Риски: разные shot могут покрывать общую часть модели; spatial leakage остаётся.",
        "",
        "### Вариант 2 — по пространственным окнам модели",
        "- Плюсы: лучше контролируется локальность target и coverage.",
        "- Риски: требуется корректное сопоставление трасс и окон; возможны окна с плохим покрытием.",
        "",
        "## Остаточные риски",
        f"- Источников внутри модели после преобразования: {shots_inside} / {len(shot_df_t)}",
        f"- Shot с приёмниками вне модели после преобразования: {recv_outside}",
        f"- Доля неиспользуемой зоны модели по X: {unused_ratio:.4f}",
        "",
        "## Дополнительные проверки",
        "- Проверить, не локальные ли SourceX/GroupX в input.sgy относительно начала профиля.",
        "- Проверить метаданные/сопроводительные файлы модели на явные xmin, dx, datum.",
        "- Проверить, не требуется ли affine-преобразование отдельно по SourceX и GroupX.",
        "- Проверить покрытие по глубине (а не только по X) перед формированием окон FWI.",
    ]
    return "\n".join(lines)


def main() -> None:
    setup_directories()
    logger = setup_logger("fwi.coordinate.audit", LOGS_DIR / "coordinate_audit.log")

    logger.info("=" * 80)
    logger.info("COORDINATE CONSISTENCY AUDIT START")
    logger.info("SEGY_PATH: %s", SEGY_PATH)
    logger.info("VP_SEGY_PATH: %s", VP_SEGY_PATH)
    logger.info("=" * 80)

    # 1-2 SEG-Y analysis
    trace_df, shot_df, segy_summary = analyze_segy_geometry(SEGY_PATH)

    # 3 model analysis
    model_stats = analyze_velocity_model(VP_SEGY_PATH)

    # 4 hypotheses
    hypotheses, best_name, best = evaluate_hypotheses(shot_df, model_stats)
    shot_df_t = _apply_hypothesis_to_shots(shot_df, best_name, best)

    # 5 plots
    p1 = PLOTS_DIR / "coord_audit_graph1_raw_shots.png"
    p2 = PLOTS_DIR / "coord_audit_graph2_transformed_shots.png"
    p3 = PLOTS_DIR / "coord_audit_graph3_ranges_overlay.png"
    p4 = PLOTS_DIR / "coord_audit_graph4_coverage_map.png"
    plot_graph1_raw_shots(shot_df, p1)
    plot_graph2_transformed_shots(shot_df_t, model_stats, p2)
    plot_graph3_ranges_overlay(shot_df, shot_df_t, model_stats, p3)
    coverage_df = plot_graph4_coverage_map(shot_df_t, model_stats, p4)

    # 6-8 coverage + recommendations
    window_size = 500.0
    window_df = recommend_windows(coverage_df, shot_df_t, window_size=window_size)

    # save artifacts
    out_trace = OUTPUT_DIR / "coord_audit_segy_trace_stats.csv"
    out_shot = OUTPUT_DIR / "coord_audit_shot_stats.csv"
    out_model = OUTPUT_DIR / "coord_audit_model_stats.json"
    out_hyp = OUTPUT_DIR / "coord_audit_hypotheses.json"
    out_win = OUTPUT_DIR / "coord_audit_window_recommendations.csv"
    out_report = OUTPUT_DIR / "coord_audit_final_report.md"

    trace_df.to_csv(out_trace, index=False, encoding="utf-8")
    shot_df_t.to_csv(out_shot, index=False, encoding="utf-8")
    with open(out_model, "w", encoding="utf-8") as f:
        json.dump(model_stats, f, indent=2, ensure_ascii=False)
    with open(out_hyp, "w", encoding="utf-8") as f:
        json.dump({"hypotheses": hypotheses, "best_name": best_name, "best": best}, f, indent=2, ensure_ascii=False)
    window_df.to_csv(out_win, index=False, encoding="utf-8")

    report = build_final_report(model_stats, segy_summary, hypotheses, best_name, best, shot_df_t, coverage_df, window_df)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Saved: %s", out_trace)
    logger.info("Saved: %s", out_shot)
    logger.info("Saved: %s", out_model)
    logger.info("Saved: %s", out_hyp)
    logger.info("Saved: %s", out_win)
    logger.info("Saved: %s", out_report)
    logger.info("Best hypothesis: %s | formula: %s", best_name, best["formula"])
    logger.info("COORDINATE CONSISTENCY AUDIT DONE")


if __name__ == "__main__":
    main()
