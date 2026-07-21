"""
FWI Pipeline — диспетчер.

Запуск:
    python main.py                  # CNN + PINN
    python main.py --mode cnn       # только CNN baseline
    python main.py --mode pinn      # только PINN
    python main.py --convert        # сначала конвертировать SEG-Y → NumPy
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

from src.config import LOGS_DIR, MODEL_MODE, OUTPUT_DIR, SPLIT_MODE, setup_directories
from src.utils import setup_logger
from src.visualization import plot_comparison_heatmap


def _convert_sgy() -> None:
  script = Path(__file__).parent / "scripts" / "convert_sgy_to_numpy.py"
  subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
  parser = argparse.ArgumentParser(description="EnergyCraft FWI Pipeline")
  parser.add_argument("--mode", choices=["all", "cnn", "pinn"], default=MODEL_MODE)
  parser.add_argument("--convert", action="store_true", help="Convert SEG-Y to NumPy first")
  args = parser.parse_args()

  setup_directories()
  logger = setup_logger("fwi.main", LOGS_DIR / "run.log")

  if args.convert:
    logger.info("Converting SEG-Y -> NumPy...")
    _convert_sgy()

  manifest = Path("data/processed/manifest.json")
  if not manifest.exists():
    logger.info("Manifest not found, running conversion...")
    _convert_sgy()

  if args.mode in ("all", "cnn"):
    logger.info(">>> CNN Baseline")
    from run_cnn_baseline import run_cnn_baseline
    run_cnn_baseline()

  if args.mode in ("all", "pinn"):
    logger.info(">>> PINN FWI")
    from run_pinn_fwi import run_pinn_fwi
    run_pinn_fwi()

  cnn_path = OUTPUT_DIR / "summary_cnn_baseline_combined.json"
  if not cnn_path.exists():
    cnn_path = OUTPUT_DIR / f"summary_cnn_baseline_{SPLIT_MODE}.json"
  pinn_path = OUTPUT_DIR / "summary_pinn_fwi_combined.json"
  if not pinn_path.exists():
    pinn_path = OUTPUT_DIR / f"summary_pinn_fwi_{SPLIT_MODE}.json"
  if cnn_path.exists() and pinn_path.exists():
    with open(cnn_path, encoding="utf-8") as f:
      cnn = json.load(f)
    with open(pinn_path, encoding="utf-8") as f:
      pinn = json.load(f)
    plot_comparison_heatmap(
      {
        "CNN": cnn.get("test_metrics_overall", cnn.get("test_metrics", {})),
        "PINN": pinn.get("per_window", [{}])[0] if pinn.get("per_window") else {},
      },
      title="CNN vs PINN — test metrics",
    )


if __name__ == "__main__":
  main()
