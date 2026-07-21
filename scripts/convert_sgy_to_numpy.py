#!/usr/bin/env python3
"""
Конвертация всех SEG-Y из data/ в NumPy (data/processed/).

Использование:
    python scripts/convert_sgy_to_numpy.py
    python scripts/convert_sgy_to_numpy.py --data-dir data --output-dir data/processed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, PROCESSED_DATA_DIR, setup_directories
from src.data.sgy_converter import convert_all_sgy_in_directory
from src.utils import setup_logger


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert SEG-Y files to NumPy")
  parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
  parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
  args = parser.parse_args()

  setup_directories()
  logger = setup_logger("convert_sgy", ROOT / "outputs" / "logs" / "convert_sgy.log")
  logger.setLevel(logging.INFO)

  manifest = convert_all_sgy_in_directory(args.data_dir, args.output_dir)
  print(f"Converted {len(manifest['datasets'])} datasets -> {args.output_dir}")
  print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
  main()
