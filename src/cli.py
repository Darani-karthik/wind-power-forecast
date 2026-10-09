"""Command-line options shared by the entry points."""
from __future__ import annotations

import argparse

from .data import default_data_dir
from .models import MODEL_NAMES


def parse_args(description: str, argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", default=default_data_dir(),
                        help="folder with the SDWPF files (default: $SDWPF_DATA_DIR or ./data)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--turbines", type=int, default=30, help="number of turbines, TurbID 1..N")
    parser.add_argument("--days", type=int, default=60, help="days of data from the start of the dataset")
    parser.add_argument("--k", type=int, default=5, help="neighbours per turbine in the graph")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args(argv)
