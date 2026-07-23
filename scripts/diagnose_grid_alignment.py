from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacenter_ocr.grid_diagnostics import run_grid_diagnostics


def build_parser() -> argparse.ArgumentParser:
    """Build the geometry-only diagnostic command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect fixed-grid alignment without initializing or running OCR."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Monitoring-sheet photograph or scan.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Folder for geometry diagnostic artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run table preparation, cell extraction, and alignment diagnostics."""

    arguments = build_parser().parse_args(argv)
    outputs, report = run_grid_diagnostics(arguments.image, arguments.output)

    horizontal = report["horizontal_measurements"]
    vertical = report["vertical_measurements"]
    print(f"Warped table: {outputs.warped_table}")
    print(f"Grid overlay: {outputs.measurement_grid_overlay}")
    print(f"Line overlay: {outputs.line_diagnostic_overlay}")
    print(f"Contact sheet: {outputs.contact_sheet}")
    print(f"Alignment report: {outputs.alignment_report}")
    print(
        "Horizontal lines: "
        f"{horizontal['matched_count']}/{horizontal['expected_count']} matched; "
        f"median error {horizontal['median_error_pixels']} px; "
        f"p95 {horizontal['p95_error_pixels']} px"
    )
    print(
        "Vertical lines: "
        f"{vertical['matched_count']}/{vertical['expected_count']} matched; "
        f"median error {vertical['median_error_pixels']} px; "
        f"p95 {vertical['p95_error_pixels']} px"
    )
    print(
        "Provisional uncalibrated alignment score: "
        f"{report['provisional_alignment_score']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
