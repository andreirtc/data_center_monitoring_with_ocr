from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacenter_ocr.geometry_benchmark import (
    BLANK_COMPARISON_FIELDNAMES,
    build_blank_analysis_comparison,
    load_label_rows,
    load_mode_cells,
    summarize_blank_analysis,
    validate_complete_labels,
    write_csv_rows,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the non-OCR normalized blank-analysis parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy and normalized blank analysis on labeled crops "
            "without importing or running PaddleOCR."
        )
    )
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate labels and write before/after blank-analysis evidence."""

    arguments = build_parser().parse_args(argv)
    labels_path = arguments.labels.resolve()
    labels = validate_complete_labels(load_label_rows(labels_path))
    cells_by_mode = {
        mode: load_mode_cells(labels_path, labels, mode)
        for mode in ("fixed", "calibrated")
    }
    rows = build_blank_analysis_comparison(labels, cells_by_mode)
    summary = summarize_blank_analysis(rows)
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        arguments.output / "blank_analysis_comparison.csv",
        BLANK_COMPARISON_FIELDNAMES,
        rows,
    )
    (arguments.output / "blank_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PaddleOCR was not imported or run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
