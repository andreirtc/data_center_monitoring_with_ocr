from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacenter_ocr.geometry_benchmark import prepare_benchmark_artifacts


def build_parser() -> argparse.ArgumentParser:
    """Build the geometry-only benchmark preparation parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare labeled fixed-versus-calibrated crops without running OCR."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "local_benchmark" / "geometry_ab",
        help="Folder for labels, crops, contact sheet, and manifest.",
    )
    parser.add_argument(
        "--trusted-labels",
        type=Path,
        default=PROJECT_ROOT / "local_benchmark" / "labels.csv",
        help="Existing trusted labels to reuse only on exact identity matches.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate Stage 3A artifacts and report labeling status."""

    arguments = build_parser().parse_args(argv)
    manifest = prepare_benchmark_artifacts(
        PROJECT_ROOT,
        arguments.output.resolve(),
        arguments.trusted_labels.resolve(),
    )
    print(f"Selected cells: {manifest['selection']['item_count']}")
    print(f"Trusted labels reused: {manifest['trusted_labels_reused']}")
    print(f"Missing labels: {manifest['missing_labels']}")
    print(f"Labels: {manifest['labels_path']}")
    print(f"Contact sheet: {manifest['labeling_contact_sheet_path']}")
    print("OCR was not initialized or run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
