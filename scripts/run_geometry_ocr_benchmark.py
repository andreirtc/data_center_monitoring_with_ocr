from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacenter_ocr.geometry_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BLANK_COMPARISON_FIELDNAMES,
    COMPARISON_FIELDNAMES,
    RESULT_FIELDNAMES,
    alignment_failure_lookup,
    assess_calibrated_safety,
    assess_hybrid_safety,
    build_blank_analysis_comparison,
    build_hybrid_rows,
    compare_geometry_rows,
    evaluate_prediction_result,
    load_benchmark_manifest,
    load_label_rows,
    load_mode_cells,
    summarize_blank_analysis,
    summarize_mode,
    validate_complete_labels,
    write_csv_rows,
)
from datacenter_ocr.processing_metrics import ProcessingMetrics


CELLS_PER_BATCH = 32


def build_parser() -> argparse.ArgumentParser:
    """Build the guarded fixed/calibrated/hybrid benchmark parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run identical OCR rules on fixed/calibrated crops and derive a "
            "confirmation-only benchmark hybrid."
        )
    )
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def production_ocr_settings() -> dict[str, Any]:
    """Return the settings shared by both benchmark geometry modes."""

    from datacenter_ocr.ocr_processing import (
        OCR_BATCH_SIZE,
        OCR_PADDING,
        OCR_SCALE,
        PRODUCTION_VARIANTS,
    )

    return {
        "preprocessing_variants": list(PRODUCTION_VARIANTS),
        "ocr_scale": OCR_SCALE,
        "ocr_padding": OCR_PADDING,
        "paddle_batch_size": OCR_BATCH_SIZE,
        "cells_per_production_batch": CELLS_PER_BATCH,
        "blank_detection": "production analyze_cell_for_blankness",
        "postprocessing": "production correct_numeric_prediction",
        "verification": "production verify_cell_results",
    }


def _predict_one_mode(
    *,
    model: Any,
    cells: Sequence[dict[str, Any]],
    processor: Callable[..., list[Any]],
) -> tuple[list[Any], ProcessingMetrics, float]:
    """Predict sheet-by-sheet so production contextual checks never cross sheets."""

    metrics = ProcessingMetrics(model_was_warm=True)
    results: list[Any] = []
    start = time.perf_counter()
    for sheet_id in ("sample", "april_2026", "may_2026"):
        sheet_cells = [cell for cell in cells if cell["sheet_id"] == sheet_id]
        prediction_cells = [
            {
                "filename": cell["filename"],
                "day": cell["day"],
                "point": cell["point"],
                "reading_type": cell["reading_type"],
                "image": cell["image"],
            }
            for cell in sheet_cells
        ]
        results.extend(
            processor(
                model=model,
                cells=prediction_cells,
                cells_per_batch=CELLS_PER_BATCH,
                metrics=metrics,
            )
        )
    return results, metrics, time.perf_counter() - start


def _warm_model(model: Any, image: Any) -> float:
    """Record a first inference separately from both measured benchmark modes."""

    from datacenter_ocr.cell_preprocessing import create_ocr_variants
    from datacenter_ocr.ocr_processing import OCR_BATCH_SIZE, prepare_for_ocr

    warmup_input = prepare_for_ocr(create_ocr_variants(image)["original"])
    start = time.perf_counter()
    warmup_results = list(
        model.predict(input=[warmup_input], batch_size=OCR_BATCH_SIZE)
    )
    elapsed = time.perf_counter() - start
    if len(warmup_results) != 1:
        raise RuntimeError("PaddleOCR warm-up returned an unexpected result count.")
    return elapsed


def counterbalanced_execution_orders() -> tuple[tuple[str, str], ...]:
    """Return the two mode orders used to remove first-run timing bias."""

    return (("fixed", "calibrated"), ("calibrated", "fixed"))


def summarize_timing_trials(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average timed trials; warm-up is intentionally not an input."""

    if not trials:
        raise ValueError("At least one timed trial is required.")
    input_counts = {int(trial["ocr_input_count"]) for trial in trials}
    if len(input_counts) != 1:
        raise RuntimeError("Counterbalanced OCR input counts changed between trials.")
    return {
        "trial_count": len(trials),
        "ocr_input_count": input_counts.pop(),
        "total_ocr_time_seconds": round(
            sum(float(trial["ocr_time_seconds"]) for trial in trials)
            / len(trials),
            6,
        ),
        "total_processing_time_seconds": round(
            sum(float(trial["processing_time_seconds"]) for trial in trials)
            / len(trials),
            6,
        ),
        "trials": [dict(trial) for trial in trials],
    }


def _result_signature(results: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            result.filename,
            tuple(sorted(result.predictions.items())),
            result.consensus_prediction,
            result.final_value,
            result.is_blank,
            result.needs_review,
            result.blocks_export,
        )
        for result in results
    )


def _run_counterbalanced_predictions(
    *,
    model: Any,
    cells_by_mode: Mapping[str, Sequence[dict[str, Any]]],
    processor: Callable[..., list[Any]],
) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]], bool]:
    """Run fixed/calibrated twice in opposite orders and average timing."""

    representative_results: dict[str, list[Any]] = {}
    signatures: dict[str, tuple[tuple[Any, ...], ...]] = {}
    timing_trials: dict[str, list[dict[str, Any]]] = {
        "fixed": [],
        "calibrated": [],
    }
    predictions_consistent = True
    for order_index, execution_order in enumerate(
        counterbalanced_execution_orders(), start=1
    ):
        for position, geometry_mode in enumerate(execution_order, start=1):
            results, metrics, processing_seconds = _predict_one_mode(
                model=model,
                cells=cells_by_mode[geometry_mode],
                processor=processor,
            )
            signature = _result_signature(results)
            if geometry_mode not in representative_results:
                representative_results[geometry_mode] = results
                signatures[geometry_mode] = signature
            elif signatures[geometry_mode] != signature:
                predictions_consistent = False
            timing_trials[geometry_mode].append(
                {
                    "order_index": order_index,
                    "position_in_order": position,
                    "execution_order": list(execution_order),
                    "ocr_input_count": metrics.ocr_input_image_count,
                    "ocr_time_seconds": metrics.ocr_prediction_seconds,
                    "processing_time_seconds": round(processing_seconds, 6),
                }
            )
    timing = {
        geometry_mode: summarize_timing_trials(trials)
        for geometry_mode, trials in timing_trials.items()
    }
    return representative_results, timing, predictions_consistent


def _blank_analysis_lookup(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    from datacenter_ocr.blank_cell_detection import analyze_cell_for_blankness

    lookup: dict[str, dict[str, Any]] = {}
    for cell in cells:
        analysis = analyze_cell_for_blankness(cell["image"])
        lookup[str(cell["filename"])] = {
            "ink_ratio": round(float(analysis.ink_ratio), 6),
            "component_count": analysis.significant_component_count,
            "largest_component_ratio": round(
                float(analysis.largest_component_ratio), 6
            ),
            "analysis_width": analysis.analysis_width,
            "analysis_height": analysis.analysis_height,
        }
    return lookup


def _join_results(
    *,
    labels: Sequence[dict[str, Any]],
    results: Sequence[Any],
    geometry_mode: str,
    failures: dict[tuple[str, str], bool],
    blank_analysis_by_filename: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate predictions only after OCR has completed without ground truth."""

    from datacenter_ocr.numeric_postprocessing import correct_numeric_prediction

    labels_by_filename = {str(label["filename"]): label for label in labels}
    result_by_filename = {result.filename: result for result in results}
    if set(labels_by_filename) != set(result_by_filename):
        raise RuntimeError(
            f"{geometry_mode} OCR results do not match the selected label identities."
        )
    evaluated: list[dict[str, Any]] = []
    for label in labels:
        result = result_by_filename[str(label["filename"])]
        proposed_value = (
            ""
            if result.is_blank
            else correct_numeric_prediction(
                result.consensus_prediction, result.reading_type
            ).corrected_text
        )
        evaluated.append(
            evaluate_prediction_result(
                label,
                result,
                geometry_mode,
                proposed_value,
                failures.get((geometry_mode, result.filename), False),
                blank_analysis_by_filename.get(result.filename),
            )
        )
    return evaluated


def run_benchmark(labels_path: Path, output_folder: Path) -> dict[str, Any]:
    """Validate labels, then run fixed/calibrated/hybrid on only 54 cells."""

    labels_path = labels_path.resolve()
    label_rows = load_label_rows(labels_path)
    labels = validate_complete_labels(label_rows)

    fixed_cells = load_mode_cells(labels_path, labels, "fixed")
    calibrated_cells = load_mode_cells(labels_path, labels, "calibrated")
    cells_by_mode = {
        "fixed": fixed_cells,
        "calibrated": calibrated_cells,
    }
    blank_comparison_rows = build_blank_analysis_comparison(
        labels, cells_by_mode
    )
    blank_analysis_summary = summarize_blank_analysis(blank_comparison_rows)
    benchmark_manifest = load_benchmark_manifest(labels_path)
    sheet_alignment_metrics = benchmark_manifest.get(
        "sheet_alignment_metrics", {}
    )
    expected_sheet_ids = {str(label["sheet_id"]) for label in labels}
    if set(sheet_alignment_metrics) != expected_sheet_ids:
        raise ValueError(
            "Benchmark manifest sheet alignment metrics do not match labels. "
            "Regenerate Stage 3B geometry artifacts before running OCR."
        )
    fixed_settings = production_ocr_settings()
    calibrated_settings = production_ocr_settings()
    if fixed_settings != calibrated_settings:
        raise RuntimeError("Fixed and calibrated OCR settings are not identical.")

    # PaddleOCR is deliberately imported only after all labels pass validation.
    from paddleocr import TextRecognition
    from datacenter_ocr.ocr_processing import (
        process_measurement_cells_with_blank_detection,
    )

    model_start = time.perf_counter()
    model = TextRecognition(device="cpu")
    model_construction_seconds = time.perf_counter() - model_start
    shape_warmup_seconds = {
        "fixed": round(_warm_model(model, fixed_cells[0]["image"]), 6),
        "calibrated": round(
            _warm_model(model, calibrated_cells[0]["image"]), 6
        ),
    }

    results_by_mode, timing_by_mode, predictions_consistent = (
        _run_counterbalanced_predictions(
            model=model,
            cells_by_mode=cells_by_mode,
            processor=process_measurement_cells_with_blank_detection,
        )
    )
    if not predictions_consistent:
        raise RuntimeError(
            "Counterbalanced trials produced different OCR results; refusing "
            "to report a timing-biased or nondeterministic comparison."
        )

    failures = alignment_failure_lookup(labels_path)
    blank_analysis_by_mode = {
        "fixed": _blank_analysis_lookup(fixed_cells),
        "calibrated": _blank_analysis_lookup(calibrated_cells),
    }
    fixed_rows = _join_results(
        labels=labels,
        results=results_by_mode["fixed"],
        geometry_mode="fixed",
        failures=failures,
        blank_analysis_by_filename=blank_analysis_by_mode["fixed"],
    )
    calibrated_rows = _join_results(
        labels=labels,
        results=results_by_mode["calibrated"],
        geometry_mode="calibrated",
        failures=failures,
        blank_analysis_by_filename=blank_analysis_by_mode["calibrated"],
    )
    hybrid_rows, hybrid_policy_counts = build_hybrid_rows(
        fixed_rows,
        calibrated_rows,
        sheet_alignment_metrics,
    )
    summaries = {
        "fixed": summarize_mode(
            fixed_rows,
            ocr_input_count=timing_by_mode["fixed"]["ocr_input_count"],
            total_ocr_seconds=timing_by_mode["fixed"]["total_ocr_time_seconds"],
            total_processing_seconds=timing_by_mode["fixed"][
                "total_processing_time_seconds"
            ],
        ),
        "calibrated": summarize_mode(
            calibrated_rows,
            ocr_input_count=timing_by_mode["calibrated"]["ocr_input_count"],
            total_ocr_seconds=timing_by_mode["calibrated"][
                "total_ocr_time_seconds"
            ],
            total_processing_seconds=timing_by_mode["calibrated"][
                "total_processing_time_seconds"
            ],
        ),
        "hybrid": summarize_mode(
            hybrid_rows,
            ocr_input_count=(
                timing_by_mode["fixed"]["ocr_input_count"]
                + timing_by_mode["calibrated"]["ocr_input_count"]
            ),
            total_ocr_seconds=(
                timing_by_mode["fixed"]["total_ocr_time_seconds"]
                + timing_by_mode["calibrated"]["total_ocr_time_seconds"]
            ),
            total_processing_seconds=(
                timing_by_mode["fixed"]["total_processing_time_seconds"]
                + timing_by_mode["calibrated"]["total_processing_time_seconds"]
            ),
        ),
    }
    summaries["hybrid"]["policy_counts"] = hybrid_policy_counts
    comparisons = compare_geometry_rows(fixed_rows, calibrated_rows)
    calibrated_safety = assess_calibrated_safety(
        summaries["fixed"], summaries["calibrated"]
    )
    hybrid_safety = assess_hybrid_safety(
        summaries["fixed"], summaries["calibrated"], summaries["hybrid"]
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_folder / "benchmark_results.csv",
        RESULT_FIELDNAMES,
        [*fixed_rows, *calibrated_rows, *hybrid_rows],
    )
    write_csv_rows(
        output_folder / "row_comparison.csv",
        COMPARISON_FIELDNAMES,
        comparisons,
    )
    write_csv_rows(
        output_folder / "blank_analysis_comparison.csv",
        BLANK_COMPARISON_FIELDNAMES,
        blank_comparison_rows,
    )
    (output_folder / "blank_analysis_summary.json").write_text(
        json.dumps(blank_analysis_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "modes": summaries,
        "blank_analysis_before_after": blank_analysis_summary,
        "calibrated_safety_assessment": calibrated_safety,
        "hybrid_production_gates": hybrid_safety,
    }
    (output_folder / "benchmark_metrics.json").write_text(
        json.dumps(metrics_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "labels_path": str(labels_path),
        "output_folder": str(output_folder.resolve()),
        "cell_count_per_mode": len(labels),
        "full_sheet_ocr_was_run": False,
        "model_construction_seconds": round(model_construction_seconds, 6),
        "shape_warmup_seconds_excluded_from_timing": shape_warmup_seconds,
        "geometry_execution_orders": [
            list(order) for order in counterbalanced_execution_orders()
        ],
        "counterbalanced_timing": timing_by_mode,
        "counterbalanced_predictions_consistent": predictions_consistent,
        "ocr_settings": {
            "fixed": fixed_settings,
            "calibrated": calibrated_settings,
            "identical": True,
        },
        "hybrid_policy": {
            "sheet_selection_uses_filename_or_identity": False,
            "sheet_alignment_metrics": sheet_alignment_metrics,
            "counts": hybrid_policy_counts,
            "calibrated_selected_confirmation_required": True,
            "shared_wrong_ocr_value_is_geometry_detectable": False,
        },
        "ground_truth_use": (
            "Labels are joined only after mode predictions complete; they are "
            "never included in OCR input dictionaries or hybrid selection."
        ),
        "crop_alignment_failure_definition": (
            "A preparation-time geometry rejection, or a nonblank ground-truth "
            "cell for which blank detection/OCR produced no numeric text in any "
            "variant."
        ),
        "production_geometry_default_changed": False,
    }
    (output_folder / "benchmark_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"metrics": metrics_report, "run_manifest": run_manifest}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark only after every required label is complete."""

    arguments = build_parser().parse_args(argv)
    report = run_benchmark(arguments.labels, arguments.output)
    print(
        json.dumps(
            {
                "fixed": report["metrics"]["modes"]["fixed"]["overall"],
                "calibrated": report["metrics"]["modes"]["calibrated"][
                    "overall"
                ],
                "hybrid": report["metrics"]["modes"]["hybrid"]["overall"],
                "hybrid_policy_counts": report["metrics"]["modes"]["hybrid"][
                    "policy_counts"
                ],
                "hybrid_production_gates": report["metrics"][
                    "hybrid_production_gates"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
