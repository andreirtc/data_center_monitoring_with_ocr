from __future__ import annotations

from dataclasses import replace
from threading import Lock
import time
import unittest

import numpy as np

from datacenter_ocr.background_ocr import (
    BackgroundOCRManager,
    OCRJobRequest,
)
from datacenter_ocr.processing_metrics import ProcessingMetrics
from datacenter_ocr.sheet_processing import PreparedMonitoringSheet
from datacenter_ocr.sheet_queue import archive_prepared_sheet


def archived_sheet(sheet_id: str) -> object:
    preview = np.full((8, 10, 3), 255, dtype=np.uint8)
    crop = np.full((3, 4, 3), 255, dtype=np.uint8)
    return archive_prepared_sheet(
        PreparedMonitoringSheet(
            detection_preview=preview,
            warped_table=preview.copy(),
            measurement_grid_overlay=preview.copy(),
            measurement_boxes=[],
            cells=[
                {
                    "filename": f"{sheet_id}.png",
                    "day": 1,
                    "point": 1,
                    "reading_type": "temperature",
                    "image": crop,
                }
            ],
        )
    )


def request(owner_id: str, sheet_id: str) -> OCRJobRequest:
    return OCRJobRequest(
        owner_id=owner_id,
        sheet_id=sheet_id,
        source_filename=f"{sheet_id}.png",
        geometry_mode="calibrated",
        prepared_sheet=archived_sheet(sheet_id),
        processing_metrics=replace(
            ProcessingMetrics(),
            source_filename=f"{sheet_id}.png",
            uploaded_fingerprint=sheet_id,
        ),
        cells_per_batch=2,
    )


def wait_for_terminal(
    manager: BackgroundOCRManager,
    job_id: str,
    owner_id: str,
) -> str:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(job_id, owner_id)
        if snapshot is not None and snapshot.state in ("succeeded", "failed"):
            return snapshot.state
        time.sleep(0.005)
    raise AssertionError(f"Job {job_id} did not finish.")


class BackgroundOCRManagerTests(unittest.TestCase):
    def test_jobs_run_sequentially_and_keep_sheet_identity(self) -> None:
        concurrency_lock = Lock()
        running = 0
        maximum_running = 0
        execution_order: list[str] = []
        model_construction_count = 0

        def fake_processor(**kwargs: object) -> list:
            nonlocal running, maximum_running
            cells = kwargs["cells"]
            progress = kwargs["progress_callback"]
            sheet_id = cells[0]["filename"].removesuffix(".png")
            with concurrency_lock:
                running += 1
                maximum_running = max(maximum_running, running)
                execution_order.append(sheet_id)
            progress(1, 1)
            time.sleep(0.03)
            with concurrency_lock:
                running -= 1
            return []

        def model_factory() -> object:
            nonlocal model_construction_count
            model_construction_count += 1
            return object()

        manager = BackgroundOCRManager(
            model_factory=model_factory,
            cell_processor=fake_processor,
            row_builder=lambda results: [],
        )
        try:
            first = manager.submit(request("owner", "sheet-1"))
            second = manager.submit(request("owner", "sheet-2"))

            self.assertEqual("succeeded", wait_for_terminal(manager, first, "owner"))
            self.assertEqual(
                "succeeded",
                wait_for_terminal(manager, second, "owner"),
            )
            self.assertEqual(["sheet-1", "sheet-2"], execution_order)
            self.assertEqual(1, maximum_running)
            first_result = manager.consume_result(first, "owner")
            second_result = manager.consume_result(second, "owner")
            self.assertEqual(1, model_construction_count)
            self.assertFalse(
                first_result.processing_metrics.model_was_warm
            )
            self.assertTrue(
                second_result.processing_metrics.model_was_warm
            )
            self.assertEqual(
                "sheet-2",
                second_result.sheet_id,
            )
        finally:
            manager.shutdown()

    def test_duplicate_active_submission_returns_existing_job(self) -> None:
        def fake_processor(**kwargs: object) -> list:
            time.sleep(0.03)
            return []

        manager = BackgroundOCRManager(
            model_factory=lambda: object(),
            cell_processor=fake_processor,
            row_builder=lambda results: [],
        )
        try:
            first = manager.submit(request("owner", "sheet-1"))
            duplicate = manager.submit(request("owner", "sheet-1"))

            self.assertEqual(first, duplicate)
        finally:
            manager.shutdown()

    def test_owner_cannot_inspect_or_consume_another_sessions_job(self) -> None:
        manager = BackgroundOCRManager(
            model_factory=lambda: object(),
            cell_processor=lambda **kwargs: [],
            row_builder=lambda results: [],
        )
        try:
            job_id = manager.submit(request("owner-a", "sheet-1"))
            self.assertEqual(
                "succeeded",
                wait_for_terminal(manager, job_id, "owner-a"),
            )

            self.assertIsNone(manager.snapshot(job_id, "owner-b"))
            self.assertIsNone(manager.consume_result(job_id, "owner-b"))
            self.assertIsNotNone(manager.consume_result(job_id, "owner-a"))
        finally:
            manager.shutdown()

    def test_failure_is_reported_and_can_be_discarded_for_retry(self) -> None:
        def failing_processor(**kwargs: object) -> list:
            raise ValueError("synthetic OCR failure")

        manager = BackgroundOCRManager(
            model_factory=lambda: object(),
            cell_processor=failing_processor,
            row_builder=lambda results: [],
        )
        try:
            job_id = manager.submit(request("owner", "sheet-1"))

            self.assertEqual(
                "failed",
                wait_for_terminal(manager, job_id, "owner"),
            )
            snapshot = manager.snapshot(job_id, "owner")
            self.assertIn("synthetic OCR failure", snapshot.error_message)
            self.assertTrue(manager.discard_terminal_job(job_id, "owner"))
        finally:
            manager.shutdown()


if __name__ == "__main__":
    unittest.main()
