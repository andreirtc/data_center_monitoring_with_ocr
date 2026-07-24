from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


class StreamlitDayVerificationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(source)

    def _day_form(self) -> ast.With:
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                context = item.context_expr
                if (
                    not isinstance(context, ast.Call)
                    or _call_name(context) != "form"
                    or not context.args
                    or not isinstance(context.args[0], ast.Name)
                    or context.args[0].id != "day_form_key"
                ):
                    continue
                return node
        self.fail("Day Verification form not found.")

    def test_enter_submits_confirm_day_and_next_first(self) -> None:
        day_form = self._day_form()
        context = day_form.items[0].context_expr
        self.assertIsInstance(context, ast.Call)
        enter_keyword = next(
            keyword
            for keyword in context.keywords
            if keyword.arg == "enter_to_submit"
        )
        self.assertIsInstance(enter_keyword.value, ast.Constant)
        self.assertTrue(enter_keyword.value.value)

        form_calls = sorted(
            [
            call
            for call in ast.walk(day_form)
            if isinstance(call, ast.Call)
            and _call_name(call) == "form_submit_button"
            ],
            key=lambda call: call.lineno,
        )
        self.assertEqual(3, len(form_calls))
        self.assertIsInstance(form_calls[0].args[0], ast.Constant)
        self.assertEqual(
            "Confirm Day and Next",
            form_calls[0].args[0].value,
        )
        labels = [call.args[0].value for call in form_calls]
        self.assertEqual(1, labels.count("Confirm Day and Next"))
        self.assertEqual(1, labels.count("Save Day"))
        self.assertEqual(1, labels.count("Previous Day"))

    def test_day_actions_are_rendered_once_below_the_grid(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertNotIn("top_controls", source)
        self.assertIn("action_columns = st.columns", source)
        self.assertLess(
            source.index("for point in range(1, 9):"),
            source.index("action_columns = st.columns"),
        )

    def test_day_form_has_no_action_dropdown(self) -> None:
        day_form = self._day_form()
        selectbox_labels = [
            call.args[0].value
            for call in ast.walk(day_form)
            if isinstance(call, ast.Call)
            and _call_name(call) == "selectbox"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ]
        self.assertNotIn("Action", selectbox_labels)

    def test_navigation_and_day_callbacks_do_not_call_ocr(self) -> None:
        callback_names = {
            "activate_queue_sheet",
            "lock_orientation_for_preflight",
            "select_adjacent_queue_sheet",
            "submit_day_callback",
            "previous_day_callback",
        }
        callbacks = {
            node.name: node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name in callback_names
        }
        self.assertEqual(callback_names, set(callbacks))
        for callback in callbacks.values():
            called_names = {
                _call_name(call)
                for call in ast.walk(callback)
                if isinstance(call, ast.Call)
            }
            self.assertNotIn(
                "process_measurement_cells_with_blank_detection",
                called_names,
            )

    def test_day_verification_is_the_first_results_tab(self) -> None:
        tabs_call = next(
            call
            for call in ast.walk(self.tree)
            if isinstance(call, ast.Call) and _call_name(call) == "tabs"
        )
        self.assertIsInstance(tabs_call.args[0], ast.List)
        labels = [item.value for item in tabs_call.args[0].elts]
        self.assertEqual("Day Verification", labels[0])
        self.assertIn("Full Monitoring Table", labels)
        self.assertNotIn("Detailed Review", labels)
        self.assertIn("Export Excel", labels)

    def test_detailed_ocr_data_is_demoted_to_troubleshooting(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn(
            "Advanced OCR details for troubleshooting",
            source,
        )
        self.assertNotIn('"Save detailed review"', source)

    def test_full_sheet_uses_adaptive_review_required_ocr(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn('recognition_strategy="adaptive"', source)

    def test_ocr_is_submitted_to_single_background_worker(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        worker_source = (
            PROJECT_ROOT / "datacenter_ocr" / "background_ocr.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"Queue OCR with {geometry_choice} extraction"', source)
        self.assertIn("OCRJobRequest(", source)
        self.assertNotIn(
            "process_measurement_cells_with_blank_detection(",
            source,
        )
        self.assertIn("ThreadPoolExecutor(", worker_source)
        self.assertIn("max_workers=1", worker_source)
        self.assertNotIn("streamlit", worker_source)

    def test_background_status_refreshes_only_the_completed_active_sheet(
        self,
    ) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertIn('@st.fragment(run_every="2s")', source)
        self.assertIn('"Waiting for OCR"', source)
        self.assertIn('"OCR running ·', source)
        self.assertIn("collect_completed_background_jobs(", source)
        self.assertIn("snapshot.sheet_id == active_sheet_id", source)
        self.assertIn('st.rerun(scope="app")', source)
        self.assertIn(
            '"sheet when you are ready to load its proposals."',
            source,
        )

    def test_completed_sheet_switch_skips_heavy_preflight_by_default(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertIn("processed_result_available = (", source)
        self.assertIn(
            '"Show extraction diagnostics or OCR replacement controls"',
            source,
        )
        self.assertIn(
            "if show_extraction_diagnostics and (",
            source,
        )
        self.assertIn(
            "key == \"prepared_sheet\"",
            source,
        )

    def test_uploaded_document_image_is_rendered_lazily(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertIn(
            'uploaded_document_expander = st.expander(',
            source,
        )
        self.assertIn(
            "if uploaded_document_expander.open:",
            source,
        )
        self.assertIn(
            'on_change="rerun"',
            source,
        )

    def test_large_result_previews_render_only_on_the_selected_tab(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertIn(
            'key=f"results_tab_{uploaded_fingerprint}"',
            source,
        )
        self.assertIn("if preview_tab.open:", source)
        self.assertIn(
            'key=f"result_grid_overlay_{uploaded_fingerprint}"',
            source,
        )

    def test_preflight_summary_optional_keys_are_hot_reload_safe(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn('alignment_summary.get("warnings", ())', source)
        self.assertIn('alignment_summary.get("notices", ())', source)

    def test_uploader_accepts_pdf_and_enables_sheet_queue(self) -> None:
        uploader = next(
            call
            for call in ast.walk(self.tree)
            if isinstance(call, ast.Call)
            and _call_name(call) == "file_uploader"
        )
        type_keyword = next(
            keyword for keyword in uploader.keywords if keyword.arg == "type"
        )
        self.assertIsInstance(type_keyword.value, ast.List)
        accepted_types = [item.value for item in type_keyword.value.elts]
        self.assertIn("pdf", accepted_types)
        multiple_keyword = next(
            keyword
            for keyword in uploader.keywords
            if keyword.arg == "accept_multiple_files"
        )
        self.assertIsInstance(multiple_keyword.value, ast.Constant)
        self.assertTrue(multiple_keyword.value.value)
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("Every PDF page", source)
        self.assertIn("navigation never starts OCR", source)
        self.assertNotIn('"Process all"', source)

    def test_portrait_orientation_is_visible_and_locked_before_geometry(
        self,
    ) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertIn('"Page orientation"', source)
        self.assertIn('"Auto (rotate left)"', source)
        self.assertIn('"Rotate right"', source)
        self.assertIn("orientation_preflight_locked", source)
        self.assertIn(
            "Automatically rotated this portrait scan",
            source,
        )


if __name__ == "__main__":
    unittest.main()
