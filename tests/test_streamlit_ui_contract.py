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
        self.assertEqual(2, len(form_calls))
        self.assertIsInstance(form_calls[0].args[0], ast.Constant)
        self.assertEqual(
            "Confirm Day and Next",
            form_calls[0].args[0].value,
        )
        self.assertEqual("Save Day", form_calls[1].args[0].value)

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
        callback_names = {"submit_day_callback", "previous_day_callback"}
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
        self.assertIn("Detailed Review", labels)
        self.assertIn("Export Excel", labels)

    def test_preflight_summary_optional_keys_are_hot_reload_safe(self) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn('alignment_summary.get("warnings", ())', source)
        self.assertIn('alignment_summary.get("notices", ())', source)


if __name__ == "__main__":
    unittest.main()
