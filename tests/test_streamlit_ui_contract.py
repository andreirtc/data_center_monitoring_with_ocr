from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


class StreamlitReviewContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(source)

    def test_quick_form_enter_submits_its_only_submit_button(self) -> None:
        quick_forms: list[ast.With] = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                context = item.context_expr
                if not isinstance(context, ast.Call) or _call_name(context) != "form":
                    continue
                enter_keyword = next(
                    (
                        keyword
                        for keyword in context.keywords
                        if keyword.arg == "enter_to_submit"
                    ),
                    None,
                )
                if (
                    enter_keyword is not None
                    and isinstance(enter_keyword.value, ast.Constant)
                    and enter_keyword.value.value is True
                ):
                    quick_forms.append(node)

        self.assertEqual(1, len(quick_forms))
        form_calls = [
            call
            for call in ast.walk(quick_forms[0])
            if isinstance(call, ast.Call)
            and _call_name(call) == "form_submit_button"
        ]
        self.assertEqual(1, len(form_calls))
        self.assertIsInstance(form_calls[0].args[0], ast.Constant)
        self.assertEqual(
            "Save reviewed items on this page",
            form_calls[0].args[0].value,
        )

    def test_quick_form_has_no_action_dropdown(self) -> None:
        quick_form = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and _call_name(item.context_expr) == "form"
                and any(
                    keyword.arg == "enter_to_submit"
                    for keyword in item.context_expr.keywords
                )
                for item in node.items
            )
        )
        selectbox_labels = [
            call.args[0].value
            for call in ast.walk(quick_form)
            if isinstance(call, ast.Call)
            and _call_name(call) == "selectbox"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ]
        self.assertNotIn("Action", selectbox_labels)


if __name__ == "__main__":
    unittest.main()
