# DataCenterMonitorOCR contributor guidance

## Purpose and architecture

This Streamlit application processes photographed or scanned Toyota Data Center Monthly Monitoring Sheets. It detects and straightens the table, extracts 496 temperature/humidity cells (31 days × 8 points × 2 readings), detects blanks, runs multi-variant PaddleOCR, supports human review and table editing, and exports into the official Excel template.

- `streamlit_app.py`: UI and Streamlit session-state orchestration only.
- `datacenter_ocr/`: reusable OCR, image/table processing, verification, record, state-update, and export logic.
- `templates/`: official Excel template; treat it as immutable.
- `tests/`: business-rule, state-update, and export tests.

## Core constraints

- Preserve the working OCR, blank detection, table detection, cell layout, and Excel-template mapping unless the user explicitly requests changes.
- Keep validation and verification business logic centralized in reusable `datacenter_ocr` modules, outside `streamlit_app.py`.
- Treat OCR agreement, confidence, format validity, absolute limits, operational severity, anomaly detection, and human confirmation as separate concepts.
- Distinguish blocking errors, human-confirmation items, operational warnings, and informational notices. Only genuinely unresolved or invalid readings should block export.
- Never silently repair ambiguous OCR text such as `22.`. Preserve it or present a proposed correction and require human confirmation.
- Do not add SQLite or another database.
- Use type hints and clear docstrings for reusable logic.

## Review and Streamlit state

- Review must support partial batch saves: users may save some items while leaving others unresolved.
- The default review interface must be compact, paginated, and practical for many items. A detailed one-item mode may be offered as a secondary view.
- Never render hundreds of expanded review cards simultaneously.
- Use the same stable result identifier—preferably `filename`—for review selection, displayed content, widget keys, and saved updates.
- Preserve corrections, blank confirmations, selection, and review state across Streamlit reruns and pages without rerunning OCR.
- Avoid unnecessary explicit `st.rerun()` calls and double-refresh behavior. Prefer callbacks that update canonical session state before the normal widget rerun.

## Monitoring-table edits

- Saves must be partial: unrelated unresolved or invalid OCR readings must never prevent valid changes elsewhere from being saved.
- Validate only fields the user actually changed.
- When a reading is marked blank, clear or ignore its numeric value and skip numeric validation for it.
- A row with both temperature and humidity confirmed blank is a resolved blank row.
- If only one reading is blank, flag a blank-state mismatch.
- Failed validation must leave the displayed editor consistent with the saved backend state.
- Show compact, row-specific validation errors instead of one large global error message.
- Keep `CellOCRResult`, monitoring rows, Review items, table status, and export data synchronized through one canonical update path.

## Excel export

- Export into a copy of `templates/Toyota_Data_Center_Temperature_Monitoring_Template.xlsx` while preserving formatting, formulas, borders, logos, and print layout.
- Never modify the original template.
- Keep export blocked while blocking errors or unresolved required confirmations remain.

## Working practices

- Before completion, run `python -m compileall` and the relevant tests.
- Show the user a concise summary and diff for review.
- Never commit, push, reset, revert, restore, stash, discard, or overwrite user changes without explicit approval.
- Ask before installing dependencies or making broad architectural changes.