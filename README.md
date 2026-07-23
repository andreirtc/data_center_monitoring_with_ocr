# Data Center Monthly Monitoring OCR

This OCR-assisted encoding application converts photographed or scanned Toyota
Data Center Monthly Monitoring Sheets into human-verified temperature and
humidity records and exports them into the official Excel template.

The system is designed as a human-in-the-loop workflow. OCR produces proposed
readings, while uncertain, malformed, anomalous, or inconsistent readings are
shown with their extracted handwritten crops for verification.

## Main features

- Detects and straightens a photographed monitoring table.
- Detects the complete printed Day 1-31 row span before building either grid.
- Shows straight fixed and locally calibrated extraction previews before OCR.
- Keeps fixed extraction as the conservative initial choice.
- Extracts 496 measurement cells: 31 days x 8 points x 2 readings.
- Detects blank cells before OCR.
- Runs three image variants through PaddleOCR.
- Records OCR agreement and confidence.
- Validates format, absolute limits, blank consistency, and anomalies.
- Provides a primary 31-day verification workspace with all eight points and
  exact handwritten crops for the selected day.
- Provides secondary image-assisted table editing and detailed cell review.
- Applies partial, filename-keyed updates without rerunning OCR.
- Exports into a copy of the official Excel template while preserving its
  formatting, formulas, borders, logos, and print layout.

## Technology

- Python 3.13
- Streamlit for the user interface
- OpenCV for table detection, perspective correction, preprocessing, cropping,
  and blank detection
- PaddleOCR and PaddlePaddle for pretrained text recognition
- NumPy and Pandas for image and tabular data handling
- Python ZIP and XML processing for safe Excel-template updates

## OCR model and training

The project does **not** train or fine-tune a custom OCR neural network.

It integrates PaddleOCR's pretrained `TextRecognition` model and runs it on the
CPU. The custom work is the domain-specific pipeline around the model:

1. Detect and straighten the fixed company form.
2. Extract each known temperature and humidity cell.
3. Detect blanks before recognition.
4. Create original, grayscale, and contrast-enhanced OCR inputs.
5. Enlarge and pad each crop before batch recognition.
6. Select a consensus result from the three OCR variants.
7. Apply conservative numeric postprocessing and verification rules.
8. Require human confirmation when a result is uncertain or ambiguous.

Manually labeled sample cells and the scripts in `scripts/` can be used to
evaluate and calibrate preprocessing, thresholds, and business rules. This is
evaluation and calibration, not model-weight training.

## Processing workflow

```text
Upload image
    -> detect the complete 32-boundary Day 1-31 row sequence
    -> prepare straight fixed and locally calibrated previews without OCR
    -> inspect overlays, representative crops, metrics, and warnings
    -> explicitly choose fixed or locally calibrated extraction
    -> run OCR
    -> extract 496 cells using the chosen geometry
    -> classify blank cells
    -> preprocess nonblank crops
    -> run pretrained PaddleOCR
    -> select consensus predictions
    -> verify and flag readings
    -> verify each day and apply sparse filename-keyed corrections
    -> export the official Excel workbook
```

## Project structure

```text
streamlit_app.py        Streamlit entry point and session orchestration
datacenter_ocr/         Reusable OCR, image, verification, update, and export logic
templates/              Official immutable Excel template
tests/                  Verification, state-update, UI-contract, and export tests
scripts/                Development, benchmark, and diagnostic command-line tools
test_images/            Local sample photographs; ignored by Git
local_benchmark/        Local labels and evaluation artifacts; ignored by Git
output/                 Generated crops and reports; ignored by Git
requirements.txt        Reproducible Python dependency versions
```

Only `streamlit_app.py` is the web application. Files under `datacenter_ocr/`
are supporting modules, and files under `scripts/` are development utilities.

## Installation

From PowerShell in the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first PaddleOCR model initialization may download pretrained model files.
Later runs reuse PaddleOCR's local model cache.

## Running the application

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

Then open the local URL shown by Streamlit, normally
`http://localhost:8501`.

For best results, upload a clear photograph containing the complete table with
minimal glare, blur, shadow, and perspective distortion.

## Verification behavior

- Final numeric values must contain exactly one decimal place.
- Temperature must be between 10.0 and 50.0.
- Humidity must be between 0.0 and 100.0.
- Low-confidence or materially disagreeing OCR variants require confirmation.
- Ambiguous text such as `22.` is not silently repaired.
- A temperature/humidity blank mismatch remains blocking.
- Operational temperature warnings remain visible but do not automatically
  block export.
- Export remains blocked while invalid values or required confirmations remain.
- Export also remains blocked until all 31 applicable days are explicitly
  confirmed.
- Operational warnings remain visible but do not independently block day
  confirmation or export.

## Day Verification workflow

After OCR, **Day Verification** is the primary workspace. Select a day to see
Points 1 through 8 in company-form order. Every row shows temperature and
humidity crop thumbnails, editable values, blank controls, compact status, and
the reason attention is required. Crop lookups use the exact stable OCR-result
filename, and each thumbnail can be enlarged.

Pressing Enter inside the day form submits its first action, **Confirm Day and
Next**. A successful confirmation saves only that day's controls, confirms all
16 valid readings, and advances to the next unconfirmed day. **Save Day** keeps
partial corrections without confirming the day. Editing a confirmed day later
through any interface invalidates that day's confirmation.

Full Monitoring Table, Detailed Review, Sheet Previews, and Export Excel remain
secondary tabs. All edits update the same canonical `CellOCRResult` objects
through the filename-keyed sparse patch engine and do not rerun OCR.

Day confirmations and unfinished corrections persist only in the active
Streamlit session. Closing the browser session or stopping Streamlit clears
unfinished work; this stage intentionally adds no database.

## Running checks

```powershell
.\.venv\Scripts\python.exe -m compileall datacenter_ocr streamlit_app.py tests
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
git diff --check
```

## Geometry-only grid diagnostics

The Stage 1 alignment diagnostic prepares and straightens a sheet, extracts the
existing fixed 496-cell layout, and measures nearby printed grid lines. It does
not import, initialize, or run PaddleOCR, and its uncalibrated score does not
affect OCR or export behavior.

```powershell
.\.venv\Scripts\python.exe -B scripts\diagnose_grid_alignment.py `
    --image test_images\sample.png `
    --output output\grid_diagnostic\sample
```

Each run writes the standardized table, current grid overlay, detected-line
overlay, the fixed days 1/16/31 by points 1/4/8 contact sheet, and an alignment
JSON report. Reference coordinates remain the safe fallback. When at least 30
of the 32 horizontal boundaries form a strong complete-sheet sequence, both
extraction modes use that detected top/bottom span. Fixed mode keeps straight,
evenly spaced rows; calibrated mode may additionally follow bounded local
curvature.

## Fixed-versus-calibrated OCR benchmark

Stage 3A prepares a local, ignored 54-cell benchmark across the sample, April
2026, and May 2026 sheets. Preparation writes wider context, fixed crops,
calibrated crops, a labeling contact sheet, and `labels.csv` without importing
or running PaddleOCR:

```powershell
.\.venv\Scripts\python.exe -B scripts\prepare_geometry_ocr_benchmark.py
```

Complete every missing `expected_value` in
`local_benchmark/geometry_ab/labels.csv`. For a genuinely blank cell, leave the
value empty and set `expected_blank` to `true`. Do not copy an OCR prediction
into the ground-truth columns.

After all labels are complete, run the controlled A/B benchmark explicitly:

```powershell
.\.venv\Scripts\python.exe -B scripts\run_geometry_ocr_benchmark.py `
    --labels local_benchmark\geometry_ab\labels.csv `
    --output local_benchmark\geometry_ab\results
```

The runner refuses incomplete or invalid labels before PaddleOCR is imported.
It constructs the model once, measures first-inference warm-up separately, and
uses the same production blank detection, three preprocessing variants,
batching, postprocessing, and verification for both crop modes. Benchmark
reports are evidence only and do not change the fixed production default.

Stage 3B evaluates blankness on an aspect-preserving `112 x 40` analysis
canvas, with the same interpolation, border exclusion, component filtering,
and unchanged ink threshold for both crop geometries. Run its labeled
before/after blank check without importing PaddleOCR:

```powershell
.\.venv\Scripts\python.exe -B scripts\evaluate_geometry_blank_analysis.py `
    --labels local_benchmark\geometry_ab\labels.csv `
    --output local_benchmark\geometry_ab\stage3b_blank_analysis
```

The 54-cell OCR runner now warms both crop shapes and times fixed and
calibrated modes in both execution orders. It also derives a benchmark-only
hybrid: stable low-drift sheets retain fixed crops, materially drifted sheets
select calibrated crops, and calibrated selections or cross-mode
disagreements remain confirmation-required. The hybrid is not used by
Streamlit, does not select using ground truth, and does not change the fixed
production geometry default.

The Streamlit pipeline records observational stage timings and inference
counters in the processing-results diagnostics expander. The development
full-sheet runner also writes `processing_metrics.json` and a complete
machine-readable `cell_results.csv` under `output/full_sheet_ocr/`.

## Local files and OCR speed

The following folders serve different purposes:

| Folder | Purpose | Needed for normal app execution? |
| --- | --- | --- |
| `.venv/` | Installed Python packages | Yes, unless another environment is used |
| PaddleOCR model cache | Downloaded pretrained model weights | Yes; stored outside this repository by PaddleOCR/PaddleX |
| `output/` | Generated crops and diagnostic reports | No |
| `test_images/` | Local sample sheets for testing and demonstrations | No |
| `local_benchmark/` | Labels, prepared inputs, and evaluation reports | No |
| `learning_exercises/` | Early learning and prototype code | No |
| `__pycache__/` | Automatically generated Python bytecode | No; regenerated automatically |

Images under `output/`, `test_images/`, and `local_benchmark/` do not make live
OCR faster. They are useful for development, evaluation, troubleshooting, and
demonstrations. Runtime speed mainly benefits from keeping the virtual
environment installed, retaining PaddleOCR's model cache, and reusing the
model loaded by Streamlit during the active session.

## Excel export

The application writes into a copy of
`templates/Toyota_Data_Center_Temperature_Monitoring_Template.xlsx`. It updates
the required worksheet XML cells while preserving the rest of the workbook.
The original template is never modified.

## Limitations

- Handwriting recognition is not guaranteed to be 100% accurate.
- Poor lighting, cropped borders, severe blur, or strong perspective can reduce
  table detection and OCR quality.
- The extraction geometry is intentionally specialized for the supported
  Toyota monitoring-sheet layout.
- Accuracy claims should be based on a manually labeled evaluation set rather
  than individual demonstrations.
