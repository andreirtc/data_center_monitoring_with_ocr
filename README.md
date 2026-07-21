# Data Center Monthly Monitoring OCR

This application converts photographed or scanned Toyota Data Center Monthly
Monitoring Sheets into reviewed temperature and humidity records and exports
them into the official Excel template.

The system is designed as a human-in-the-loop workflow. OCR produces proposed
readings, while uncertain, malformed, anomalous, or inconsistent readings are
shown with their extracted handwritten crops for verification.

## Main features

- Detects and straightens a photographed monitoring table.
- Extracts 496 measurement cells: 31 days x 8 points x 2 readings.
- Detects blank cells before OCR.
- Runs three image variants through PaddleOCR.
- Records OCR agreement and confidence.
- Validates format, absolute limits, blank consistency, and anomalies.
- Provides image-assisted table editing and manual review.
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
    -> detect and straighten the table
    -> extract 496 cells
    -> classify blank cells
    -> preprocess nonblank crops
    -> run pretrained PaddleOCR
    -> select consensus predictions
    -> verify and flag readings
    -> human review and sparse corrections
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

## Running checks

```powershell
.\.venv\Scripts\python.exe -m compileall datacenter_ocr streamlit_app.py tests
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
git diff --check
```

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
