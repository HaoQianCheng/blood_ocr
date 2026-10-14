[中文](README.md) | [English](README_EN.md)

# blood_ocr

An experimental OCR project for blood-bag label images, with PC inference using RapidOCR/ONNX, RKNN inference on RK3576 boards, and an image-upload page with an HTTP API. It also includes image-orientation correction, assisted blood-type annotation, OCR previews, and model-conversion utilities.

Inference returns text, confidence scores, and text boxes. The project does not implement a complete structured label parser or medical decision workflow. Results require human verification and must not be used as the basis for clinical or transfusion decisions. This guide is based on source inspection; dependency installation, inference, and board execution have not been tested for this documentation.

## Execution modes

| Mode | Entry point | Environment | Output |
| --- | --- | --- | --- |
| PC OCR | `run.py` | Python + `rapidocr_onnxruntime` | Terminal text and confidence |
| Board OCR | `rknn_run.py` | RK3576 + RKNNLite | Terminal text and confidence |
| Board Web/API | `rknn_server.py` | Board environment plus Flask | Upload page and JSON results |

The PC entry point calls `RapidOCR()` and does not use the repository's `.rknn` models. Board entry points use `rknn_core/` to load `models/det.rknn`, `cls.rknn`, and `rec.rknn` for text detection, orientation classification, and recognition.

## Repository layout

```text
blood_ocr/
├── run.py                    # PC RapidOCR entry point
├── rknn_run.py               # Board CLI
├── rknn_server.py            # Board Flask service
├── requirements.txt         # Existing Paddle list; mismatched with the PC entry point
├── requirements-rknn.txt    # Board dependencies; Flask is missing
├── rknn_core/               # RKNNLite engine and det/cls/rec processing
├── models/                  # Three RKNN models
├── templates/index.html     # Image-upload test page
├── export/                  # ONNX inspection, dictionary extraction, RKNN conversion
│   └── ppocr_keys_v1.txt     # Tracked recognition dictionary
├── train/                   # Data preparation, manual annotation, preview tools
└── output/                  # Tracked cropped images
```

`test_images/` and `data/` are not included in the current source tree. Supply test images you are authorized to use, with identifying information removed. Label images from the repository are not reproduced here.

## PC: RapidOCR / ONNX

### Environment

Python, pip, and virtual-environment support are required. The repository does not declare or pin a compatible PC Python/RapidOCR combination. The following is an installation starting point based on imports, not a tested lockfile.

```bash
git clone https://github.com/HaoQianCheng/blood_ocr.git
cd blood_ocr
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

```bash
python -m pip install rapidocr-onnxruntime Pillow
```

Important: the existing `requirements.txt` lists only `paddlepaddle==2.6.2`, `paddleocr==2.9.1`, and `opencv-python-headless`. It omits `rapidocr_onnxruntime`, which is imported by `run.py` and the `train/` scripts. Running only `pip install -r requirements.txt` does not satisfy the current PC entry point. This documentation does not change that file or claim the suggested installation has been tested.

### Run

Run from the repository root, substituting your image paths:

```bash
python run.py /path/to/image.jpg
python run.py /path/to/images
```

Without an argument, the script reads `test_images/`. Directory mode scans only the immediate directory and selects `.jpg`, `.jpeg`, `.png`, and `.bmp` files, case-insensitively. Results are printed to the terminal; JSON and annotated images are not saved by this entry point. A message is printed when no text is detected.

## RK3576: board CLI and Web/API

### Environment and models

Comments in `requirements-rknn.txt` specify RK3576 and Python 3.10; it pins `rknn-toolkit-lite2==2.3.2`. The board OS and NPU driver must be compatible with the models and runtime. An ordinary PC environment is not a substitute for board inference. A complete system image and driver-installation procedure are not supplied.

In a separate Python environment on the board:

```bash
python3 -m pip install -r requirements-rknn.txt
# Needed only for Web/API mode; absent from the existing dependency list
python3 -m pip install Flask
```

### Adjust hard-coded paths first

Both `rknn_run.py` and `rknn_server.py` contain these fixed paths. Check and edit each entry point before launching it:

```python
MODEL_DIR = "/root/blood_ocr_rknn/models"
DICT_PATH = "/root/blood_ocr_rknn/rknn_core/ppocr_keys_v1.txt"
```

- `MODEL_DIR` must point to the directory containing all three `.rknn` files.
- The tracked dictionary is `export/ppocr_keys_v1.txt`, not a file under `rknn_core/`. For a checkout at `/root/blood_ocr_rknn`, change `DICT_PATH` in both entry points to `/root/blood_ocr_rknn/export/ppocr_keys_v1.txt`.
- For any other checkout location, replace both paths with the actual absolute paths. Model paths are not exposed as command-line options.

### Command line

```bash
python3 rknn_run.py /path/to/image.jpg
python3 rknn_run.py /path/to/images
```

The default directory, nonrecursive scan, and extension filtering match the PC entry point. Unreadable images are skipped; model resources are released after normal completion.

### Web service and API

```bash
python3 rknn_server.py
```

The service binds to `0.0.0.0:8000`. Open `http://<board-ip>:8000/` for the upload page. Models are loaded when the service module is imported, so invalid model/dictionary paths or runtime problems prevent startup.

Call the API with a test image containing no personal information:

```bash
curl -X POST http://127.0.0.1:8000/api/ocr \
  -F "image=@/path/to/image.jpg"
```

Use `multipart/form-data` with a file field named `image`. A successful response contains:

- `ok`: `true`
- `elapsed_ms`: OCR call time in milliseconds, excluding upload and image decoding
- `results`: an array of objects with `text`, `confidence`, and `box` (four points in original-image coordinates)

If no text is recognized, `results` is empty. A missing `image` field or an undecodable image returns HTTP 400 with `ok: false` and `error`.

The current service uses Flask's development server and implements no authentication, TLS, upload-size limit, or production deployment configuration. It binds to all network interfaces by default. Use it only on a trusted test network and add access controls and security configuration before deployment. Do not upload identifying or sensitive images to an unprotected service.

## Data utilities (PC)

These tools use RapidOCR and Pillow and are intended to be run in the PC environment:

```bash
python train/fix_orientation.py /path/to/images data/aligned
python train/preview_ocr.py data/aligned
python train/label.py data/aligned data/labels.txt
```

- `fix_orientation.py`: tries 0/90/180/270-degree rotations and chooses an orientation using OCR and keyword scoring. Defaults to `output/` → `data/aligned/`.
- `preview_ocr.py`: draws boxes, text, and confidence scores into the fixed output directory `data/preview/`. Defaults to reading `output/`. It first tries the `msyh.ttc` font; other systems may require an available Chinese font instead.
- `label.py`: opens images in an image viewer and prompts for confirmation or correction of A/B/O/AB labels. Writes `filename<TAB>blood_type`. A graphical image viewer is required. Saving rewrites the label file; quitting early can omit existing labels for files not yet visited, so back up the file first.

The `train/` directory currently contains data utilities, with no model-training entry point or training configuration.

## Model inspection and conversion (optional)

RKNN models are already included; conversion is not a prerequisite for board inference. To rebuild them, prepare a compatible Linux/WSL conversion environment, RapidOCR ONNX models, `onnx`, and RKNN-Toolkit2 providing `rknn.api`. Toolkit2 serves a different purpose from the board's RKNN-Toolkit-Lite2.

- `export/inspect_onnx.py`: inspects model inputs/outputs and extracts the dictionary from recognition-model metadata. Edit `MODELS_DIR` and `DICT_OUT` first.
- `export/onnx2rknn.py`: edit `ONNX_DIR` and `OUT_DIR` first. The current target is `rk3576`, quantization is disabled, detection uses four configured dynamic input shapes, and classification/recognition use fixed shapes.
- `export/download_rknn.py`: downloads a specific Toolkit2 2.3.2 wheel for CPython 3.10/x86_64 to `/tmp/rknn_toolkit2.whl`; it does not install it. Check link availability, wheel naming, and platform compatibility separately.

The conversion scripts contain absolute paths from the author's machine. After preparing the environment and replacing those paths, run `python3 export/inspect_onnx.py` and `python3 export/onnx2rknn.py` from the repository root. Conversion and generated-model compatibility have not been validated here.

## Known limitations

- Dependency lists have the omissions and entry-point mismatch described above; there is no unified, verified installation procedure.
- The RKNN pipeline uses a fixed recognition-confidence threshold of `0.6` and batch=1 classification/recognition per crop. Its outputs are not guaranteed to match PC RapidOCR.
- No automated tests, accuracy evaluation, or reliable performance benchmarks are supplied. Confidence scores do not establish medical correctness.
- Paths, fonts, and Windows cuDNN path handling in the utilities assume parts of a local setup and require review for cross-platform use.
- No license file is supplied in the current repository. Verify usage and redistribution rights for code, models, and data separately.
