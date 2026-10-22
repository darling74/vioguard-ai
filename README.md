# VIOGUARD AI - Video Violence Detection

A Flask web application that analyses an uploaded video with a **pretrained X3D-M 3D-CNN** and reports
**VIOLENCE DETECTED / NON-VIOLENCE DETECTED**, a violence probability, a per-segment timeline, processing time and more.

> **Read this first - accuracy.** The selected pretrained model reports **F1 = 0.93** on the RWF-2000 *validation*
> split (published by its authors; not an independent test). **A 95 % accuracy has NOT been demonstrated** for this
> project. Run `python evaluate.py` on your own labelled videos to measure the real number (see
> [docs/ACCURACY_REPORT.md](docs/ACCURACY_REPORT.md)). The app never invents or hardcodes a result.

| | |
|---|---|
| Model | X3D-M (`final/final_x3d_realtime.pt`, ~12 MB) from [visionlab-ai/school-violence-detection-models](https://huggingface.co/visionlab-ai/school-violence-detection-models) |
| Input | 16 frames per window, 224x224, RGB, mean 0.45 / std 0.225 |
| Decision | P(violence) >= 0.40 (the authors' recalibrated threshold) |
| Licence | HF tag says MIT, README says research / non-commercial -> **use for research/education only** |
| Python | **3.12** recommended (3.10 and 3.11 should also work) |

---

## 1. Requirements

* Windows 10/11 with **Python 3.12** ([python.org](https://www.python.org/downloads/) - tick *"Add python.exe to PATH"*)
* **VS Code** with the *Python* extension
* ~2 GB free disk space (PyTorch) and an internet connection for the one-time install + 12 MB model download
* No GPU needed (CPU works; an NVIDIA GPU is optional)

## 2. Open the project in VS Code

1. Unzip `AI_Violence_Detection_Improved.zip` (for example to `C:\Projects\AI_Violence_Detection_Improved`).
2. VS Code -> **File -> Open Folder...** -> choose the unzipped folder.
3. Open the terminal: **Terminal -> New Terminal** (it should be **PowerShell**).

## 3. Create and activate a virtual environment (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of the prompt. If PowerShell says *"running scripts is disabled"*, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

In VS Code press **Ctrl+Shift+P -> "Python: Select Interpreter"** and pick the `.venv` one.

## 4. Install the requirements

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Download the pretrained model (automatic)

```powershell
python download_model.py
```

This downloads `final_x3d_realtime.pt` (12,268,364 bytes) from Hugging Face into `models\` and **verifies its size and
SHA-256** (`e833f69d...f0349`). Nothing is faked: if the download fails you get an explanation and the web app shows
"model unavailable" instead of making up predictions.

Then check that everything works together:

```powershell
python check_setup.py
```

## 6. Run the app

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Stop the server with **Ctrl+C**.

## 7. Using the app

1. Drag a video onto the upload area or click **Select video** (MP4, AVI or MOV, up to 200 MB).
2. Check the preview, then press **Analyze video**.
3. Watch the real progress bar (upload -> frame extraction -> per-segment analysis).
4. Read the result:
   * **Red "VIOLENCE DETECTED"** or **green "NON-VIOLENCE DETECTED"**.
   * **Violence probability** - the model's softmax output for the strongest window (aggregated, see below).
     The white tick on the gauge is the decision threshold (40 %).
   * **Confidence** - probability of the *predicted* class. **Result strength** says whether the score is
     borderline / moderate / strong relative to the threshold. It is *not* a calibrated guarantee.
   * **Segment timeline** - one block per analysed window; red blocks are at/above threshold. Click a block to jump there.
   * Processing time, duration, segments/frames analysed, resolution and FPS.
5. **Analyze another video** resets the page. Uploaded files are deleted from the server as soon as analysis ends.

### How a video is analysed

* Videos up to 7.5 s are analysed as one window (like the model's 5 s training clips).
* Longer videos are split into **5 s windows with 50 % overlap** (at most 24; very long videos are spread evenly).
* Each window: 16 evenly spaced frames -> resize 224x224 -> RGB -> `(x/255 - 0.45) / 0.225` -> X3D-M -> softmax -> P(violence).
* Window scores are aggregated (`AGGREGATION` in `.env`): `max` (default, the authors' rule), `mean`, `topk`, or `consecutive`.
  `max` catches short incidents but can raise false alarms on long videos; `evaluate.py` can tell you which mode and
  threshold work best on *your* data.

### API (for scripts)

| Method | URL | Purpose |
|---|---|---|
| GET | `/api/health` | model state, limits |
| POST | `/api/analyze` (field `video`) | start a job -> `{job_id}` |
| GET | `/api/jobs/<id>` | progress and result |
| POST | `/predict` (field `video`) | synchronous, same field name as the original project |

```powershell
curl.exe -F "video=@C:\path\clip.mp4" http://127.0.0.1:5000/predict
```

## 8. Run the automated tests

```powershell
pip install -r requirements-dev.txt
pytest -v
```

(or without pytest: `python -m unittest discover -s tests -t .`). Tests that need PyTorch (random-weight loader tests)
run only when PyTorch is installed. **Passing tests proves the software works, not that the model is accurate.**

## 9. Measure the real accuracy (recommended)

Prepare a folder with `Violence/` and `NonViolence/` sub-folders (for example the Kaggle
[Real Life Violence Situations dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- a *different* dataset from the model's RWF-2000 training data), then:

```powershell
python evaluate.py --data "C:\datasets\Real Life Violence Dataset"
```

It splits videos into disjoint validation/test sets, picks aggregation + threshold on validation only, and writes
`outputs\evaluation_report.md` with accuracy, precision, recall, F1, confusion matrix, FP/FN counts and a 95 % confidence interval.
Try `--limit 40` first for a quick trial; results are cached in `outputs\eval_cache.json`.

## 10. Configuration

Copy `.env.example` to `.env` and edit (all optional): port, upload limit, `DEVICE=auto|cpu|cuda`, `VIOLENCE_THRESHOLD`,
`AGGREGATION`, logging... Debug mode is **off** by default; the app is served with Waitress, which also works on Windows.

### Optional: NVIDIA GPU

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```
`DEVICE=auto` then uses CUDA automatically. CPU is already fast for this small model.

## 11. Fixing common errors

| Message / symptom | Fix |
|---|---|
| `py` is not recognised | Reinstall Python and tick *Add to PATH*, or use `python -m venv .venv` |
| *running scripts is disabled* | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `pip install` fails on `pytorchvideo` / `av` | Upgrade pip (`python -m pip install --upgrade pip`). Use Python 3.12 64-bit. If it still fails, install its helpers first (`pip install av fvcore iopath parameterized networkx`), then `pip install pytorchvideo==0.1.5 --no-deps` |
| `ModuleNotFoundError: torchvision.transforms.functional_tensor` | Handled automatically by the app. If you see it elsewhere, run `python check_setup.py` |
| Download error / HTTP 403 / timeout | `huggingface.co` is blocked or offline. Try another network/VPN, set `HTTPS_PROXY`, or set `HF_ENDPOINT` to a mirror, then rerun `python download_model.py` |
| `SHA-256 mismatch` / wrong size | Partial download. `python download_model.py --force` |
| Web page says *Model unavailable* | Run `python check_setup.py` - it names the missing piece |
| *No frame could be decoded* | The file is corrupted or uses an unsupported codec. Re-export it as H.264 MP4 |
| Port 5000 already in use | Set `PORT=5001` in `.env` and open `http://127.0.0.1:5001` |
| Very slow on long videos | Decoding dominates; lower `MAX_VIDEO_SECONDS` or trim the video. Each window costs about the same |

If something still fails, run `python check_setup.py` and send its output together with `logs\app.log`.

## 12. Project structure

```
app.py                 Flask app factory, routes, error handling
config.py              settings from .env + models/model_config.json
violence_model.py      cached X3D-M loader, preprocessing, inference
video_processor.py     decoding, window planning, frame sampling, aggregation, analyzer
jobs.py                background jobs with real progress + file cleanup
download_model.py      verified weight download (size + SHA-256)
check_setup.py         installation diagnostics
evaluate.py            leakage-aware accuracy evaluation
models/model_config.json   model facts, preprocessing, threshold, source, hash
templates/ static/     dashboard UI (HTML, CSS, JS)
tests/                 automated tests
docs/                  accuracy report, model verification, test report
```

## 13. Known limitations

* **Accuracy is unverified here** (see top). The authors' F1 0.93 is on data used for model selection.
* Trained on RWF-2000 (YouTube fights). Contact sports, crowds, low light or CCTV angles can cause errors.
* Binary only: it does not say *what* violence occurred or *who* is involved.
* Audio is ignored. Only 16 sampled frames per window are inspected, so extremely brief events can be missed.
* Licence ambiguity: treat the weights as research/non-commercial.
* Never use the output as the sole basis for decisions about people.

## 14. Credits

Model: *School Violence Detection: A Comparative Study of 3D CNN Architectures* (Nguyen, UIT-VNU-HCM, 2026) -
`visionlab-ai/school-violence-detection-models`. Architecture: X3D (Feichtenhofer, CVPR 2020) via PyTorchVideo.
Original project: Mandeep Kharb (MoBiLSTM, Kaggle RLVS dataset).


## VIOGUARD AI interface changes
- Simplified the page to focus on video upload, analysis progress, and results.
- Updated the visible product name to **VIOGUARD AI**.
- Preserved the result video, violence probability, confidence, duration, processing time, frame/segment counts, resolution/FPS, and segment timeline.
- Added a short warning tone when the model classifies a video as violent. Some browsers block automatic sound after background processing; the result page then offers **Play warning sound**.
- The warning is an alert for human review, not proof that violence occurred.
