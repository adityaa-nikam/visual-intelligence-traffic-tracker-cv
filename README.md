# Drone Traffic Intelligence

**An end-to-end computer vision learning project for detecting, tracking, and analyzing road users in aerial video.**

The pipeline turns drone footage into persistent object tracks, road-user trajectories, and traffic summaries. When matching DJI telemetry is available and passes stability checks, it can also estimate motion in ground-plane units. Without valid telemetry, it stays in pixel space instead of inventing metres or km/h.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-vision%20pipeline-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8-7A3EF0)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

> **Project status:** educational prototype. Detection and traffic analytics have not yet been benchmarked against a representative, hand-labeled test set. This is not a certified traffic-safety or real-time monitoring system.

## Why this project

This project explores the full path from video pixels to useful, reviewable measurements—not just running an object detector. It combines aerial object detection, multi-object tracking, camera geometry, telemetry parsing, trajectory processing, and a browser-based results dashboard.

It is designed to make its assumptions visible: class provenance is retained, metric calibration is gated on telemetry quality, and physical conflict analysis is withheld when the scene is uncalibrated.

## What it does

- **Detects aerial road users** with a YOLOv8n checkpoint fine-tuned on VisDrone.
- **Tracks detections across frames** with ByteTrack to produce persistent track IDs.
- **Builds trajectories** from tracked observations and derives speed, heading, dwell, and movement features.
- **Uses DJI SRT telemetry optionally** to estimate a camera pose and project image points onto a flat ground plane when configured stability checks pass.
- **Can estimate a road mask** for a fixed or hovering camera when the optional segmentation setting is enabled.
- **Summarizes traffic** with counts, queues, congestion indicators, turning movements, directional flow, headways, and selected anomaly candidates.
- **Renders results** as annotated video, trajectory visualizations, downloadable tables, and an interactive Streamlit dashboard.
- **Degrades honestly without telemetry:** reports pixel-space motion and marks metric-only analyses unavailable.

## Demo

The dashboard accepts a traffic video and an optional matching DJI `.SRT` telemetry file. It displays processing progress and opens the generated run when analysis finishes.

For a reliable first test, use a short, low-resolution aerial clip. The CPU-only setup can be slow on longer videos. If the video is a cut from a longer recording, provide the clip's start time so telemetry timestamps can be aligned.

## Architecture

```text
Video ──> YOLOv8 aerial detector ──> ByteTrack ──> tracked observations
                                                       │
DJI SRT ──> telemetry checks ──> ground-plane projection (optional)
                                                       │
                                                       v
                                                trajectories.csv
                                                       │
                     ┌─────────────────────────────────┼──────────────────────┐
                     v                                 v                      v
              traffic analytics                interaction analysis     event candidates
                     └─────────────────────────────────┼──────────────────────┘
                                                       v
                                  annotated video + tables + Streamlit dashboard
```

`trajectories.csv` is the interface between computer vision and downstream analytics. Detection and tracking run once; analytics operate on the resulting trajectory data. This keeps the expensive vision stage separate from analysis logic and makes it easier to inspect or retune results.

## Tech stack

| Area | Tools | Role |
|---|---|---|
| Language | Python | Pipeline orchestration and analytics |
| Detection | PyTorch, Ultralytics YOLOv8n | Aerial road-user detections |
| Tracking | ByteTrack | Object association and track IDs over time |
| Video and geometry | OpenCV, NumPy | Frame processing and spatial calculations |
| Analytics | Pandas, SciPy | Trajectory tables, spatial neighborhoods, and summaries |
| Dashboard | Streamlit, Plotly | Upload workflow, run results, and charts |
| Model distribution | Hugging Face Hub | Download a pinned detector checkpoint and verify its checksum |

## Getting started locally

### 1. Requirements

- Python 3.10 or newer (Python 3.11 is a good default)
- FFmpeg available on your PATH for optional video preparation and telemetry extraction
- CPU works for short tests; a CUDA-enabled PyTorch install is recommended for faster local processing

### 2. Install dependencies and the detector

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python tools/download_aerial_model.py
```

The downloader fetches the pinned VisDrone checkpoint, checks its SHA-256 digest, and verifies the expected class names. The weights are intentionally not stored in Git.

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload an aerial traffic video, and optionally attach the matching `.SRT` telemetry file. The dashboard downloads and verifies the model on first launch if it is missing.

To run the pipeline directly instead of using the upload form:

```bash
python run_pipeline.py --video path/to/traffic.mp4 --tag demo
```

With matching DJI telemetry:

```bash
python run_pipeline.py --video path/to/traffic.mp4 --srt path/to/traffic.SRT --tag demo
```

Useful commands:

```bash
python run_pipeline.py --max-frames 60 --tag smoke
python run_pipeline.py --no-video
python -m unittest discover -s tests -v
```

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository. Keep uploaded footage, telemetry, model weights, and generated outputs out of Git; `.gitignore` excludes these local files.
2. At [share.streamlit.io](https://share.streamlit.io), create an app from the repository, select the branch, and set the entrypoint to `app.py`.
3. In the deployment's **Advanced settings → Secrets**, add:

   ```toml
   ISOLATE_SESSION_OUTPUTS = "true"
   ```

4. Deploy and wait for the initial model download. Test using a short clip under the configured 100 MB upload limit.

The app keeps each visitor's run list in a separate temporary session directory when this setting is enabled. Uploaded media and results may be removed when the host restarts, so do not upload private footage. Community Cloud resources are limited; longer video jobs can be slow or run out of memory. Use this deployment as a learning demo, not a production service.

## Evaluation and limitations

Computer vision results should be measured against labeled data. This repository includes an evaluation workflow in [`MODEL_AND_EVALUATION.md`](MODEL_AND_EVALUATION.md) and `tools/evaluate_labeled_run.py`, but no representative labeled benchmark is currently included. Therefore, this README does **not** claim detection mAP, precision/recall, IDF1, or HOTA scores.

Other important limitations:

- The detector can miss tiny, blurred, or occluded objects; tracking cannot recover objects that were never detected.
- The model's VisDrone categories are mapped into the dashboard's road-user vocabulary; some fine-grained vehicle distinctions are heuristic.
- Metric output assumes a flat road plane and depends on correctly matched telemetry, a stable camera pose, and valid camera parameters. It is an estimate, not surveyed ground truth.
- Conflict indicators are analytical candidates under a simplified motion model—not confirmed near misses, collisions, or safety violations.
- The single-frame road-mask option is intended for fixed or hovering-camera footage.

More detail on model provenance, calibration checks, evaluation methodology, and caveats is in [`MODEL_AND_EVALUATION.md`](MODEL_AND_EVALUATION.md).

## What this project demonstrates

- Building a modular vision pipeline instead of treating detection as the whole application.
- Connecting detections to multi-object tracking, then to trajectory-level analytics.
- Handling aerial-specific challenges such as small objects, occlusion, frame sampling, and class mapping.
- Applying camera geometry and telemetry carefully, with explicit validity checks and fallback behavior.
- Designing reproducible model setup, configurable runs, downloadable outputs, and a usable dashboard.
- Separating measured results from assumptions—and identifying where ground-truth evaluation is still needed.

## Repository structure

```text
app.py                         Streamlit upload and results dashboard
run_pipeline.py                End-to-end pipeline orchestration
config/config.yaml             Detector, video, calibration, and analytics settings
config/bytetrack.yaml          ByteTrack tracker configuration
pipeline/detector.py            YOLO model setup and class mapping
pipeline/classify.py            Per-track class refinement helpers
pipeline/tracker.py             Detection-to-track association
pipeline/calibration.py         DJI telemetry and ground-plane projection
pipeline/trajectory.py          Track cleanup and kinematic features
pipeline/analytics.py            Traffic counts, queues, congestion, and movement
pipeline/interactions.py         Pairwise interactions and conflict indicators
pipeline/anomalies.py            Event candidate generation
pipeline/road_mask.py            Optional road-mask estimation
pipeline/visualize.py            Annotated video and trajectory maps
tools/download_aerial_model.py   Pinned model download and verification
tools/evaluate_labeled_run.py    Labeled-run evaluation utility
tests/                           Regression tests
MODEL_AND_EVALUATION.md          Model provenance and evaluation notes
```

## Acknowledgements

The aerial detector is a pinned public YOLOv8n/VisDrone checkpoint; see [`MODEL_AND_EVALUATION.md`](MODEL_AND_EVALUATION.md) for its source, checksum, class mapping, and license note. Ultralytics provides the YOLO implementation and ByteTrack integration. Please review the upstream model and library licenses before redistribution or commercial use.
