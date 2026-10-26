# Detector setup and honest evaluation

## Aerial detector

Video uploads use [`mshamrai/yolov8n-visdrone`](https://huggingface.co/mshamrai/yolov8n-visdrone),
a YOLOv8n checkpoint trained on the VisDrone aerial-object dataset. The downloader
pins revision `28ca510`, verifies SHA-256
`e5f2685462a10a24e05bd6e5db33e11eb46e0113c96b3440db61483d70831a60`, and checks
the checkpoint's class order before it is accepted:

```powershell
pip install -r requirements.txt
python tools/download_aerial_model.py
```

The checkpoint is intentionally not committed to source control. It is about 6 MB
and is reproducibly fetched from the pinned public model repository. Review the
model repository's OpenRAIL license and terms before redistribution or commercial
use.

| VisDrone output | Project label |
|---|---|
| pedestrian, people | pedestrian |
| bicycle | bicycle |
| car | car |
| van | lgv |
| truck | truck, then optionally size-refined when calibrated |
| bus | bus |
| motor, tricycle, awning-tricycle | motorcycle |

Grouping the three-wheel classes with `motorcycle` is a deliberate vocabulary
limitation, not a claim that those vehicle types are identical. Model confidence
and raw box labels are retained in `boxes_<run>.csv` for review.

## Ground-truth evaluation

The project currently has no hand-labeled validation set, so its run counts are
descriptive outputs—not measured accuracy. To evaluate a run, annotate a
representative sample of the *processed frame indices* and save a CSV with:

```text
frame,class,x1,y1,x2,y2,object_id
```

Coordinates are pixels in the original uploaded video. `class` must use the
detector's mapped labels (`car`, `lgv`, `truck`, `bus`, `motorcycle`, `bicycle`,
`pedestrian`). Label vans as `lgv`; label a detector truck as `truck` even if later
size heuristics call its track HGV. Keep each object's `object_id` consistent
between frames. The ID column is optional for detection-only evaluation. By
default only frames present in the label CSV are scored. To include annotated
empty frames (important for measuring false positives), provide a frame manifest
CSV with a `frame` column and use `--frames`.

Run the operational-threshold comparison against the matching pipeline box file:

```powershell
python tools/evaluate_labeled_run.py labels.csv outputs/boxes_<run-tag>.csv --iou 0.5 --frames annotated_frames.csv --output outputs/evaluation_<run-tag>.json
```

It reports per-class and overall precision, recall, F1, and (when IDs are present)
IDF1 using global identity matching of IoU-matched detections. It is **not mAP**:
the pipeline saves boxes after its confidence filters, so this evaluates one
operating point. To report mAP, retain pre-filter model predictions and evaluate
them over a confidence sweep. Label several clips with different viewpoints,
lighting, densities, object sizes, and occlusions before presenting one headline
score.

## Calibration limits

The metric projection uses one representative DJI pose. It is now rejected when
robust telemetry spreads exceed configured gates for altitude, gimbal attitude,
zoom, or horizontal GPS drift, or when the SRT has too few records / insufficient
time coverage of the clip. These checks help catch camera motion; passing them does
not replace surveyed ground-distance validation. Without accepted metric
calibration, speeds remain px/s and physical conflict grading is explicitly
unavailable.
