"""
Detection: Ultralytics YOLO wrapper plus dataset-aware class mapping.

Uploaded aerial video uses the VisDrone-trained YOLOv8n checkpoint. COCO mapping
is retained for older runs and explicitly configured generic models; class IDs are
never assumed to have the same meaning across model families.

Honesty note on classes
-----------------------
The project vocabulary is car / LGV / HGV / bus / truck / motorcycle / bicycle /
pedestrian. VisDrone supplies aerial labels including car, van, truck, bus and
motor; vans map to LGV and three-wheel categories are grouped with motorcycles.
HGV remains a size-based refinement of truck, only attempted when calibrated
geometry is available and always labeled as a heuristic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from ultralytics import YOLO

# COCO id -> canonical project label, for legacy/generic model checkpoints.
COCO_TO_CLASS = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# VisDrone aerial model ids -> the dashboard's canonical vocabulary. Van is
# kept distinct as LGV; the dataset's three-wheeler categories are grouped with
# motorcycles because this pipeline has no separate tricycle analytics class.
VISDRONE_TO_CLASS = {
    0: "pedestrian", 1: "pedestrian", 2: "bicycle", 3: "car",
    4: "lgv", 5: "truck", 6: "motorcycle", 7: "motorcycle",
    8: "bus", 9: "motorcycle",
}

# Everything that is not a pedestrian/bicycle is treated as a motor vehicle for
# counting and queueing purposes.
MOTOR_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "lgv", "hgv"}

# Stable per-class BGR colours for the overlay.
CLASS_COLORS = {
    "car": (80, 200, 120),
    "motorcycle": (255, 190, 60),
    "bus": (240, 120, 60),
    "truck": (200, 90, 220),
    "lgv": (200, 90, 220),
    "hgv": (150, 60, 190),
    "bicycle": (100, 220, 240),
    "pedestrian": (90, 130, 255),
}


def resolve_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def load_detector(cfg: dict):
    """Load the YOLO model and report the resolved runtime settings."""
    dcfg = cfg["detector"]
    device = resolve_device(dcfg.get("device", "auto"))
    model = YOLO(dcfg["weights"])
    raw_names = model.names
    names = {int(k): str(v).lower() for k, v in raw_names.items()} if isinstance(raw_names, dict) else {i: str(v).lower() for i, v in enumerate(raw_names)}
    if set(names.values()) >= {"pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"}:
        class_map = VISDRONE_TO_CLASS
    else:
        class_map = COCO_TO_CLASS

    # Two-tier confidence. YOLO is run at `conf` (the permissive floor) so that
    # small road users are proposed at all; `strict_conf` then filters the large,
    # easy classes back to a high bar, and `class_conf` overrides per class.
    # Measured on this clip: a single 0.25 floor found 13 two-wheelers across 24
    # frames, a single 0.15 floor found 259 but dropped on-road plausibility from
    # 0.80 to 0.65 by admitting rooftop clutter as cars.
    strict = float(dcfg.get("strict_conf", 0.25))
    class_conf = {c: strict for c in ("car", "bus", "truck")}
    class_conf.update({c: float(dcfg.get("small_conf", 0.15))
                       for c in ("motorcycle", "bicycle", "pedestrian")})
    class_conf.update(dcfg.get("class_conf", {}) or {})

    on_gpu = device not in ("cpu",)
    weights_path = Path(dcfg["weights"])
    weights_sha256 = None
    if weights_path.is_file():
        digest = hashlib.sha256()
        with weights_path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        weights_sha256 = digest.hexdigest()
    settings = {
        "weights": dcfg["weights"],
        "weights_sha256": weights_sha256,
        "imgsz": dcfg.get("imgsz", 1280),
        "conf": dcfg.get("conf", 0.25),
        "iou": dcfg.get("iou", 0.6),
        "classes": dcfg.get("classes", sorted(class_map)),
        "class_map": class_map,
        "model_names": names,
        "device": device,
        "strict_conf": strict,
        "class_conf": class_conf,
        # fp16 only helps on GPU; on CPU it is slower or unsupported.
        "half": bool(dcfg.get("half", True)) and on_gpu,
    }
    return model, settings
