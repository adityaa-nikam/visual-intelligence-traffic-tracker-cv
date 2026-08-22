"""
Detection: YOLO wrapper plus the COCO -> challenge class mapping.

We use pretrained Ultralytics YOLOv8 rather than training a custom model. At the
altitude these clips were flown (~70 m) road users are large enough in the frame
that the COCO detector transfers well, and a 4-hour budget does not allow for
labelling an aerial dataset.

Honesty note on classes
-----------------------
The challenge vocabulary is car / LGV / HGV / bus / truck / motorcycle /
pedestrian. COCO gives us: person, bicycle, car, motorcycle, bus, truck. There
is **no LGV/HGV distinction in the model**. We therefore keep the model's own
`truck` label as the detection class, and split LGV vs HGV separately in the
analytics layer using the vehicle's measured ground footprint (only possible
because telemetry gives us metric calibration). That split is labelled as a
size heuristic everywhere it appears - it is never presented as a model output.
"""

from __future__ import annotations

import torch
from ultralytics import YOLO

# COCO id -> challenge-facing name.
COCO_TO_CLASS = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
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

    on_gpu = device not in ("cpu",)
    settings = {
        "weights": dcfg["weights"],
        "imgsz": dcfg.get("imgsz", 1280),
        "conf": dcfg.get("conf", 0.25),
        "iou": dcfg.get("iou", 0.6),
        "classes": dcfg.get("classes", sorted(COCO_TO_CLASS)),
        "device": device,
        # fp16 only helps on GPU; on CPU it is slower or unsupported.
        "half": bool(dcfg.get("half", True)) and on_gpu,
    }
    return model, settings
