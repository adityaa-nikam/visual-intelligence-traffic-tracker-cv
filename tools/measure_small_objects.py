"""
Measure what raising inference resolution and lowering confidence actually does
for SMALL objects, instead of assuming it helps.

Aerial footage is a small-object problem: at 70 m a motorcycle is a handful of
pixels. The textbook responses are (a) infer at higher resolution and (b) lower
the confidence floor. Both also invite false positives, so "we tuned it" is not a
result - the result is the measured change, broken down in a way that shows
whether the extra detections are plausible.

Three pieces of evidence are reported per setting:

1. detections per frame, per class
2. the SIZE distribution of detections - if a setting only adds large boxes it did
   nothing for small objects, whatever the headline count says
3. what fraction of the extra detections fall INSIDE the road mask. Real missed
   vehicles are on the carriageway; hallucinations are scattered over rooftops and
   vegetation. This reuses the segmentation from pipeline/road_mask.py, so the two
   features validate each other.

There is no hand-labelled ground truth for this clip, so this script does NOT
report recall or precision. It reports the delta and the evidence about its
plausibility, which is what is honestly available.

Usage:
    python tools/measure_small_objects.py --frames 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import road_mask as rm
from pipeline.detector import COCO_TO_CLASS

# Bucket edges in pixel area. A car at this altitude is roughly 60x110 px = 6600,
# so anything under ~1600 px is a motorcycle, pedestrian or a distant vehicle.
BUCKETS = [(0, 400, "tiny"), (400, 1600, "small"),
           (1600, 6400, "medium"), (6400, 10**9, "large")]


def bucket(area: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= area < hi:
            return name
    return "large"


def run_setting(model, frames, imgsz, conf, classes, iou):
    """Detect on a fixed list of frames and return one row per detection."""
    rows = []
    for fi, img in frames:
        res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou,
                            classes=classes, verbose=False)
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cid = int(b.cls[0])
                rows.append({
                    "frame": fi,
                    "class": COCO_TO_CLASS.get(cid, str(cid)),
                    "conf": float(b.conf[0]),
                    "x": (x1 + x2) / 2, "y": y2,
                    "area": max(x2 - x1, 0) * max(y2 - y1, 0),
                })
    d = pd.DataFrame(rows)
    if not d.empty:
        d["size"] = d["area"].map(bucket)
    return d


def summarise(d: pd.DataFrame, n_frames: int) -> dict:
    if d.empty:
        return {"detections": 0}
    return {
        "detections": int(len(d)),
        "per_frame": round(len(d) / n_frames, 1),
        "by_class": d["class"].value_counts().to_dict(),
        "by_size": d["size"].value_counts().to_dict(),
        "median_conf": round(float(d["conf"].median()), 3),
        "median_area_px": int(d["area"].median()),
        # Called out separately because two-wheelers are the class that visibly
        # went missing, and they are the smallest motor vehicle in the scene.
        "two_wheelers": int(d["class"].isin(["motorcycle", "bicycle"]).sum()),
        "two_wheeler_median_area_px": (
            int(d.loc[d["class"].isin(["motorcycle", "bicycle"]), "area"].median())
            if d["class"].isin(["motorcycle", "bicycle"]).any() else None
        ),
    }


def on_road_fraction(d: pd.DataFrame, mask: np.ndarray) -> float | None:
    """Fraction of detections whose road-contact point is on the carriageway."""
    if d.empty or mask is None:
        return None
    h, w = mask.shape[:2]
    xi = np.clip(d["x"].to_numpy(), 0, w - 1).astype(int)
    yi = np.clip(d["y"].to_numpy(), 0, h - 1).astype(int)
    return round(float((mask[yi, xi] > 0).mean()), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--frames", type=int, default=40, help="frames sampled across the clip")
    ap.add_argument("--out", default="outputs/small_object_study.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    video = cfg["video"]["path"]
    dcfg = cfg["detector"]

    # Sample frames evenly across the clip so the comparison is not biased toward
    # one traffic state (an empty 5 seconds would flatter the low-conf setting).
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), args.frames).astype(int)
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if ok:
            frames.append((int(fi), img))
    cap.release()
    print(f"sampled {len(frames)} frames from {total} ({video})")

    from ultralytics import YOLO
    model = YOLO(dcfg["weights"])

    # Road mask for the plausibility check, from the first sampled frame.
    mask = None
    if frames and cfg.get("segmentation", {}).get("enabled", False):
        tp = os.path.join(cfg["output"]["dir"], "trajectories.csv")
        bp = os.path.join(cfg["output"]["dir"], "boxes.csv")
        if os.path.exists(tp):
            obs = pd.read_csv(tp)
            boxes = pd.read_csv(bp) if os.path.exists(bp) else None
            mask, _ = rm.build_road_mask(obs, frames[0][1], cfg, boxes=boxes,
                                        frame_index=frames[0][0])
            print(f"road mask ready ({(mask > 0).mean():.1%} of frame)")

    # 1920 is the NATIVE frame width. 2560 and 3200 deliberately infer ABOVE
    # native: YOLO's smallest detection head has a stride of 8, so a motorcycle
    # that is 22 px wide at native scale occupies under 3 cells of that head and
    # is effectively invisible to it. Upsampling the input does not add
    # information, but it does give the network more cells per object, which is
    # the specific reason it recovers small objects. Cost is the trade-off.
    settings = {
        "baseline_1280_c25": dict(imgsz=1280, conf=0.25),
        "native_1920_c25": dict(imgsz=1920, conf=0.25),
        "upscale_2560_c25": dict(imgsz=2560, conf=0.25),
        "upscale_3200_c25": dict(imgsz=3200, conf=0.25),
        "native_1920_c15": dict(imgsz=1920, conf=0.15),
    }

    results, dets = {}, {}
    for name, s in settings.items():
        print(f"running {name} ...", flush=True)
        d = run_setting(model, frames, s["imgsz"], s["conf"],
                        dcfg["classes"], dcfg["iou"])
        dets[name] = d
        results[name] = {**s, **summarise(d, len(frames)),
                         "on_road_fraction": on_road_fraction(d, mask)}
        print(f"   {results[name]['detections']} detections, "
              f"{results[name].get('by_size', {})}")

    base = dets["baseline_1280_c25"]
    deltas = {}
    for name, d in dets.items():
        if name == "baseline_1280_c25" or d.empty or base.empty:
            continue
        extra = len(d) - len(base)
        # Which size classes the extra detections landed in. This is the number
        # that decides whether the change helped SMALL objects specifically.
        by_size = {}
        for _, _, sz in BUCKETS:
            b = int((base["size"] == sz).sum())
            n = int((d["size"] == sz).sum())
            by_size[sz] = {"baseline": b, "tuned": n, "delta": n - b}
        by_class = {}
        for c in sorted(set(base["class"]) | set(d["class"])):
            b = int((base["class"] == c).sum())
            n = int((d["class"] == c).sum())
            by_class[c] = {"baseline": b, "tuned": n, "delta": n - b}
        deltas[name] = {
            "extra_detections": extra,
            "extra_pct": round(100.0 * extra / max(len(base), 1), 1),
            "by_size": by_size,
            "by_class": by_class,
        }

    out = {
        "frames_sampled": len(frames),
        "video": video,
        "size_buckets_px": {n: [lo, hi] for lo, hi, n in BUCKETS},
        "settings": results,
        "deltas_vs_baseline": deltas,
        "caveats": [
            "No hand-labelled ground truth exists for this clip, so recall and "
            "precision are NOT reported - only the measured change in detections.",
            "A higher detection count is not automatically better: lower confidence "
            "buys recall with false positives.",
            "on_road_fraction is the plausibility check. Real missed road users are "
            "on the carriageway; false positives scatter over rooftops and vegetation.",
        ],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n" + json.dumps(deltas, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
