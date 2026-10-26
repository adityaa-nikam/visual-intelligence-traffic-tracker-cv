"""Compare an annotated frame CSV with a pipeline boxes_*.csv export.

This is an operational-threshold evaluation (precision/recall/F1), not mAP.
See MODEL_AND_EVALUATION.md for the annotation schema and limitations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

BOX = ("x1", "y1", "x2", "y2")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def evaluate(gt: pd.DataFrame, pred: pd.DataFrame, threshold: float,
             annotated_frames: set[int] | None = None) -> dict:
    required = {"frame", "class", *BOX}
    for name, table in (("ground truth", gt), ("predictions", pred)):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")
    if "confidence" not in pred:
        pred = pred.assign(confidence=1.0)

    gt = gt.copy()
    pred = pred.copy()
    gt["frame"] = gt["frame"].astype(int)
    pred["frame"] = pred["frame"].astype(int)
    if annotated_frames is None:
        annotated_frames = set(gt["frame"].unique())
    gt = gt[gt["frame"].isin(annotated_frames)].copy()
    # The prediction export covers the full video; detections outside the
    # annotated sample must not be counted as false positives.
    pred = pred[pred["frame"].isin(annotated_frames)].copy()
    gt["class"] = gt["class"].astype(str).str.lower()
    pred["class"] = pred["class"].astype(str).str.lower()

    classes = sorted(set(gt["class"]) | set(pred["class"]))
    class_results = {}
    matched_identity_pairs: list[tuple[str, str]] = []
    total_matches = 0
    for cls in classes:
        class_gt = gt[gt["class"] == cls]
        class_pred = pred[pred["class"] == cls].sort_values("confidence", ascending=False)
        used: set[int] = set()
        tp = fp = 0
        for pi, p in class_pred.iterrows():
            candidates = class_gt[class_gt["frame"] == int(p["frame"])]
            best_idx, best_iou = None, 0.0
            pb = p[list(BOX)].to_numpy(dtype=float)
            for gi, g in candidates.iterrows():
                if gi in used:
                    continue
                score = iou(pb, g[list(BOX)].to_numpy(dtype=float))
                if score > best_iou:
                    best_idx, best_iou = gi, score
            if best_idx is not None and best_iou >= threshold:
                used.add(best_idx)
                tp += 1
                total_matches += 1
                if "object_id" in gt and "track_id" in pred:
                    matched_identity_pairs.append((str(gt.at[best_idx, "object_id"]),
                                                   str(p["track_id"])))
            else:
                fp += 1
        fn = len(class_gt) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_results[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision + recall else 0.0,
        }

    tp = sum(v["tp"] for v in class_results.values())
    fp = sum(v["fp"] for v in class_results.values())
    fn = sum(v["fn"] for v in class_results.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result = {
        "iou_threshold": threshold,
        "frames_annotated": int(len(annotated_frames)),
        "ground_truth_objects": int(len(gt)),
        "predicted_objects": int(len(pred)),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        "by_class": class_results,
        "metric_note": "Single confidence operating point; not mAP or a benchmark score.",
    }

    if "object_id" in gt and "track_id" in pred:
        gt_ids = sorted(gt["object_id"].dropna().astype(str).unique())
        pred_ids = sorted(pred["track_id"].dropna().astype(str).unique())
        counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
        gi = {v: i for i, v in enumerate(gt_ids)}
        pi = {v: i for i, v in enumerate(pred_ids)}
        for a, b in matched_identity_pairs:
            if a in gi and b in pi:
                counts[gi[a], pi[b]] += 1
        idtp = int(counts[linear_sum_assignment(-counts)].sum()) if counts.size else 0
        idfp, idfn = len(pred) - idtp, len(gt) - idtp
        result["IDF1"] = round(2 * idtp / max(2 * idtp + idfp + idfn, 1), 4)
        result["identity_note"] = "IDF1 from global identity matching of IoU-matched detections."
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ground_truth", help="CSV: frame,class,x1,y1,x2,y2[,object_id]")
    ap.add_argument("predictions", help="boxes_*.csv produced by run_pipeline.py")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU match threshold (default: 0.5)")
    ap.add_argument("--frames", help="optional CSV with a `frame` column; include empty frames")
    ap.add_argument("--output", help="optional JSON output path")
    args = ap.parse_args()
    frames = None
    if args.frames:
        frame_table = pd.read_csv(args.frames)
        if "frame" not in frame_table:
            raise ValueError("Frame manifest CSV must contain a `frame` column.")
        frames = set(frame_table["frame"].astype(int))
    report = evaluate(pd.read_csv(args.ground_truth), pd.read_csv(args.predictions), args.iou, frames)
    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
