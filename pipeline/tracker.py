"""
Multi-object tracking: ByteTrack over YOLO detections.

Ultralytics ships ByteTrack, so `model.track(persist=True)` gives us a tuned,
well-tested implementation for free - no reason to hand-roll one in a hackathon.
The parameters that actually matter for this footage live in
`config/bytetrack.yaml`; the important one is a long `track_buffer` so an ID
survives occlusion (trees, buses, overpasses) and long stationary dwell at a
red light instead of being reborn as a new ID.

This module yields plain per-frame records. It deliberately knows nothing about
trajectories or analytics - that separation is what lets later levels add new
analytics without touching detection or tracking.
"""

from __future__ import annotations

from typing import Iterator

import cv2

from pipeline.detector import COCO_TO_CLASS


def iter_tracked_frames(
    model,
    video_path: str,
    det_settings: dict,
    tracker_cfg: str,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[dict]:
    """
    Run detection + tracking over a video.

    Yields one dict per processed frame:
        {
          "frame": int,            # index in the source video
          "t": float,              # seconds from clip start
          "image": np.ndarray,     # BGR frame
          "tracks": [ {track_id, cls, conf, xyxy}, ... ]
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = -1
    processed = 0

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_stride > 1 and frame_idx % frame_stride:
                continue
            if max_frames is not None and processed >= max_frames:
                break

            # `half` is only passed when enabled: newer Ultralytics warns on the
            # key regardless of value, which would spam the log every frame.
            kw = dict(
                persist=True,               # keep tracker state across calls
                tracker=tracker_cfg,
                imgsz=det_settings["imgsz"],
                conf=det_settings["conf"],
                iou=det_settings["iou"],
                classes=det_settings["classes"],
                device=det_settings["device"],
                verbose=False,
            )
            if det_settings.get("half"):
                kw["half"] = True
            results = model.track(image, **kw)

            tracks = []
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                clss = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.float().cpu().tolist()
                xyxy = boxes.xyxy.float().cpu().numpy()
                for tid, c, cf, box in zip(ids, clss, confs, xyxy):
                    tracks.append(
                        {
                            "track_id": int(tid),
                            "cls": COCO_TO_CLASS.get(int(c), str(int(c))),
                            "conf": float(cf),
                            "xyxy": box,
                        }
                    )

            yield {
                "frame": frame_idx,
                "t": frame_idx / fps,
                "image": image,
                "tracks": tracks,
            }
            processed += 1
    finally:
        cap.release()


def video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info
