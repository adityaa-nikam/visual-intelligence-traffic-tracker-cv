from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from pipeline.calibration import build_projector
from pipeline.interactions import find_interactions
from pipeline.trajectory import compute_kinematics
from tools.evaluate_labeled_run import evaluate


class ProjectFixTests(unittest.TestCase):
    def test_conflicts_are_unavailable_without_metric_calibration(self):
        obs = pd.DataFrame([{"track_id": 1, "frame": 0}])
        self.assertTrue(find_interactions(obs, {}, calibrated=False).empty)

    def test_evaluation_ignores_unannotated_video_frames(self):
        gt = pd.DataFrame([{
            "frame": 5, "class": "car", "x1": 10, "y1": 10,
            "x2": 30, "y2": 30, "object_id": "car-a",
        }])
        pred = pd.DataFrame([
            {"frame": 5, "class": "car", "x1": 10, "y1": 10,
             "x2": 30, "y2": 30, "track_id": 9, "confidence": 0.9},
            {"frame": 100, "class": "car", "x1": 10, "y1": 10,
             "x2": 30, "y2": 30, "track_id": 10, "confidence": 0.9},
        ])
        result = evaluate(gt, pred, 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["IDF1"], 1.0)

    def test_calibration_uses_clip_timestamp_window(self):
        cues = []
        times = [0.0] + [1.0 + i * 0.05 for i in range(10)] + [2.0]
        for frame, sec in enumerate(times, start=1):
            millis = int(round(sec * 1000))
            timestamp = f"00:00:{millis // 1000:02d},{millis % 1000:03d}"
            yaw = 45 if sec == 2.0 else 0
            cues.append(
                f"{frame}\n{timestamp} --> {timestamp}\n"
                f"FrameCnt: {frame} focal_len: 24.00 latitude: 18.5 longitude: 73.7 "
                f"rel_alt: 70.0 gb_yaw: {yaw} gb_pitch: -90.0 gb_roll: 0.0 "
                "dzoom_ratio: 1.0"
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.srt"
            path.write_text("\n\n".join(cues), encoding="utf-8")
            cfg = {
                "telemetry": {"srt_path": str(path), "sensor_width_35mm": 36.0},
                "video": {"clip_start_s": 1.0},
            }
            projector, info = build_projector(cfg, 1920, 1080, clip_duration_s=0.5)
        self.assertIsNotNone(projector)
        self.assertEqual(info["records"], 10)
        self.assertEqual(info["gimbal_yaw_deg"], 0.0)

    def test_acceleration_smoothing_does_not_cross_track_boundary(self):
        rows = []
        for tid in (1, 2):
            for frame in range(12):
                t = frame / 10.0
                x = t * t * 20 if tid == 1 else 100 + 5 * t
                rows.append({
                    "track_id": tid, "class": "car", "frame": frame,
                    "timestamp": t, "x": x, "y": 0.0,
                    "world_x": x, "world_y": 0.0, "footprint_m": np.nan,
                })
        cfg = {
            "trajectory": {"smooth_window": 1, "speed_window_s": 0.4, "min_track_frames": 8},
            "video": {"frame_stride": 1},
        }
        result = compute_kinematics(pd.DataFrame(rows), 10.0, cfg, calibrated=True)
        linear = result[result["track_id"] == 2]["accel_kph_s"].dropna()
        self.assertTrue((linear.abs() < 1e-8).all())


if __name__ == "__main__":
    unittest.main()
