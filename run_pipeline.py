"""
Traffic Intelligence - main pipeline runner.

    drone video
        -> YOLO detection
        -> ByteTrack (stable IDs)
        -> trajectory extraction (metric, via DJI telemetry)
        -> analytics / interactions / anomalies
        -> annotated video + CSVs + summary JSON

Usage:
    python run_pipeline.py                          # uses config/config.yaml
    python run_pipeline.py --max-frames 150         # quick smoke test
    python run_pipeline.py --video data/multi_road.mp4 --srt data/multi_road_full.srt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import analytics, anomalies, classify, interactions, visualize
from pipeline import road_mask as road_mask_mod
from pipeline.calibration import build_projector
from pipeline.detector import load_detector
from pipeline.tracker import iter_tracked_frames, video_info
from pipeline.trajectory import (
    TrajectoryStore,
    apply_track_classes,
    compute_kinematics,
    resolve_track_classes,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Drone traffic intelligence pipeline")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--video", default=None, help="override video path")
    ap.add_argument("--srt", default=None, help="override telemetry SRT path")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="skip annotated video rendering")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument(
        "--in-tag", default=None,
        help="tag to READ existing trajectories/boxes from, when it differs from "
             "--tag. Lets a re-analysis read the untagged run and write a tagged one.",
    )
    ap.add_argument(
        "--overwrite-video", action="store_true",
        help="overwrite the annotated video instead of writing the next _vN version",
    )
    ap.add_argument(
        "--from-trajectories", nargs="?", const="__default__", default=None,
        metavar="CSV",
        help="skip detection+tracking and re-derive all analytics from an existing "
             "trajectories CSV. This is the architectural payoff: detection is the only "
             "expensive stage, so retuning a threshold or adding a new analytic costs "
             "seconds instead of a full re-processing run.",
    )
    args = ap.parse_args()
    in_tag = args.tag if args.in_tag is None else args.in_tag

    cfg = load_config(args.config)
    configured_weights = str(cfg.get("detector", {}).get("weights", ""))
    local_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), configured_weights)
    if not args.from_trajectories and configured_weights.lower().endswith(".onnx") and not os.path.isfile(local_weights):
        local_aerial = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
        if os.path.isfile(local_aerial):
            cfg["detector"]["weights"] = "models/best.pt"
            print("Configured ONNX weights not found; using the installed VisDrone detector")
        else:
            raise SystemExit(
                "Configured detector weights are missing. Install the aerial detector with "
                "`python tools/download_aerial_model.py`."
            )
    if args.video:
        cfg["video"]["path"] = args.video
        # Older saved upload configs may still point at the COCO checkpoint.
        # Switch those upload runs to the aerial model when it is installed.
        if os.path.normcase(os.path.normpath("outputs/uploads")) in os.path.normcase(os.path.normpath(args.video)):
            aerial_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
            if os.path.isfile(aerial_checkpoint):
                cfg["detector"].update({
                    "weights": "models/best.pt", "classes": list(range(10)),
                    "imgsz": 1280, "conf": 0.10, "strict_conf": 0.20,
                    "small_conf": 0.08,
                })
                print("Using the VisDrone aerial detector for uploaded footage")
    if args.srt:
        cfg["telemetry"]["srt_path"] = args.srt
    if args.max_frames is not None:
        cfg["video"]["max_frames"] = args.max_frames
    if args.stride is not None:
        cfg["video"]["frame_stride"] = args.stride
    if args.no_video:
        cfg["output"]["write_video"] = False

    video_path = cfg["video"]["path"]
    stride = max(1, int(cfg["video"].get("frame_stride", 1)))
    max_frames = cfg["video"].get("max_frames")
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    def out(key: str) -> str:
        base = cfg["output"][key]
        if not args.tag:
            return base
        root, ext = os.path.splitext(base)
        return f"{root}_{args.tag}{ext}"

    def in_path(name: str, ext: str = ".csv") -> str:
        """Path of an input side-table, using --in-tag (defaults to --tag)."""
        return os.path.join(out_dir, f"{name}{'_' + in_tag if in_tag else ''}{ext}")

    def out_path(name: str, ext: str = ".csv") -> str:
        """Path of an output side-table, using --tag."""
        return os.path.join(out_dir, f"{name}{'_' + args.tag if args.tag else ''}{ext}")

    def next_video_path(base: str) -> str:
        """
        Never clobber a rendered video: pick the next free `_vN`.

        Renders are the artifact a human actually reviews, so each iteration stays
        on disk side by side and successive versions can be compared directly.
        """
        if args.overwrite_video or not os.path.exists(base):
            return base
        root, ext = os.path.splitext(base)
        n = 2
        while os.path.exists(f"{root}_v{n}{ext}"):
            n += 1
        return f"{root}_v{n}{ext}"

    vinfo = video_info(video_path)
    # PyTorch detectors accept dynamic input shapes. Preserve each video's
    # orientation and aspect ratio instead of forcing portrait uploads into a
    # landscape canvas; the resulting dimensions are rounded to model stride 32.
    if str(cfg.get("detector", {}).get("weights", "")).lower().endswith(".pt"):
        configured_size = cfg["detector"].get("imgsz", 1280)
        max_side = max(configured_size) if isinstance(configured_size, (list, tuple)) else int(configured_size)
        scale = min(1.0, max_side / max(vinfo["width"], vinfo["height"]))
        target_w = max(32, int(round(vinfo["width"] * scale / 32)) * 32)
        target_h = max(32, int(round(vinfo["height"] * scale / 32)) * 32)
        cfg["detector"]["imgsz"] = [target_h, target_w]
        print(f"      adapting detector input to video orientation: {target_h}x{target_w}")
    detector_path = str(cfg.get("detector", {}).get("weights", ""))
    if not args.from_trajectories and detector_path.startswith("models/") and not os.path.isfile(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), detector_path
    )):
        raise SystemExit(
            f"Detector weights not found: {detector_path}. Install them with "
            "`python tools/download_aerial_model.py`."
        )
    fps = vinfo["fps"]
    print(f"[1/6] video {video_path}  {vinfo['width']}x{vinfo['height']} @ {fps:.2f}fps  "
          f"{vinfo['frames']} frames  stride={stride}")

    # ---- Calibration -------------------------------------------------------
    projector, calib_info = build_projector(
        cfg, vinfo["width"], vinfo["height"],
        clip_duration_s=vinfo["frames"] / max(vinfo["fps"], 1e-6),
    )
    calibrated = projector is not None
    if calibrated:
        print(f"      calibrated from DJI telemetry: alt={calib_info['altitude_m']}m  "
              f"pitch={calib_info['gimbal_pitch_deg']}deg  "
              f"{calib_info['metres_per_px_at_centre']} m/px at centre")
    else:
        print(f"      NO metric calibration ({calib_info['reason']}) -> pixel-space analytics only")

    # ---- Detection + tracking ---------------------------------------------
    # (loaded lazily inside the else-branch below: --from-trajectories must not
    #  pay the cost of loading YOLO at all.)

    store = TrajectoryStore(projector)
    box_rows: list[dict] = []
    t0 = time.time()
    n_frames = 0

    reuse = args.from_trajectories
    class_report: dict = {}
    if reuse:
        # Re-analysis path: detection/tracking already happened, so read the saved
        # trajectories and jump straight to analytics.
        traj_path = reuse
        if reuse == "__default__":
            root, ext = os.path.splitext(cfg["output"]["trajectories_csv"])
            traj_path = f"{root}_{in_tag}{ext}" if in_tag else f"{root}{ext}"
        obs = pd.read_csv(traj_path)
        boxes_path = in_path("boxes")
        if not os.path.exists(boxes_path):
            # Fall back to the untagged boxes file: --tag names the OUTPUT, and a
            # re-render usually reads a previous untagged run's boxes.
            alt = os.path.join(out_dir, "boxes.csv")
            if os.path.exists(alt):
                boxes_path = alt
        boxes = pd.read_csv(boxes_path) if os.path.exists(boxes_path) else pd.DataFrame()
        # A saved trajectories.csv has already had its per-frame labels replaced by a
        # resolved per-track class, with the model's own output kept in `class_raw`.
        # Re-classification must vote on the RAW labels; voting on resolved ones just
        # re-derives the previous answer and reports a 0% override rate.
        if "class_raw" in obs.columns:
            obs["class"] = obs["class_raw"]
            obs = obs.drop(columns=["class_raw"] +
                           [c for c in ("class_source",) if c in obs.columns])
            print("      restored raw per-frame labels from class_raw before re-classifying")
        calibrated = "sx" in obs.columns and obs["sx"].notna().any()
        n_frames = int(obs["frame"].nunique())
        elapsed = 1e-6
        print(f"[2/6] REUSING {traj_path}: {len(obs)} observations, "
              f"{obs['track_id'].nunique()} tracks - no detection run")
        if boxes.empty:
            print("      (no boxes.csv -> annotated video cannot be re-rendered)")
            cfg["output"]["write_video"] = False
        print("[3/6] trajectories: reused; re-running classification")
        # Classification is cheap and depends only on geometry, so it is re-run on
        # reuse rather than trusting whatever labels the CSV was saved with. The
        # ground dimensions need `depth_m`, which older runs did not record - it can
        # be recovered from the saved boxes without re-detecting anything, since the
        # projection is a pure function of the box and the (hovering) camera pose.
        if calibrated and projector is not None and not boxes.empty and "depth_m" not in obs:
            bx = boxes.copy()
            cx = (bx["x1"] + bx["x2"]) / 2.0
            probes = np.stack([np.c_[cx, bx["y2"]], np.c_[bx["x1"], bx["y2"]],
                               np.c_[bx["x2"], bx["y2"]], np.c_[cx, bx["y1"]]],
                              axis=1).reshape(-1, 2)
            gp = projector.pixels_to_ground(probes).reshape(-1, 4, 2)
            bx["footprint_m"] = np.linalg.norm(gp[:, 2] - gp[:, 1], axis=1).round(2)
            bx["depth_m"] = np.linalg.norm(gp[:, 3] - gp[:, 0], axis=1).round(2)
            obs = obs.drop(columns=[c for c in ("footprint_m", "depth_m") if c in obs.columns])
            obs = obs.merge(bx[["frame", "track_id", "footprint_m", "depth_m"]],
                            on=["frame", "track_id"], how="left")
            print(f"      recovered footprint_m/depth_m for {len(bx)} boxes by reprojection")
        obs = classify.measure_ground_dimensions(obs)
        classes = classify.classify_tracks(obs, cfg, calibrated)
        obs = apply_track_classes(obs, classes)
        class_report = classify.classification_report(classes)
        if class_report:
            print(f"      resolved: {class_report['resolved_by_class']}")
            print(f"      model would have said: {class_report['model_would_have_said']}")
            print(f"      measured size overrode the model on "
                  f"{class_report['size_overrode_model']} tracks "
                  f"({class_report['size_overrode_model_pct']}%)")
        obs.to_csv(out("trajectories_csv"), index=False)
    else:
        model, det_settings = load_detector(cfg)
        print(f"[2/6] detector {det_settings['weights']} on device={det_settings['device']} "
              f"imgsz={det_settings['imgsz']} half={det_settings['half']}")

        for fr in iter_tracked_frames(
            model, video_path, det_settings, cfg["tracker"]["config"],
            frame_stride=stride, max_frames=max_frames,
        ):
            store.add_frame(fr["frame"], fr["t"], fr["tracks"])
            for tr in fr["tracks"]:
                x1, y1, x2, y2 = tr["xyxy"]
                box_rows.append(
                    {"frame": fr["frame"], "track_id": tr["track_id"],
                     "class": tr["cls"], "model_label": tr["model_label"],
                     "class_id": tr["class_id"], "confidence": float(tr["conf"]),
                     "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                )
            n_frames += 1
            if n_frames % 100 == 0:
                el = time.time() - t0
                print(f"      {n_frames} frames  {n_frames/el:.1f} fps  "
                      f"{len(fr['tracks'])} tracked in frame")

        elapsed = max(time.time() - t0, 1e-6)
        print(f"      done: {n_frames} frames in {elapsed:.1f}s ({n_frames/elapsed:.1f} fps)")

        raw = store.to_frame()
        boxes = pd.DataFrame(box_rows)
        if raw.empty:
            print("no detections - nothing to analyse")
            return 1

        # ---- Trajectories --------------------------------------------------
        print("[3/6] trajectories: kinematics + measured-size classification")
        obs = compute_kinematics(raw, fps, cfg, calibrated)
        # Fine-grained class comes from MEASURED ground size, not the COCO head
        # (which called 92% of everything "car" from this altitude). Dimensions
        # need `heading_px_deg`, so this runs after kinematics.
        obs = classify.measure_ground_dimensions(obs)
        classes = classify.classify_tracks(obs, cfg, calibrated)
        obs = apply_track_classes(obs, classes)
        class_report = classify.classification_report(classes)
        if class_report:
            print(f"      resolved: {class_report['resolved_by_class']}")
            print(f"      model would have said: {class_report['model_would_have_said']}")
            print(f"      measured size overrode the model on "
                  f"{class_report['size_overrode_model']} tracks "
                  f"({class_report['size_overrode_model_pct']}%): "
                  f"{'; '.join(class_report['override_flow'][:5])}")
        obs.to_csv(out("trajectories_csv"), index=False)
        # Boxes are persisted so --from-trajectories can also re-render the video.
        boxes.to_csv(os.path.join(out_dir, f"boxes{'_' + args.tag if args.tag else ''}.csv"),
                     index=False)
        print(f"      {len(obs)} observations over {obs['track_id'].nunique()} tracks "
              f"-> {out('trajectories_csv')}")

    det_settings = locals().get("det_settings", {"weights": cfg["detector"]["weights"],
                                                "imgsz": cfg["detector"]["imgsz"],
                                                "conf": cfg["detector"]["conf"],
                                                "iou": cfg["detector"]["iou"],
                                                "classes": cfg["detector"]["classes"],
                                                "device": "reused", "half": False})

    # ---- Road segmentation -------------------------------------------------
    # Runs on ONE frame because the aircraft hovers (see road_mask.py). It gives
    # the analytics layer a drivable-area denominator, so density is per square
    # metre OF ROAD rather than per square metre of frame - the frame includes
    # rooftops and vegetation that no vehicle can ever occupy.
    road_mask = None
    road_info = {"enabled": False, "reason": "disabled in config"}
    if cfg.get("segmentation", {}).get("enabled", False):
        print("[3b/6] road segmentation")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cfg["segmentation"].get("frame_index", 0)))
        ok, seg_frame = cap.read()
        cap.release()
        if not ok:
            road_info = {"enabled": False, "reason": "could not read a frame to segment"}
            print("      could not read a frame -> skipped")
        else:
            road_mask, road_info = road_mask_mod.build_road_mask(
                obs, seg_frame, cfg, boxes=boxes,
                frame_index=int(cfg["segmentation"].get("frame_index", 0)),
            )
            ov = road_mask_mod.render_mask_overlay(
                seg_frame, road_mask, out_path("road_mask", ".png")
            )
            road_info["overlay"] = ov
            road_info["road_area_m2"] = road_mask_mod.mask_ground_area_m2(road_mask, projector)
            print(f"      {road_info['source']}  road covers "
                  f"{road_info.get('road_area_frac', road_info['travelled_area_frac']):.1%} of frame"
                  + (f"  = {road_info['road_area_m2']} m2" if road_info.get("road_area_m2") else ""))
            if road_info.get("agreement"):
                a = road_info["agreement"]
                print(f"      SAM vs observed-traffic agreement: IoU {a['iou']}, "
                      f"SAM covers {a['sam_covers_travelled_frac']:.0%} of travelled area")
            if cfg["segmentation"].get("filter_detections", False):
                obs, filt = road_mask_mod.filter_to_road(
                    obs, road_mask,
                    classes_exempt=tuple(cfg["segmentation"].get("exempt_classes", ["pedestrian"])),
                    min_on_road_frac=float(cfg["segmentation"].get("min_on_road_fraction", 0.5)),
                )
                road_info["filter"] = filt
                print(f"      off-road filter: dropped {filt['tracks_dropped']} of "
                      f"{filt['tracks_before']} tracks ({filt['observations_dropped']} observations) "
                      f"- {filt.get('dropped_by_class', {})}")
                # Boxes must be filtered too, or the renderer would keep drawing
                # rectangles for tracks that no longer exist in the analytics.
                if boxes is not None and not boxes.empty:
                    boxes = boxes[boxes["track_id"].isin(obs["track_id"].unique())].reset_index(drop=True)

    # ---- Analytics ---------------------------------------------------------
    print("[4/6] analytics")
    summary = analytics.build_track_summary(obs, cfg, calibrated)
    counts = analytics.class_counts(summary)
    queues = analytics.detect_queues(obs, cfg, calibrated)
    congestion = analytics.congestion_timeline(obs, cfg, calibrated)
    active = analytics.active_track_timeline(obs)
    flow = analytics.directional_flow(summary, calibrated)
    turns = analytics.turning_movements(summary)
    stability = analytics.id_stability_stats(obs, summary, fps, stride)
    stationary = analytics.stationary_candidates(summary, cfg)
    # LEVEL 2: per-object velocity and acceleration in real units.
    kin, kin_stats = analytics.build_kinematics(obs, summary, cfg, calibrated)
    if kin_stats:
        print(f"      kinematics: {kin_stats['road_users']} road users, fleet mean "
              f"{kin_stats['fleet_mean_speed_kph']} km/h, hardest braking "
              f"{kin_stats['hardest_braking_ms2']} m/s2")

    inter = interactions.find_interactions(obs, cfg, calibrated)
    headways = interactions.following_headways(obs, inter, calibrated)
    # Duration + population let the summary report conflict RATES, which are
    # comparable across clips of different length and demand; a raw count is not.
    clip_duration_s = float(obs["timestamp"].max() - obs["timestamp"].min()) if not obs.empty else 0.0
    if calibrated:
        inter_summary = interactions.interaction_summary(
            inter, duration_s=clip_duration_s, n_road_users=len(summary)
        )
        inter_summary["available"] = True
        inter_summary["unit"] = "metric"
    else:
        inter_summary = {
            "available": False,
            "reason": "Metric calibration is required for physical conflict envelopes and closing-speed grading.",
            "unit": None,
            "pairs": None,
            "potential_conflicts": None,
            "critical": None,
            "serious": None,
        }

    events = anomalies.build_events(obs, summary, queues, inter, cfg, calibrated)
    events.to_csv(out("events_csv"), index=False)

    # Side tables the dashboard reads.
    summary.to_csv(os.path.join(out_dir, f"track_summary{'_' + args.tag if args.tag else ''}.csv"), index=False)
    for name, table in (
        ("queues", queues), ("congestion", congestion), ("active_tracks", active),
        ("directional_flow", flow), ("turning_movements", turns),
        ("interactions", inter), ("headways", headways),
        ("kinematics", kin),
    ):
        p = os.path.join(out_dir, f"{name}{'_' + args.tag if args.tag else ''}.csv")
        (table if table is not None and not table.empty else pd.DataFrame()).to_csv(p, index=False)

    speed_stats = {}
    if calibrated:
        moving = obs[obs["speed_kph"] > float(cfg["analytics"]["stationary_speed_kph"])]
        if not moving.empty:
            speed_stats = {
                "unit": "km/h",
                "basis": "telemetry-calibrated ground plane; estimate",
                "mean_moving_speed": round(float(moving["speed_kph"].mean()), 1),
                "median_moving_speed": round(float(moving["speed_kph"].median()), 1),
                "p85_speed": round(float(np.nanpercentile(moving["speed_kph"], 85)), 1),
                # p99.5 is the headline "top speed": the raw max is a handful of
                # frames on 4 of 292 tracks (measured), i.e. box jitter, not traffic.
                "robust_max_speed_p99_5": round(float(np.nanpercentile(moving["speed_kph"], 99.5)), 1),
                "max_speed": round(float(moving["speed_kph"].max()), 1),
                "max_speed_caveat": "raw max is jitter-sensitive; use p85 / p99.5 for reporting",
            }
    else:
        speed_stats = {"unit": "px/s", "basis": "no metric calibration available - km/h not reported",
                       "mean_moving_speed": round(float(obs["speed_px_s"].mean()), 1)}

    try:
        with open(args.config, "rb") as config_file:
            config_hash = hashlib.sha256(config_file.read()).hexdigest()
    except OSError:
        config_hash = None
    effective_config_hash = hashlib.sha256(
        yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()
    package_versions = {}
    for package in ("ultralytics", "torch", "opencv-python", "numpy", "pandas", "scipy", "PyYAML"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    summary_json = {
        "run": {
            "tag": args.tag or "default",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_file": os.path.abspath(args.config),
            "config_sha256": config_hash,
            "effective_config_sha256": effective_config_hash,
            "python_version": platform.python_version(),
            "package_versions": package_versions,
        },
        "video": {**vinfo, "path": video_path, "stride": stride,
                  "frames_processed": n_frames, "processing_fps": round(n_frames / elapsed, 2)},
        "detector": det_settings,
        "tracker": {"type": "bytetrack", "config": cfg["tracker"]["config"]},
        "calibration": calib_info,
        "road_segmentation": road_info,
        "counts": counts,
        "classification": class_report,
        "kinematics": kin_stats,
        "id_stability": stability,
        "speed": speed_stats,
        "congestion": {
            "levels": congestion["level"].value_counts().to_dict() if not congestion.empty else {},
            "peak_score": round(float(congestion["score"].max()), 3) if not congestion.empty else None,
            "method": "weighted blend of slow-fraction, speed deficit vs observed free-flow, and relative density; analytical estimate, not a certified level of service",
        },
        "queues": {
            "detections": int(len(queues)) if not queues.empty else 0,
            "largest_vehicles": int(queues["vehicles"].max()) if not queues.empty else 0,
            "longest": float(queues["length"].max()) if not queues.empty else 0.0,
            "unit": queues["length_unit"].iloc[0] if not queues.empty else None,
        },
        "turning_movements": turns.groupby("turn_type")["road_users"].sum().to_dict() if not turns.empty else {},
        "stationary_candidates": int(len(stationary)),
        "interactions": inter_summary,
        "events": {
            "total": int(len(events)),
            "by_type": events["event_type"].value_counts().to_dict() if not events.empty else {},
            "by_severity": events["severity"].value_counts().to_dict() if not events.empty else {},
        },
    }
    with open(out("summary_json"), "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)

    print(f"      road users: {counts['total']}  {counts['by_class']}")
    print(f"      events: {summary_json['events']['total']}  {summary_json['events']['by_type']}")
    print(f"      interactions: {inter_summary}")

    # ---- Trajectory map ----------------------------------------------------
    print("[5/6] trajectory map")
    canvas = visualize.render_trajectory_map(obs, summary, calibrated)
    if canvas is not None:
        mp = os.path.join(out_dir, f"trajectory_map{'_' + args.tag if args.tag else ''}.png")
        cv2.imwrite(mp, canvas)
        print(f"      -> {mp}")

    # ---- Annotated video ---------------------------------------------------
    if cfg["output"].get("write_video", True):
        vid_out = next_video_path(out("annotated_video"))
        print(f"[6/6] rendering annotated video -> {vid_out}")
        r = visualize.render_annotated_video(
            video_path=video_path, out_path=vid_out,
            obs=obs, boxes=boxes, summary=summary, events=events,
            congestion=congestion, counts=counts, calibrated=calibrated,
            fps=fps, frame_stride=stride,
            trail_len=int(cfg["output"].get("trail_len", 45)),
            max_frames=max_frames,
            road_mask=road_mask,
        )
        print(f"      -> {r['output']} ({r['frames_written']} frames)")
    else:
        print("[6/6] skipped annotated video (--no-video)")

    print("\nDONE. outputs in", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
