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
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import analytics, anomalies, interactions, visualize
from pipeline import contact_points as contact_mod
from pipeline import road_mask as road_mask_mod
from pipeline.calibration import build_projector
from pipeline.detector import load_detector
from pipeline.tracker import iter_tracked_frames, video_info
from pipeline.trajectory import (
    TrajectoryStore,
    apply_contact_refinement,
    apply_track_classes,
    compute_kinematics,
    contact_use_seg,
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
    ap.add_argument(
        "--refine-contacts", dest="refine_contacts", action="store_true", default=None,
        help="refine trajectory anchors with MobileSAM instance masks (overrides "
             "segmentation.contact.enabled). Works with --from-trajectories, so an "
             "existing run can be refined without re-detecting.",
    )
    ap.add_argument(
        "--no-refine-contacts", dest="refine_contacts", action="store_false",
        help="keep the bbox bottom-centre anchor - the baseline for the A/B comparison",
    )
    args = ap.parse_args()
    in_tag = args.tag if args.in_tag is None else args.in_tag

    cfg = load_config(args.config)
    if args.video:
        cfg["video"]["path"] = args.video
    if args.srt:
        cfg["telemetry"]["srt_path"] = args.srt
    if args.max_frames is not None:
        cfg["video"]["max_frames"] = args.max_frames
    if args.stride is not None:
        cfg["video"]["frame_stride"] = args.stride
    if args.no_video:
        cfg["output"]["write_video"] = False
    if args.refine_contacts is not None:
        cfg.setdefault("segmentation", {}).setdefault("contact", {})["enabled"] = args.refine_contacts

    refine_on = bool((cfg.get("segmentation", {}) or {}).get("contact", {}).get("enabled", False))

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

    def _contact_line(info: dict) -> str:
        if not info.get("enabled"):
            return f"skipped: {info.get('reason', 'disabled')}"
        s = (f"{info['masks_accepted']}/{info['boxes_prompted']} masks accepted "
             f"({info['mask_success_rate']:.0%}) over {info['frames_segmented']} frames")
        if info.get("rejections"):
            s += f"; rejected {info['rejections']}"
        return s

    vinfo = video_info(video_path)
    fps = vinfo["fps"]
    print(f"[1/7] video {video_path}  {vinfo['width']}x{vinfo['height']} @ {fps:.2f}fps  "
          f"{vinfo['frames']} frames  stride={stride}")

    # ---- Calibration -------------------------------------------------------
    projector, calib_info = build_projector(cfg, vinfo["width"], vinfo["height"])
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

    seg_frame_index = int((cfg.get("segmentation", {}) or {}).get("frame_index", 0))
    contact_use = "seg" if contact_use_seg(cfg) else "bbox"
    contact_info: dict = {"enabled": False, "reason": "not requested"}
    merge_stats: dict = {"applied": False}
    vehicle_union = None
    swept_union = None
    refined = pd.DataFrame()

    reuse = args.from_trajectories
    if reuse:
        # Re-analysis path: detection/tracking already happened, so read the saved
        # trajectories and jump straight to analytics.
        traj_path = reuse
        if reuse == "__default__":
            root, ext = os.path.splitext(cfg["output"]["trajectories_csv"])
            traj_path = f"{root}_{in_tag}{ext}" if in_tag else f"{root}{ext}"
        obs = pd.read_csv(traj_path)
        boxes_path = in_path("boxes")
        boxes = pd.read_csv(boxes_path) if os.path.exists(boxes_path) else pd.DataFrame()
        calibrated = "sx" in obs.columns and obs["sx"].notna().any()
        n_frames = int(obs["frame"].nunique())
        elapsed = 1e-6
        print(f"[2/7] REUSING {traj_path}: {len(obs)} observations, "
              f"{obs['track_id'].nunique()} tracks - no detection run")
        if boxes.empty:
            print("      (no boxes.csv -> annotated video cannot be re-rendered)")
            cfg["output"]["write_video"] = False

        if refine_on and not boxes.empty:
            # The architectural payoff, applied to segmentation: boxes.csv holds every
            # detection, so MobileSAM can re-anchor an existing run without paying for
            # YOLO again. This is what makes the bbox-vs-mask A/B cheap enough to run.
            print("[3/7] contact-point refinement (MobileSAM instance masks)")
            refined, contact_info, vehicle_union, swept_union = contact_mod.refine_contacts(
                video_path, boxes, cfg, frame_stride=stride, max_frames=max_frames,
                capture_frame=seg_frame_index,
            )
            print("      " + _contact_line(contact_info))

            # Everything derived from the old anchor has to go: speed, bearing and
            # the footprint-based size split are all functions of position.
            derived = ["class_raw", "class_source", "sx", "sy", "speed_kph",
                       "speed_px_s", "bearing_deg", "heading_px_deg", "accel_kph_s"]
            obs = obs.drop(columns=[c for c in derived if c in obs.columns])
            # Undo the previous LGV/HGV split so it is re-decided on the new
            # contact-line footprint rather than frozen from the box-corner one.
            if "class" in obs.columns:
                obs["class"] = obs["class"].replace({"lgv": "truck", "hgv": "truck"})

            obs, merge_stats = apply_contact_refinement(obs, refined, projector, use=contact_use)
            print(f"      anchors: {merge_stats.get('from_mask', 0)} from mask, "
                  f"{merge_stats.get('from_interpolated_offset', 0)} interpolated, "
                  f"{merge_stats.get('kept_bbox', 0)} kept bbox")

            print("[4/7] trajectories: kinematics recomputed on the refined anchor")
            obs = compute_kinematics(obs, fps, cfg, calibrated)
            classes = resolve_track_classes(obs, cfg, calibrated)
            obs = apply_track_classes(obs, classes)
            obs.to_csv(out("trajectories_csv"), index=False)
            if not refined.empty:
                refined.to_csv(out_path("segments"), index=False)
            print(f"      {len(obs)} observations over {obs['track_id'].nunique()} tracks "
                  f"-> {out('trajectories_csv')}")
        else:
            print(f"[3/7] contact-point refinement: {_contact_line(contact_info)}")
            print("[4/7] trajectories: reused as-is (kinematics already computed)")
    else:
        model, det_settings = load_detector(cfg)
        print(f"[2/7] detector {det_settings['weights']} on device={det_settings['device']} "
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

        # ---- Contact-point refinement --------------------------------------
        # Second pass over the video, before kinematics, because speed and bearing
        # are derived from position - refining afterwards would leave them
        # describing the old anchor. See pipeline/contact_points.py.
        if refine_on:
            print("[3/7] contact-point refinement (MobileSAM instance masks)")
            refined, contact_info, vehicle_union, swept_union = contact_mod.refine_contacts(
                video_path, boxes, cfg, frame_stride=stride, max_frames=max_frames,
                capture_frame=seg_frame_index,
            )
            print("      " + _contact_line(contact_info))
            raw, merge_stats = apply_contact_refinement(raw, refined, projector, use=contact_use)
            print(f"      anchors: {merge_stats.get('from_mask', 0)} from mask, "
                  f"{merge_stats.get('from_interpolated_offset', 0)} interpolated, "
                  f"{merge_stats.get('kept_bbox', 0)} kept bbox")
            if not refined.empty:
                refined.to_csv(out_path("segments"), index=False)
        else:
            contact_info = {"enabled": False, "reason": "disabled in config / --no-refine-contacts"}
            print("[3/7] contact-point refinement: disabled -> bbox bottom-centre anchor")

        # ---- Trajectories --------------------------------------------------
        print("[4/7] trajectories: kinematics + per-track class resolution")
        obs = compute_kinematics(raw, fps, cfg, calibrated)
        classes = resolve_track_classes(obs, cfg, calibrated)
        obs = apply_track_classes(obs, classes)
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
        print("[4b/7] road segmentation")
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
                vehicle_mask=vehicle_union,
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
                )
                road_info["filter"] = filt
                print(f"      off-road filter: dropped {filt['observations_dropped']} of "
                      f"{filt['observations_before']} observations, "
                      f"{filt['tracks_dropped_entirely']} tracks removed entirely")

    # ---- Analytics ---------------------------------------------------------
    print("[5/7] analytics")
    summary = analytics.build_track_summary(obs, cfg, calibrated)
    counts = analytics.class_counts(summary)
    queues = analytics.detect_queues(obs, cfg, calibrated)
    congestion = analytics.congestion_timeline(obs, cfg, calibrated)
    active = analytics.active_track_timeline(obs)
    flow = analytics.directional_flow(summary, calibrated)
    turns = analytics.turning_movements(summary)
    stability = analytics.id_stability_stats(obs, summary, fps, stride)
    stationary = analytics.stationary_candidates(summary, cfg)

    inter = interactions.find_interactions(obs, cfg, calibrated)
    headways = interactions.following_headways(obs, inter, calibrated)
    # Duration + population let the summary report conflict RATES, which are
    # comparable across clips of different length and demand; a raw count is not.
    clip_duration_s = float(obs["timestamp"].max() - obs["timestamp"].min()) if not obs.empty else 0.0
    inter_summary = interactions.interaction_summary(
        inter, duration_s=clip_duration_s, n_road_users=len(summary)
    )

    events = anomalies.build_events(obs, summary, queues, inter, cfg, calibrated)
    events.to_csv(out("events_csv"), index=False)

    # Side tables the dashboard reads.
    summary.to_csv(os.path.join(out_dir, f"track_summary{'_' + args.tag if args.tag else ''}.csv"), index=False)
    for name, table in (
        ("queues", queues), ("congestion", congestion), ("active_tracks", active),
        ("directional_flow", flow), ("turning_movements", turns),
        ("interactions", inter), ("headways", headways),
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

    # Contact-point refinement is reported with MEASURED effect, not asserted.
    # `metrics` tests a falsifiable prediction (shift must grow with off-nadir angle
    # and object height); `path_quality` is computed for every run, refined or not,
    # so the bbox-vs-mask A/B is a diff of two summary.json files.
    contact_block = {**contact_info, "merge": merge_stats}
    if contact_info.get("enabled"):
        contact_block["metrics"] = contact_mod.contact_metrics(obs, projector)
    quality = contact_mod.path_quality(obs, summary, calibrated)

    summary_json = {
        "video": {**vinfo, "path": video_path, "stride": stride,
                  "frames_processed": n_frames, "processing_fps": round(n_frames / elapsed, 2)},
        "detector": det_settings,
        "tracker": {"type": "bytetrack", "config": cfg["tracker"]["config"]},
        "calibration": calib_info,
        "contact_refinement": contact_block,
        "path_quality": quality,
        "road_segmentation": road_info,
        "counts": counts,
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
    if contact_block.get("metrics"):
        m = contact_block["metrics"]
        print(f"      contact shift: median {m.get('median_shift_px')}px"
              + (f" = {m.get('median_shift_m')}m" if m.get("median_shift_m") is not None else "")
              + f"  by class {m.get('median_shift_px_by_class', {})}")
        if m.get("by_image_row_band"):
            trend = {k: v["median_shift_px"] for k, v in m["by_image_row_band"].items()}
            print(f"      shift by image row (top->bottom, should DECREASE): {trend}")
    if quality:
        print(f"      path quality: {quality}")

    # ---- Trajectory map ----------------------------------------------------
    print("[6/7] trajectory map")
    canvas = visualize.render_trajectory_map(obs, summary, calibrated)
    if canvas is not None:
        mp = os.path.join(out_dir, f"trajectory_map{'_' + args.tag if args.tag else ''}.png")
        cv2.imwrite(mp, canvas)
        print(f"      -> {mp}")

    # ---- Annotated video ---------------------------------------------------
    if cfg["output"].get("write_video", True):
        vid_out = next_video_path(out("annotated_video"))
        print(f"[7/7] rendering annotated video -> {vid_out}")
        ccfg = (cfg.get("segmentation", {}) or {}).get("contact", {}) or {}
        r = visualize.render_annotated_video(
            video_path=video_path, out_path=vid_out,
            obs=obs, boxes=boxes, summary=summary, events=events,
            congestion=congestion, counts=counts, calibrated=calibrated,
            fps=fps, frame_stride=stride,
            trail_len=int(cfg["output"].get("trail_len", 45)),
            max_frames=max_frames,
            road_mask=road_mask,
            # Draw the path the analytics actually measured, and show the
            # box -> tyre-line correction so the refinement is visible, not claimed.
            use_contact_anchor=(contact_use == "seg"),
            draw_contacts=bool(ccfg.get("draw_contacts", True)),
        )
        print(f"      -> {r['output']} ({r['frames_written']} frames), "
              f"trails from {r['trail_anchor']}")
        root, ext = os.path.splitext(vid_out)
        web_out = f"{root}_web{ext}"
        import shutil, subprocess
        if shutil.which("ffmpeg"):
            try:
                subprocess.run(
                    ["ffmpeg", "-i", vid_out, "-vcodec", "libx264", "-crf", "28", "-y", web_out],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"      -> web-compatible H.264 video: {web_out}")
            except Exception as ex:
                print(f"      (ffmpeg web conversion failed: {ex})")
    else:
        print("[7/7] skipped annotated video (--no-video)")

    print("\nDONE. outputs in", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
