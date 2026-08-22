"""
Instance-mask contact points - MobileSAM refinement of the trajectory anchor.

WHY THIS EXISTS
---------------
Every position in this system was previously the bottom-centre of a YOLO box
(`trajectory.TrajectoryStore.add_frame`), and `calibration.py` already conceded
the limitation: "a box bottom is still an approximation."

The gimbal on this footage sits at -63 deg pitch, NOT nadir. With f_px = 1280 on
1920x1080 the vertical FOV is 45.7 deg, so the frame spans 4.1 deg off-nadir at the
bottom edge to 49.9 deg at the top. An elevated point projects as though it were
displaced radially outward from the nadir by `height * tan(off-nadir)`:

    object       h      bottom (4 deg)   centre (27 deg)   top (50 deg)
    car        1.5 m       0.11 m           0.76 m           1.78 m
    bus/hgv    3.2 m       0.23 m           1.63 m           3.80 m

An axis-aligned box wraps the WHOLE silhouette, roof included. Its bottom edge y2
is the lowest silhouette pixel (roughly the near tyre line, about right), but its
horizontal centre (x1+x2)/2 is the centre of a silhouette stretched by that radial
roof displacement. So (cx, y2) corresponds to NO physical feature of the vehicle:
its v comes from the wheels and its u from a roof-contaminated average.

Three consequences follow from the geometry regardless of the exact magnitude:

  * the bias grows ~17x from the bottom of the frame to the top, so it does not
    cancel as a vehicle crosses the scene - it injects false velocity into
    speed_kph and bearing_deg;
  * it is orientation-dependent, so a TURNING vehicle's box centre shifts even
    when its true footprint centre does not;
  * for scale, the conflict envelope for two cars is 2.07 m
    (interactions.CLASS_HALF_WIDTH_M). In the upper half of the frame the
    roof-displacement scale is the same order as the entire envelope.

A real instance mask gives the actual tyre-contact line instead, plus a true
contact-line WIDTH to replace the box-corner proxy behind `footprint_m` and the
LGV/HGV split.

HOW IT RUNS
-----------
A standalone SECOND PASS over the video driven by the already-persisted
`outputs/boxes.csv`, mirroring how `visualize.render_annotated_video` replays the
clip with hindsight. That choice is load-bearing:

  * refinement finishes BEFORE `compute_kinematics`, so `trajectories.csv` stays
    the single hard interface and no analytic module learns that SAM exists;
  * it can be applied to an EXISTING run with no re-detection, because boxes.csv
    carries every box - the 25-minute YOLO pass is never repeated;
  * a SAM failure cannot destroy a completed detection run.

Cost: one `SAM(...)(img, bboxes=[...])` call per frame carries every box in that
frame - one image encode, N cheap mask decodes. MobileSAM is a TinyViT-5M encoder
(~8 ms) plus a ~4 ms decoder. A `stride` escape valve is provided anyway.

HONESTY
-------
Both points are kept. `x, y` remain the box bottom-centre; `x_seg, y_seg` hold the
refined point; config chooses which feeds the world coordinates. So the
improvement is MEASURED (see `contact_metrics` / `path_quality`) rather than
asserted, every observation records its own `contact_source`, and a bad result on
other footage is one config key away from being reverted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# cv2 is imported defensively so the pure estimators below stay unit-testable in a
# bare environment (see tests/test_contact_points.py). The video pass degrades with
# a note rather than an ImportError, which is the same contract road_mask uses.
try:
    import cv2
except Exception:                                   # pragma: no cover
    cv2 = None

REFINED_COLUMNS = [
    "frame",
    "track_id",
    "x_seg",
    "y_seg",
    "seg_l_x",
    "seg_r_x",
    "contact_w_px",
    "contact_source",
]

# A mask with fewer pixels than this in its contact band cannot support a median.
MIN_MASK_PIXELS = 24
MIN_BAND_PIXELS = 3


def _contact_from_mask(
    mask: np.ndarray,
    box: tuple[float, float, float, float],
    contact_percentile: float = 92.0,
    band_px: int = 6,
) -> tuple[float, float, float, float] | None:
    """
    Ground-contact point and contact-line width from one instance mask.

    Returns (u_c, v_c, l_x, r_x) in FULL-FRAME pixel coordinates, or None if the
    mask cannot support an estimate.

        v_c  = percentile(v of mask pixels, contact_percentile)
        band = mask pixels with v >= v_c - band_px      # the tyre-contact band
        u_c  = median(u of band)
        l,r  = 5th / 95th percentile of u over the band

    The percentile - rather than max(v) - is the whole point. The dominant SAM
    failure mode on sunlit aerial imagery is the cast shadow leaking into the mask
    and extending it downward. max(v) would follow the shadow tip and drag the
    contact point an unbounded distance; a percentile bounds that error to the
    tail SAM actually mislabelled.

    Only pixels inside the prompt box are considered, so a mask that bled into a
    neighbouring vehicle contributes nothing outside its own detection.
    """
    h, w = mask.shape[:2]
    x1 = int(max(np.floor(box[0]), 0))
    y1 = int(max(np.floor(box[1]), 0))
    x2 = int(min(np.ceil(box[2]), w))
    y2 = int(min(np.ceil(box[3]), h))
    if x2 <= x1 or y2 <= y1:
        return None

    sub = mask[y1:y2, x1:x2]
    ys, xs = np.nonzero(sub)
    if ys.size < MIN_MASK_PIXELS:
        return None

    v = ys.astype(np.float64) + y1
    u = xs.astype(np.float64) + x1

    v_c = float(np.percentile(v, contact_percentile))
    band = v >= (v_c - float(band_px))
    if band.sum() < MIN_BAND_PIXELS:
        # Degenerate percentile (very flat mask): fall back to the true bottom rows.
        band = v >= (v.max() - float(band_px))
        if band.sum() < MIN_BAND_PIXELS:
            return None

    ub = u[band]
    return (
        float(np.median(ub)),
        v_c,
        float(np.percentile(ub, 5)),
        float(np.percentile(ub, 95)),
    )


def _vet_mask(
    mask: np.ndarray,
    box: tuple[float, float, float, float],
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 1.10,
    min_containment: float = 0.70,
) -> tuple[bool, str]:
    """
    Reject degenerate masks before they can move a trajectory point.

    Extends the `frac > 0.85 or frac < 0.0005` idea already in
    `road_mask.sam_segments`, but relative to the PROMPT BOX rather than the frame,
    which is the meaningful denominator for an instance:

      * area / box_area outside [min_area_ratio, max_area_ratio] - a mask much
        larger than its own prompt means SAM grabbed the carriageway or the shadow;
        a much smaller one caught a windscreen or a wheel.
      * containment - the fraction of mask area that lies inside the prompt box.
        The detector box already excludes most of a cast shadow, so a mask spilling
        out of it is the signature of exactly the failure we care about.

    Returns (ok, reason). A rejected mask falls back to the box point, tagged.
    """
    h, w = mask.shape[:2]
    area = float(mask.sum())
    if area <= 0:
        return False, "empty_mask"

    x1 = int(max(np.floor(box[0]), 0))
    y1 = int(max(np.floor(box[1]), 0))
    x2 = int(min(np.ceil(box[2]), w))
    y2 = int(min(np.ceil(box[3]), h))
    box_area = float(max((x2 - x1) * (y2 - y1), 1))

    ratio = area / box_area
    if ratio < min_area_ratio:
        return False, "mask_too_small_vs_box"
    if ratio > max_area_ratio:
        return False, "mask_too_large_vs_box"

    containment = float(mask[y1:y2, x1:x2].sum()) / area
    if containment < min_containment:
        return False, "mask_escapes_box"
    return True, "ok"


def _iou(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Overlap of a mask with a box - used only to re-pair masks to prompts."""
    h, w = mask.shape[:2]
    x1 = int(max(np.floor(box[0]), 0))
    y1 = int(max(np.floor(box[1]), 0))
    x2 = int(min(np.ceil(box[2]), w))
    y2 = int(min(np.ceil(box[3]), h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float(mask[y1:y2, x1:x2].sum())
    union = float(mask.sum()) + float((x2 - x1) * (y2 - y1)) - inter
    return inter / max(union, 1.0)


def _pair_masks_to_boxes(masks: list[np.ndarray], rows: list) -> list[tuple[int, np.ndarray]]:
    """
    Associate returned masks with the boxes that prompted them.

    Ultralytics returns one mask per box prompt in prompt order, so the positional
    pairing is the normal path. When the counts disagree - a version difference, or
    a prompt SAM declined - fall back to greedy IoU matching instead of silently
    assigning masks to the wrong vehicles, which would corrupt trajectories in a
    way no downstream check could catch.
    """
    if len(masks) == len(rows):
        return list(enumerate(masks))

    pairs: list[tuple[int, np.ndarray]] = []
    taken: set[int] = set()
    for m in masks:
        best_i, best_iou = -1, 0.0
        for i, r in enumerate(rows):
            if i in taken:
                continue
            s = _iou(m, (r.x1, r.y1, r.x2, r.y2))
            if s > best_iou:
                best_i, best_iou = i, s
        if best_i >= 0 and best_iou > 0.1:
            taken.add(best_i)
            pairs.append((best_i, m))
    return pairs


def _resolve_device(requested: str = "auto") -> str:
    """Same policy as detector.resolve_device, but without importing torch eagerly."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def refine_contacts(
    video_path: str,
    boxes: pd.DataFrame,
    cfg: dict,
    frame_stride: int = 1,
    max_frames: int | None = None,
    capture_frame: int | None = None,
    progress_every: int = 100,
) -> tuple[pd.DataFrame, dict, np.ndarray | None]:
    """
    Second pass over the video, replacing box bottom-centres with mask contacts.

    Returns (refined, info, vehicle_union):
      refined       one row per (frame, track_id) that was processed, REFINED_COLUMNS
      info          measured coverage / rejection counts, JSON-serialisable
      vehicle_union uint8 union of accepted masks on `capture_frame`, or None.
                    Handed to `road_mask._occupied_mask` so the bare-road SAM
                    prompts can exclude actual vehicle PIXELS instead of whole
                    boxes - a free improvement to the road mask, since these masks
                    have already been paid for.

    Never raises. Every failure path returns an empty table plus a note, exactly as
    `road_mask.sam_segments` does, so the caller falls back to the box point.
    """
    ccfg = (cfg.get("segmentation", {}) or {}).get("contact", {}) or {}
    empty = pd.DataFrame(columns=REFINED_COLUMNS)

    if not ccfg.get("enabled", False):
        return empty, {"enabled": False, "reason": "disabled in config"}, None
    if cv2 is None:
        return empty, {"enabled": False, "reason": "opencv unavailable"}, None
    if boxes is None or boxes.empty or "frame" not in boxes:
        return empty, {"enabled": False, "reason": "no boxes.csv to refine"}, None

    try:
        from ultralytics import SAM
    except Exception as exc:
        return empty, {"enabled": False, "reason": f"ultralytics SAM unavailable: {exc}"}, None

    weights = (cfg.get("segmentation", {}) or {}).get("sam_weights", "mobile_sam.pt")
    refine_stride = max(1, int(ccfg.get("stride", 1)))
    pct = float(ccfg.get("contact_percentile", 92.0))
    band = int(ccfg.get("band_px", 6))
    min_ar = float(ccfg.get("min_mask_bbox_area_ratio", 0.15))
    max_ar = float(ccfg.get("max_mask_bbox_area_ratio", 1.10))
    min_cont = float(ccfg.get("min_containment", 0.70))
    max_obj = int(ccfg.get("max_objects_per_frame", 64))
    device = _resolve_device(str(ccfg.get("device", "auto")))

    try:
        model = SAM(weights)
    except Exception as exc:
        return empty, {"enabled": False, "reason": f"could not load {weights}: {exc}"}, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return empty, {"enabled": False, "reason": f"cannot open video: {video_path}"}, None

    box_by_frame = {int(f): g for f, g in boxes.groupby("frame")}
    rows: list[dict] = []
    vehicle_union: np.ndarray | None = None

    n_frames_refined = 0
    n_prompted = 0
    n_accepted = 0
    n_no_mask = 0
    n_over_cap = 0
    rejections: dict[str, int] = {}
    infer_failures = 0

    frame_idx = -1
    processed = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frame_idx += 1
            # Mirror tracker.iter_tracked_frames exactly so frame indices align
            # with the ones already recorded in boxes.csv.
            if frame_stride > 1 and frame_idx % frame_stride:
                continue
            if max_frames is not None and processed >= max_frames:
                break
            this_processed = processed
            processed += 1

            if this_processed % refine_stride:
                continue
            g = box_by_frame.get(frame_idx)
            if g is None or g.empty:
                continue

            frame_rows = list(g.itertuples(index=False))
            if len(frame_rows) > max_obj:
                # Keep the largest boxes: they are both the most reliably
                # segmented and the most biased by roof displacement.
                frame_rows.sort(key=lambda r: (r.x2 - r.x1) * (r.y2 - r.y1), reverse=True)
                n_over_cap += len(frame_rows) - max_obj
                frame_rows = frame_rows[:max_obj]

            prompts = [[float(r.x1), float(r.y1), float(r.x2), float(r.y2)] for r in frame_rows]
            n_prompted += len(prompts)

            try:
                res = model(img, bboxes=prompts, device=device, verbose=False)
            except Exception:
                infer_failures += 1
                if infer_failures <= 3:
                    continue
                # Repeated failures mean the model is unusable, not this frame.
                break

            h, w = img.shape[:2]
            masks: list[np.ndarray] = []
            for r in res:
                if r.masks is None:
                    continue
                for m in r.masks.data.cpu().numpy():
                    mm = (m > 0.5).astype(np.uint8)
                    if mm.shape != (h, w):
                        mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks.append(mm.astype(bool))
            if not masks:
                n_no_mask += len(prompts)
                continue

            n_frames_refined += 1
            capture_here = capture_frame is not None and frame_idx == int(capture_frame)
            if capture_here:
                vehicle_union = np.zeros((h, w), np.uint8)

            for i, mask in _pair_masks_to_boxes(masks, frame_rows):
                r = frame_rows[i]
                box = (float(r.x1), float(r.y1), float(r.x2), float(r.y2))

                good, reason = _vet_mask(mask, box, min_ar, max_ar, min_cont)
                if not good:
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue

                est = _contact_from_mask(mask, box, contact_percentile=pct, band_px=band)
                if est is None:
                    rejections["no_contact_band"] = rejections.get("no_contact_band", 0) + 1
                    continue

                u_c, v_c, l_x, r_x = est
                n_accepted += 1
                if capture_here and vehicle_union is not None:
                    vehicle_union[mask] = 255

                rows.append(
                    {
                        "frame": int(r.frame),
                        "track_id": int(r.track_id),
                        "x_seg": round(u_c, 2),
                        "y_seg": round(v_c, 2),
                        "seg_l_x": round(l_x, 2),
                        "seg_r_x": round(r_x, 2),
                        "contact_w_px": round(r_x - l_x, 2),
                        "contact_source": "mask",
                    }
                )

            if progress_every and n_frames_refined and n_frames_refined % progress_every == 0:
                print(f"      segmented {n_frames_refined} frames, {n_accepted} masks accepted")
    finally:
        cap.release()

    refined = pd.DataFrame(rows, columns=REFINED_COLUMNS) if rows else empty
    info = {
        "enabled": True,
        "method": "MobileSAM instance masks, one batched box-prompt call per frame",
        "weights": weights,
        "device": device,
        "stride": refine_stride,
        "frames_segmented": n_frames_refined,
        "boxes_prompted": n_prompted,
        "masks_accepted": n_accepted,
        "mask_success_rate": round(n_accepted / max(n_prompted, 1), 3),
        "prompts_without_mask": n_no_mask,
        "boxes_over_frame_cap": n_over_cap,
        "rejections": rejections,
        "inference_failures": infer_failures,
        "contact_estimator": (
            f"u = median of contact band, v = p{pct:g} of mask rows, band {band}px; "
            "percentile rather than max(v) bounds the error when a cast shadow "
            "leaks into the mask"
        ),
    }
    if refine_stride > 1:
        info["stride_note"] = (
            f"only every {refine_stride}th processed frame was segmented; the "
            "box->mask OFFSET is interpolated across the gaps (the offset is smooth "
            "in image position and orientation, the position is not)"
        )
    if n_accepted == 0:
        info["outcome"] = "no mask survived vetting -> box bottom-centre kept everywhere"
    return refined, info, vehicle_union


# --------------------------------------------------------------------------
# Validation. The plan for this feature makes a falsifiable prediction; these
# functions are what test it, and their output goes into summary.json.
# --------------------------------------------------------------------------

def contact_metrics(obs: pd.DataFrame, projector=None, n_row_bands: int = 4) -> dict:
    """
    How far did the contact point actually move, and where?

    The geometry in the module docstring predicts the shift GROWS MONOTONICALLY
    from the bottom of the frame to the top (4 deg -> 50 deg off-nadir) and SCALES
    WITH OBJECT HEIGHT. That is falsifiable: if the measured shift is flat across
    image rows, the estimator is wrong and this feature has failed its own test.
    Hence the per-row-band and per-class breakdowns rather than one average.
    """
    out: dict = {"observations": int(len(obs))}
    if obs.empty or "x_seg" not in obs.columns:
        out["refined"] = 0
        return out

    d = obs.dropna(subset=["x_seg", "y_seg"])
    out["refined"] = int(len(d))
    out["refined_frac"] = round(len(d) / max(len(obs), 1), 3)
    if "contact_source" in obs.columns:
        out["by_source"] = obs["contact_source"].fillna("bbox_fallback").value_counts().to_dict()
    if d.empty:
        return out

    dx = (d["x_seg"] - d["x"]).to_numpy(float)
    dy = (d["y_seg"] - d["y"]).to_numpy(float)
    shift_px = np.hypot(dx, dy)
    out["median_shift_px"] = round(float(np.median(shift_px)), 2)
    out["p90_shift_px"] = round(float(np.percentile(shift_px, 90)), 2)
    out["median_shift_u_px"] = round(float(np.median(np.abs(dx))), 2)
    out["median_shift_v_px"] = round(float(np.median(np.abs(dy))), 2)

    # Per class - the height dependence.
    if "class" in d.columns:
        out["median_shift_px_by_class"] = {
            str(k): round(float(np.median(np.hypot(g["x_seg"] - g["x"], g["y_seg"] - g["y"]))), 2)
            for k, g in d.groupby("class")
        }
        out["mask_success_rate_by_class"] = {
            str(k): round(float(g["x_seg"].notna().mean()), 3)
            for k, g in obs.groupby("class")
        }

    # Per image-row band - the off-nadir dependence. Band 0 is the TOP of the
    # frame (most oblique, ~50 deg), so the predicted trend is decreasing.
    v = d["y"].to_numpy(float)
    if np.isfinite(v).any():
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        edges = np.linspace(lo, hi, n_row_bands + 1)
        bands = {}
        for i in range(n_row_bands):
            sel = (v >= edges[i]) & (v <= edges[i + 1] if i == n_row_bands - 1 else v < edges[i + 1])
            if sel.sum() >= 20:
                bands[f"rows_{int(edges[i])}_{int(edges[i + 1])}"] = {
                    "n": int(sel.sum()),
                    "median_shift_px": round(float(np.median(shift_px[sel])), 2),
                }
        out["by_image_row_band"] = bands
        out["row_band_note"] = (
            "bands run top->bottom of frame = 50deg->4deg off-nadir; the predicted "
            "shift DECREASES down this list. A flat profile falsifies the method."
        )

    # Metric shift, using the same projector the trajectories use.
    if projector is not None:
        try:
            a = projector.pixels_to_ground(d[["x", "y"]].to_numpy(float))
            b = projector.pixels_to_ground(d[["x_seg", "y_seg"]].to_numpy(float))
            m = np.linalg.norm(b - a, axis=1)
            m = m[np.isfinite(m)]
            if m.size:
                out["median_shift_m"] = round(float(np.median(m)), 3)
                out["p90_shift_m"] = round(float(np.percentile(m, 90)), 3)
        except Exception:
            pass
    return out


def path_quality(obs: pd.DataFrame, summary: pd.DataFrame, calibrated: bool) -> dict:
    """
    Position-noise measures computed for EVERY run, so the A/B is a diff of two
    summary.json files rather than something that has to be recomputed by hand.

      speed_jitter        median per-track std of frame-to-frame speed change.
      straight_path_rms   for tracks the analytics called `straight`, the RMS
                          lateral deviation from a straight line fitted to the
                          RAW ground path. This is the cleanest single number,
                          because a straight-driving vehicle's true path really is
                          a line. Raw (not smoothed) coordinates on purpose -
                          smoothing would hide exactly the noise being measured.
      footprint_cv        within-track coefficient of variation of footprint_m.
                          A mask contact line should be markedly more stable than
                          two independently jittering box corners.
    """
    out: dict = {}
    if obs.empty:
        return out

    scol = "speed_kph" if calibrated else "speed_px_s"
    if scol in obs.columns:
        j = (
            obs.sort_values(["track_id", "frame"])
            .groupby("track_id")[scol]
            .apply(lambda s: s.diff().std())
            .dropna()
        )
        if not j.empty:
            out["speed_jitter"] = round(float(j.median()), 3)
            out["speed_jitter_unit"] = "km/h per frame" if calibrated else "px/s per frame"

    if calibrated and "footprint_m" in obs.columns:
        cv = obs.groupby("track_id")["footprint_m"].apply(
            lambda s: s.std() / s.mean() if s.notna().sum() > 2 and s.mean() else np.nan
        ).dropna()
        if not cv.empty:
            out["footprint_cv"] = round(float(cv.median()), 4)

    if calibrated and summary is not None and not summary.empty and "turn_type" in summary.columns:
        straight = set(summary.loc[summary["turn_type"].eq("straight"), "track_id"].astype(int))
        res: list[float] = []
        for tid, g in obs[obs["track_id"].isin(straight)].groupby("track_id"):
            p = g[["world_x", "world_y"]].dropna().to_numpy(float)
            if len(p) < 8:
                continue
            c = p - p.mean(axis=0)
            # First principal component = best-fit line; residual = lateral error.
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            lateral = c @ vt[1]
            res.append(float(np.sqrt(np.mean(lateral ** 2))))
        if res:
            out["straight_path_rms_lateral_m"] = round(float(np.median(res)), 3)
            out["straight_tracks_measured"] = len(res)
    return out
