"""
Road-area segmentation.

Why this is cheap despite SAM being a heavy model: the drone HOVERS. Measured on
this footage, across 6824 telemetry records, relative altitude varies by 0.023 m,
gimbal pitch by 0.0 deg and yaw by 0.4 deg. The scene is therefore static in image
space, so the road mask is computed from a SINGLE frame and reused for every frame
of the clip. A one-off 40 MB MobileSAM pass costs seconds, not hours.

Two independent estimates are produced and then combined:

1. `sam_mask`        - geometric. MobileSAM segments the frame; we keep the
                       components that the traffic actually travels through.
2. `travelled_mask`  - empirical. Rasterise every trajectory ground-contact point
                       we observed and dilate by a lane width. This is where road
                       users demonstrably drove.

Neither alone is trustworthy. SAM does not know which of its segments is road, and
the travelled mask only covers road that traffic already used. Their AGREEMENT is a
high-confidence drivable area, and their disagreement is itself informative
(SAM-only = road nobody used this minute; travelled-only = SAM under-segmented).

The mask is reported with a measured agreement score rather than asserted correct.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd


def _largest_components(mask: np.ndarray, min_area_frac: float = 0.005) -> np.ndarray:
    """Drop specks; keep components covering at least `min_area_frac` of the frame."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    min_area = min_area_frac * mask.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill all internal holes in a binary mask."""
    if mask is None or not np.any(mask):
        return mask
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def smooth_mask(mask: np.ndarray, close_k: int = 31, blur_k: int = 15) -> np.ndarray:
    """Smooth jagged mask boundaries using morphological closing, hole-filling, and Gaussian filtering."""
    if mask is None or not np.any(mask):
        return mask
    m = fill_holes(mask)
    if close_k > 0:
        k_size = max(3, int(close_k) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = fill_holes(m)
    if blur_k > 0:
        b_size = max(3, int(blur_k) | 1)
        blurred = cv2.GaussianBlur(m, (b_size, b_size), 0)
        _, m = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
        m = fill_holes(m)
    return _largest_components(m)


def travelled_mask(
    obs: pd.DataFrame,
    shape: tuple[int, int],
    lane_width_px: int = 40,
) -> np.ndarray:
    """
    Empirical drivable area: where road users were actually observed.

    Connects consecutive observations into continuous trajectory corridors,
    dilates by a realistic lane width, fills internal holes, and smooths boundaries.
    """
    h, w = shape
    canvas = np.zeros((h, w), np.uint8)
    if obs is None or obs.empty:
        return canvas

    xcol = "x_seg" if ("x_seg" in obs.columns and obs["x_seg"].notna().any()) else "x"
    ycol = "y_seg" if ("y_seg" in obs.columns and obs["y_seg"].notna().any()) else "y"

    pts_df = obs.dropna(subset=[xcol, ycol])
    if pts_df.empty:
        return canvas

    lane_r = max(20, int(lane_width_px))
    for tid, group in pts_df.groupby("track_id"):
        g_pts = group.sort_values("frame")[[xcol, ycol]].to_numpy(np.int32)
        if len(g_pts) > 1:
            cv2.polylines(canvas, [g_pts], False, 255, thickness=lane_r * 2, lineType=cv2.LINE_AA)
        for pt in g_pts:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), lane_r, 255, -1, lineType=cv2.LINE_AA)

    close_k = max(35, int(lane_r * 2.5) | 1)
    return smooth_mask(canvas, close_k=close_k, blur_k=21)


def _seed_points(
    trav: np.ndarray,
    occupied: np.ndarray | None,
    max_seeds: int,
    grid: int = 96,
) -> np.ndarray:
    """
    Choose SAM prompt points that are actually ON BARE ROAD.

    This is the subtle part. The obvious move - prompt SAM at the trajectory
    points - does the wrong thing: a trajectory point sits on a VEHICLE, and SAM
    is class-agnostic, so it dutifully returns the car. Measured on this footage
    that gave exactly one segment per prompt and only 35% overlap with the
    travelled area: SAM had segmented 24 cars.

    Two corrections:
      * drop candidates covered by a detection box in the segmented frame, so
        every prompt lands on empty carriageway. The hover means road a vehicle
        drove over at t=30s is bare tarmac at t=0.
      * pick at most one candidate per grid cell, so prompts spread across every
        arm of the junction instead of clustering in the busiest lane.
    """
    ys, xs = np.nonzero(trav)
    if len(xs) == 0:
        return np.empty((0, 2), np.int32)

    if occupied is not None:
        free = occupied[ys, xs] == 0
        # Only apply the filter if it leaves us something to work with.
        if free.sum() >= max_seeds:
            ys, xs = ys[free], xs[free]

    cells: dict[tuple[int, int], tuple[int, int]] = {}
    for x, y in zip(xs, ys):
        key = (int(y) // grid, int(x) // grid)
        if key not in cells:
            cells[key] = (int(x), int(y))

    pts = np.array(list(cells.values()), np.int32)
    if len(pts) > max_seeds:
        idx = np.linspace(0, len(pts) - 1, max_seeds).astype(int)
        pts = pts[idx]
    return pts


def _occupied_mask(boxes: pd.DataFrame, frame_index: int, shape: tuple[int, int],
                   pad: int = 6, vehicle_mask: np.ndarray | None = None) -> np.ndarray | None:
    """
    Pixels a road user occupies in the frame being segmented, so seeds avoid them.

    A detection BOX is a crude stand-in: it excludes the vehicle plus a lot of the
    bare tarmac around it, which throws away seed candidates. When the contact-point
    pass has already produced instance masks for this frame
    (`contact_points.refine_contacts(capture_frame=...)`), those masks are used
    instead - they cost nothing extra here and free up roughly the difference
    between a box and the silhouette inside it.

    Boxes are still excluded individually where no mask covers them, so an object
    MobileSAM declined to segment cannot quietly become a seed.
    """
    if boxes is None or boxes.empty or "frame" not in boxes:
        return None
    g = boxes[boxes["frame"] == frame_index]
    if g.empty and vehicle_mask is None:
        return None
    h, w = shape
    occ = np.zeros((h, w), np.uint8)

    vm = None
    if vehicle_mask is not None and np.any(vehicle_mask):
        vm = vehicle_mask
        if vm.shape[:2] != (h, w):
            vm = cv2.resize(vm, (w, h), interpolation=cv2.INTER_NEAREST)
        vm = vm > 0
        occ[vm] = 255
        # Grow slightly so a seed cannot land on a bumper just outside the silhouette.
        k = max(3, int(pad) * 2 + 1)
        occ = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    for r in g.itertuples(index=False):
        x1 = max(int(r.x1) - pad, 0)
        y1 = max(int(r.y1) - pad, 0)
        x2 = min(int(r.x2) + pad, w - 1)
        y2 = min(int(r.y2) + pad, h - 1)
        if vm is not None and vm[max(int(r.y1), 0):max(int(r.y2), 0),
                                max(int(r.x1), 0):max(int(r.x2), 0)].any():
            continue                          # already excluded precisely by its mask
        occ[y1:y2, x1:x2] = 255
    return occ


def sam_segments(
    frame: np.ndarray,
    seeds: np.ndarray,
    weights: str = "mobile_sam.pt",
) -> tuple[list[np.ndarray], str]:
    """
    Run MobileSAM at the given prompt points and return the raw segments.

    Deliberately returns the segments SEPARATELY rather than one merged mask, so
    the caller can vet each one. SAM is class-agnostic: it will happily return a
    rooftop or a tree canopy that looks like tarmac, and merging first makes that
    impossible to undo.

    Returns ([] , note) if the model is unavailable - the caller then falls back to
    the empirical mask rather than failing.
    """
    try:
        from ultralytics import SAM
    except Exception as exc:                       # pragma: no cover
        return [], f"ultralytics SAM unavailable: {exc}"

    if len(seeds) == 0:
        return [], "no usable bare-road seed points"

    try:
        model = SAM(weights)
        res = model(frame, points=seeds.tolist(), labels=[1] * len(seeds), verbose=False)
    except Exception as exc:
        return [], f"SAM inference failed: {exc}"

    h, w = frame.shape[:2]
    segs: list[np.ndarray] = []
    for r in res:
        if r.masks is None:
            continue
        for m in r.masks.data.cpu().numpy():
            mm = (m > 0.5).astype(np.uint8) * 255
            if mm.shape != (h, w):
                mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
            frac = mm.mean() / 255.0
            # Degenerate segments: one covering almost the whole frame grabbed the
            # entire image; a tiny one is a single vehicle or a road marking.
            if frac > 0.85 or frac < 0.0005:
                continue
            segs.append(mm)

    if not segs:
        return [], "SAM returned no usable masks"
    return segs, f"MobileSAM, {len(seeds)} bare-road prompts, {len(segs)} raw segments"


def vet_segments(
    segs: list[np.ndarray],
    trav: np.ndarray,
    min_overlap: float = 0.45,
) -> tuple[np.ndarray, dict]:
    """
    Keep only SAM segments that the observed traffic vouches for.

    SAM PROPOSES, the trajectory data DECIDES. A segment is accepted when at least
    `min_overlap` of its own area falls inside the travelled mask - i.e. it is the
    surface traffic demonstrably drove on, extended outward to its true edges
    (empty lanes, unused arms). A rooftop or tree canopy overlaps the travelled
    area at roughly zero and is rejected.

    This asymmetry is deliberate: an over-inclusive road mask is worse than a
    conservative one, because "density per square metre of road" becomes
    meaningless once rooftops are in the denominator.
    """
    keep = np.zeros_like(trav)
    tb = trav > 0
    accepted = rejected = 0
    for s in segs:
        sb = s > 0
        area = int(sb.sum())
        if area == 0:
            continue
        overlap = float((sb & tb).sum()) / area
        if overlap >= min_overlap:
            keep[sb] = 255
            accepted += 1
        else:
            rejected += 1
    return keep, {"accepted": accepted, "rejected": rejected,
                  "min_overlap_with_travelled": min_overlap}


def build_road_mask(
    obs: pd.DataFrame,
    frame: np.ndarray,
    cfg: dict,
    boxes: pd.DataFrame | None = None,
    frame_index: int = 0,
    vehicle_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Combine the empirical and geometric estimates into one drivable-area mask.

    `vehicle_mask` is the optional union of the instance masks the contact-point
    pass already produced for `frame_index` - if supplied, seeds avoid vehicle
    PIXELS rather than whole detection boxes. Returns (mask_uint8, info). `info`
    carries the measured agreement between the two independent estimates, so the
    mask's quality is reported rather than assumed.
    """
    scfg = cfg.get("segmentation", {}) or {}
    h, w = frame.shape[:2]

    lane_px = int(scfg.get("lane_width_px", 26))
    trav = travelled_mask(obs, (h, w), lane_width_px=lane_px)
    trav_frac = float((trav > 0).mean())

    info = {
        "enabled": True,
        "static_scene_justification": (
            "drone hovers (altitude spread 0.023 m, pitch 0.0 deg, yaw 0.4 deg over "
            "6824 telemetry records), so one mask serves every frame"
        ),
        "travelled_area_frac": round(trav_frac, 4),
        "sam": None,
        "agreement": None,
        "source": "travelled_only",
    }

    sam = None
    if scfg.get("use_sam", True):
        occ = _occupied_mask(boxes, frame_index, (h, w), vehicle_mask=vehicle_mask)
        seeds = _seed_points(
            trav, occ, max_seeds=int(scfg.get("sam_max_seeds", 24)),
            grid=int(scfg.get("seed_grid_px", 96)),
        )
        info["seeds"] = {
            "count": int(len(seeds)),
            "excluded_occupied_pixels": occ is not None,
            "exclusion_source": (
                "instance masks from the contact-point pass, boxes where no mask"
                if vehicle_mask is not None and np.any(vehicle_mask) else "detection boxes"
            ),
            "note": "prompts placed on bare carriageway, not on vehicles - see road_mask._seed_points",
        }
        segs, note = sam_segments(frame, seeds, weights=scfg.get("sam_weights", "mobile_sam.pt"))
        info["sam"] = note
        if segs:
            sam, vet = vet_segments(
                segs, trav, min_overlap=float(scfg.get("min_segment_overlap", 0.45))
            )
            info["segment_vetting"] = vet
            if not sam.any():
                sam = None
                info["segment_vetting"]["outcome"] = "every segment rejected -> travelled mask only"

    if sam is None:
        # Honest degradation: empirical mask only, and say so.
        return trav, info

    inter = cv2.bitwise_and(trav, sam)
    union = cv2.bitwise_or(trav, sam)
    iou = float((inter > 0).sum() / max((union > 0).sum(), 1))
    # How much of the travelled area SAM also called road. This is the number that
    # matters: if SAM missed road that traffic demonstrably used, it under-segmented.
    recall = float((inter > 0).sum() / max((trav > 0).sum(), 1))
    info["agreement"] = {
        "iou": round(iou, 3),
        "sam_covers_travelled_frac": round(recall, 3),
        "sam_area_frac": round(float((sam > 0).mean()), 4),
    }

    # Union, not intersection: we want road SAM found that was simply unused this
    # minute (empty lanes, far arms) included, while the travelled mask guarantees
    # we never lose road we have direct evidence for.
    combined = smooth_mask(cv2.bitwise_or(trav, sam), close_k=25, blur_k=15)
    info["source"] = "sam_union_travelled"
    info["road_area_frac"] = round(float((combined > 0).mean()), 4)
    return combined, info


def mask_ground_area_m2(mask: np.ndarray, projector) -> float | None:
    """
    Physical area of the mask in square metres, via the ground projection.

    Sampled on a coarse grid: each sampled pixel contributes its own local
    metres-per-pixel scale squared, because an oblique camera means scale varies
    across the frame (this one is at -63 deg pitch, not nadir).
    """
    if projector is None:
        return None
    h, w = mask.shape[:2]
    step = max(8, min(h, w) // 120)
    total = 0.0
    for v in range(0, h, step):
        for u in range(0, w, step):
            if mask[v, u] == 0:
                continue
            s = projector.ground_scale_at(u, v)
            if s is None or not np.isfinite(s):
                continue
            total += (s * step) ** 2
    return round(total, 1)


def filter_to_road(obs: pd.DataFrame, mask: np.ndarray, classes_exempt=("pedestrian",)) -> tuple[pd.DataFrame, dict]:
    """
    Drop observations whose road-contact point falls outside the drivable area.

    Pedestrians are exempt by default: footpaths are legitimately off-carriageway,
    so masking them out would delete real vulnerable road users - exactly the
    opposite of what a safety system should do.
    """
    if obs.empty or mask is None:
        return obs, {"applied": False}

    h, w = mask.shape[:2]
    xi = np.clip(obs["x"].to_numpy(), 0, w - 1).astype(np.int32)
    yi = np.clip(obs["y"].to_numpy(), 0, h - 1).astype(np.int32)
    on_road = mask[yi, xi] > 0
    exempt = obs["class"].isin(classes_exempt).to_numpy()
    keep = on_road | exempt

    dropped_tracks = sorted(set(obs.loc[~keep, "track_id"]) - set(obs.loc[keep, "track_id"]))
    stats = {
        "applied": True,
        "observations_before": int(len(obs)),
        "observations_kept": int(keep.sum()),
        "observations_dropped": int((~keep).sum()),
        "tracks_dropped_entirely": len(dropped_tracks),
        "exempt_classes": list(classes_exempt),
        "note": "off-road detections removed; pedestrians exempt (footpaths are legitimately off-carriageway)",
    }
    return obs[keep].reset_index(drop=True), stats


def render_mask_overlay(frame: np.ndarray, mask: np.ndarray, out_path: str) -> str:
    """Save a visual proof image: the frame with the drivable area tinted."""
    over = frame.copy()
    tint = np.zeros_like(frame)
    tint[mask > 0] = (0, 150, 60)
    over = cv2.addWeighted(over, 1.0, tint, 0.35, 0)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (60, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(over, "drivable area (MobileSAM + observed trajectories)",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 120), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, over)
    return out_path
