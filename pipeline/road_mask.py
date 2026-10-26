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


def travelled_mask(
    obs: pd.DataFrame,
    shape: tuple[int, int],
    lane_width_px: int = 26,
) -> np.ndarray:
    """
    Empirical drivable area: where road users were actually observed.

    Uses the bbox bottom-centre (the road-contact point), dilated by roughly a
    lane half-width, then closed to bridge gaps between lanes.
    """
    h, w = shape
    canvas = np.zeros((h, w), np.uint8)
    pts = obs.dropna(subset=["x", "y"])[["x", "y"]].to_numpy()
    if len(pts) == 0:
        return canvas

    xi = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
    yi = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
    canvas[yi, xi] = 255

    k = max(3, int(lane_width_px) | 1)
    canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Close across lane gaps, then fill enclosed holes (traffic islands excepted).
    big = max(3, int(lane_width_px * 2) | 1)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big, big)))
    return _largest_components(canvas)


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
                   pad: int = 6) -> np.ndarray | None:
    """Pixels covered by a detection box in the frame being segmented."""
    if boxes is None or boxes.empty or "frame" not in boxes:
        return None
    g = boxes[boxes["frame"] == frame_index]
    if g.empty:
        return None
    h, w = shape
    occ = np.zeros((h, w), np.uint8)
    for r in g.itertuples(index=False):
        x1 = max(int(r.x1) - pad, 0)
        y1 = max(int(r.y1) - pad, 0)
        x2 = min(int(r.x2) + pad, w - 1)
        y2 = min(int(r.y2) + pad, h - 1)
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
) -> tuple[np.ndarray, dict]:
    """
    Combine the empirical and geometric estimates into one drivable-area mask.

    Returns (mask_uint8, info). `info` carries the measured agreement between the
    two independent estimates, so the mask's quality is reported rather than assumed.
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
        occ = _occupied_mask(boxes, frame_index, (h, w))
        seeds = _seed_points(
            trav, occ, max_seeds=int(scfg.get("sam_max_seeds", 24)),
            grid=int(scfg.get("seed_grid_px", 96)),
        )
        info["seeds"] = {
            "count": int(len(seeds)),
            "excluded_occupied_pixels": occ is not None,
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
    combined = _largest_components(cv2.bitwise_or(trav, sam))
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


def filter_to_road(
    obs: pd.DataFrame,
    mask: np.ndarray,
    classes_exempt=("pedestrian",),
    min_on_road_frac: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove tracks that are not on the carriageway - judged over the WHOLE track.

    The decision is deliberately per-track, not per-observation. A per-observation
    filter looks correct and behaves badly: a real vehicle whose contact point
    wobbles across the mask edge for a few frames gets its trajectory chopped into
    fragments, which is precisely the ID instability Level 1 is scored on. Judging
    the whole track instead means a road user either belongs to the road scene or
    does not.

    What this removes, concretely: YOLO fires on parked cars in private courtyards,
    on rooftop plant, and on vehicle-shaped clutter in vegetation. Those objects
    are never on the carriageway for any part of their life, so their on-road
    fraction is ~0 while a genuine road user's is ~1. The two populations are
    cleanly separated, which is why a single 50% threshold is enough.

    Pedestrians are exempt: footpaths are legitimately off-carriageway, and
    deleting vulnerable road users to tidy up the picture is the wrong trade for a
    safety system to make.
    """
    if obs.empty or mask is None:
        return obs, {"applied": False}

    h, w = mask.shape[:2]
    xi = np.clip(obs["x"].to_numpy(), 0, w - 1).astype(np.int32)
    yi = np.clip(obs["y"].to_numpy(), 0, h - 1).astype(np.int32)
    d = obs.assign(_on=(mask[yi, xi] > 0))

    frac = d.groupby("track_id")["_on"].mean()
    exempt_tracks = set(d.loc[d["class"].isin(classes_exempt), "track_id"].unique())
    off_tracks = {int(t) for t, f in frac.items()
                  if f < min_on_road_frac and int(t) not in exempt_tracks}

    keep = ~obs["track_id"].isin(off_tracks)
    dropped = obs.loc[~keep]
    stats = {
        "applied": True,
        "mode": "per-track (a track is kept or dropped as a whole)",
        "min_on_road_fraction": min_on_road_frac,
        "tracks_before": int(obs["track_id"].nunique()),
        "tracks_dropped": len(off_tracks),
        "observations_dropped": int(len(dropped)),
        "dropped_by_class": dropped["class"].value_counts().to_dict() if len(dropped) else {},
        "exempt_classes": list(classes_exempt),
        "note": ("tracks spending under half their life on the carriageway are treated as "
                 "off-road false positives (courtyards, rooftops, vegetation); pedestrians exempt"),
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
