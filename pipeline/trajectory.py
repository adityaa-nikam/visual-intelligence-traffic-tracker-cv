"""
Trajectory data model - the reusable core of the whole system.

Everything downstream (counts, speeds, queues, congestion, conflicts, anomalies,
and whatever later levels ask for) reads from the trajectory table produced here.
Detection and tracking never appear again after this module, which is what makes
new analytics cheap to add.

An object's position starts as the BOTTOM-CENTRE of its bounding box, and is then
optionally refined to the true tyre-contact line of a MobileSAM instance mask:

        +--------------+          box bottom-centre (x, y)
        |     CAR      |            v from the lowest silhouette pixel,
        |     #17      |            u from a ROOF-CONTAMINATED average
        +------o-------+          -> corresponds to no physical feature
           \\......../            mask contact band (x_seg, y_seg)
            `--oo--'               the actual tyre line

Why the box point is biased here: the gimbal is at -63 deg pitch, so the frame
spans 4 deg off-nadir at the bottom edge to 50 deg at the top, and an elevated
point projects displaced radially by `height * tan(off-nadir)`. The bias therefore
GROWS ~17x up the frame and depends on vehicle orientation, so it does not cancel -
it injects false velocity. See pipeline/contact_points.py for the full derivation
and the measured result.

Per-observation schema written to outputs/trajectories.csv:
    track_id, class, frame, timestamp, x, y, confidence,
    world_x, world_y, footprint_m,
    x_seg, y_seg, footprint_seg_m, contact_source,   mask refinement (see below)
    sx, sy,                       smoothed position (metres if calibrated)
    speed_kph, speed_px_s, bearing_deg, heading_px_deg, accel_kph_s
`world_*`, `speed_kph`, `footprint_m` and `bearing_deg` are only populated when
telemetry calibration succeeded; otherwise they are left empty and the pipeline
falls back to pixel-space kinematics rather than inventing metric values.

BOTH contact points are kept. `x, y` always hold the box bottom-centre; `x_seg,
y_seg` hold the mask contact point where one survived vetting; `contact_source`
records per observation which was used. `segmentation.contact.use` decides which
feeds `world_x/world_y`, so the refinement is measurable, reversible, and degrades
to the box point rather than to a fabricated value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OBS_COLUMNS = [
    "track_id",
    "class",
    "frame",
    "timestamp",
    "x",
    "y",
    "confidence",
    "world_x",
    "world_y",
    "footprint_m",
    "box_w",
    "box_h",
    # Mask-refined contact point. Populated by apply_contact_refinement(); left as
    # NaN / "bbox" when segmentation is off or no mask survived vetting.
    "x_seg",
    "y_seg",
    "footprint_seg_m",
    "contact_source",
]


class TrajectoryStore:
    """Accumulates one observation row per tracked object per frame."""

    def __init__(self, projector=None):
        self.projector = projector
        self._rows: list[dict] = []

    def add_frame(self, frame: int, t: float, tracks: list[dict]) -> None:
        if not tracks:
            return

        # Bottom-centre plus the two bottom corners; the corners give us the
        # vehicle's ground footprint width, used for the LGV/HGV size split.
        pts = []
        for tr in tracks:
            x1, y1, x2, y2 = tr["xyxy"]
            pts.append([(x1 + x2) / 2.0, y2])
            pts.append([x1, y2])
            pts.append([x2, y2])

        ground = None
        if self.projector is not None:
            ground = self.projector.pixels_to_ground(np.array(pts))

        for i, tr in enumerate(tracks):
            x1, y1, x2, y2 = tr["xyxy"]
            row = {
                "track_id": tr["track_id"],
                "class": tr["cls"],
                "frame": frame,
                "timestamp": round(t, 3),
                "x": round(float((x1 + x2) / 2.0), 2),
                "y": round(float(y2), 2),
                "confidence": round(float(tr["conf"]), 3),
                "world_x": np.nan,
                "world_y": np.nan,
                "footprint_m": np.nan,
                "box_w": round(float(x2 - x1), 1),
                "box_h": round(float(y2 - y1), 1),
                "x_seg": np.nan,
                "y_seg": np.nan,
                "footprint_seg_m": np.nan,
                "contact_source": "bbox",
            }
            if ground is not None:
                centre, left, right = ground[3 * i], ground[3 * i + 1], ground[3 * i + 2]
                if not np.isnan(centre).any():
                    row["world_x"] = round(float(centre[0]), 3)
                    row["world_y"] = round(float(centre[1]), 3)
                if not (np.isnan(left).any() or np.isnan(right).any()):
                    row["footprint_m"] = round(float(np.linalg.norm(right - left)), 2)
            self._rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=OBS_COLUMNS)
        return pd.DataFrame(self._rows, columns=OBS_COLUMNS)


def contact_use_seg(cfg: dict) -> bool:
    """Does config ask for the mask contact point to drive the kinematics?"""
    ccfg = (cfg.get("segmentation", {}) or {}).get("contact", {}) or {}
    return str(ccfg.get("use", "seg")).lower() == "seg"


def apply_contact_refinement(
    raw: pd.DataFrame,
    refined: pd.DataFrame,
    projector=None,
    use: str = "seg",
    fill_gaps: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Merge MobileSAM contact points into the raw observations and re-project.

    MUST run before `compute_kinematics`, because speed and bearing are derived
    from position - refining afterwards would leave the kinematics describing the
    old anchor.

    `fill_gaps` interpolates the box->mask OFFSET along each track, never the
    absolute position. The offset is smooth in image position and vehicle
    orientation; the position is not. This covers two cases:

      * `segmentation.contact.stride > 1`, where most frames were never segmented;
      * individual frames where the mask was rejected by `_vet_mask`.

    The second matters more than it looks. Snapping a single rejected frame back to
    the box point would reintroduce a 0.1-1.8 m discontinuity in the middle of an
    otherwise refined track - and a discontinuity is precisely what a centred
    difference turns into a phantom speed spike. Mixing the two anchors within one
    track is worse than either anchor used consistently, so gaps are bridged and
    tagged `mask_interpolated` rather than reverted.

    A track with NO accepted mask anywhere keeps the box point throughout and stays
    tagged `bbox`. Returns (observations, stats).
    """
    for c in ("x_seg", "y_seg", "footprint_seg_m"):
        if c not in raw.columns:
            raw[c] = np.nan
    if "contact_source" not in raw.columns:
        raw["contact_source"] = "bbox"

    if refined is None or refined.empty:
        return raw, {"applied": False, "reason": "no refined contact points"}

    keep = [c for c in ("frame", "track_id", "x_seg", "y_seg", "seg_l_x", "seg_r_x")
            if c in refined.columns]
    raw = raw.drop(columns=[c for c in ("x_seg", "y_seg", "seg_l_x", "seg_r_x")
                            if c in raw.columns])
    raw = raw.merge(refined[keep], on=["frame", "track_id"], how="left")
    raw = raw.sort_values(["track_id", "frame"]).reset_index(drop=True)

    n_direct = int(raw["x_seg"].notna().sum())
    raw["contact_source"] = np.where(raw["x_seg"].notna(), "mask", "bbox")

    n_interp = 0
    if fill_gaps:
        raw["_dx"] = raw["x_seg"] - raw["x"]
        raw["_dy"] = raw["y_seg"] - raw["y"]
        raw["_w"] = raw["seg_r_x"] - raw["seg_l_x"]
        g = raw.groupby("track_id", sort=False)
        for c in ("_dx", "_dy", "_w"):
            raw[c] = g[c].transform(lambda s: s.interpolate(limit_direction="both"))
        fill = raw["x_seg"].isna() & raw["_dx"].notna() & raw["_dy"].notna()
        n_interp = int(fill.sum())
        if n_interp:
            raw.loc[fill, "x_seg"] = raw.loc[fill, "x"] + raw.loc[fill, "_dx"]
            raw.loc[fill, "y_seg"] = raw.loc[fill, "y"] + raw.loc[fill, "_dy"]
            half = raw.loc[fill, "_w"] / 2.0
            raw.loc[fill, "seg_l_x"] = raw.loc[fill, "x_seg"] - half
            raw.loc[fill, "seg_r_x"] = raw.loc[fill, "x_seg"] + half
            raw.loc[fill, "contact_source"] = "mask_interpolated"
        raw = raw.drop(columns=["_dx", "_dy", "_w"])

    stats = {
        "applied": use == "seg",
        "use": use,
        "observations": int(len(raw)),
        "from_mask": n_direct,
        "from_interpolated_offset": n_interp,
        "kept_bbox": int((raw["contact_source"] == "bbox").sum()),
        "note": (
            "x,y always keep the box bottom-centre so the improvement stays "
            "measurable; world_x/world_y follow segmentation.contact.use"
        ),
    }

    if use != "seg":
        stats["outcome"] = "contact points recorded but NOT used (use: bbox)"
        return raw, stats

    # Re-project the refined point through the SAME GroundProjector the box point
    # used, so nothing about the calibration changes - only the pixel we ask about.
    if projector is not None:
        sel = (raw["x_seg"].notna() & raw["y_seg"].notna()).to_numpy()
        idx = np.flatnonzero(sel)
        if idx.size:
            sub = raw.loc[sel]
            xs = sub["x_seg"].to_numpy(float)
            ys = sub["y_seg"].to_numpy(float)
            lx = sub["seg_l_x"].fillna(sub["x_seg"]).to_numpy(float)
            rx = sub["seg_r_x"].fillna(sub["x_seg"]).to_numpy(float)

            # One batched projection: centre, contact-line left, contact-line right.
            pts = np.empty((idx.size * 3, 2), float)
            pts[0::3, 0], pts[0::3, 1] = xs, ys
            pts[1::3, 0], pts[1::3, 1] = lx, ys
            pts[2::3, 0], pts[2::3, 1] = rx, ys
            gnd = projector.pixels_to_ground(pts)

            centre, left, right = gnd[0::3], gnd[1::3], gnd[2::3]
            ok_c = np.isfinite(centre).all(axis=1)
            if ok_c.any():
                raw.loc[idx[ok_c], "world_x"] = centre[ok_c, 0]
                raw.loc[idx[ok_c], "world_y"] = centre[ok_c, 1]
            ok_w = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
            if ok_w.any():
                width = np.linalg.norm(right[ok_w] - left[ok_w], axis=1)
                raw.loc[idx[ok_w], "footprint_seg_m"] = np.round(width, 2)
                # The mask contact line is a better footprint than two box corners,
                # so it also becomes the footprint the LGV/HGV split reads.
                raw.loc[idx[ok_w], "footprint_m"] = np.round(width, 2)
            stats["world_reprojected"] = int(ok_c.sum())
            stats["footprint_from_contact_line"] = int(ok_w.sum())
    else:
        stats["world_reprojected"] = 0
        stats["note_uncalibrated"] = (
            "no metric calibration; the refined point still drives pixel-space "
            "kinematics via compute_kinematics"
        )
    return raw, stats


def _bearing_from_delta(dx: pd.Series, dy: pd.Series) -> pd.Series:
    """Compass bearing (0 = north, 90 = east) from an east/north displacement."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def compute_kinematics(df: pd.DataFrame, fps: float, cfg: dict, calibrated: bool) -> pd.DataFrame:
    """
    Add smoothed positions, speed, heading and acceleration to the observations.

    Raw frame-to-frame differences on a bounding-box corner are far too noisy to
    quote as a speed, so positions are smoothed with a centred moving average and
    speed is a centred difference across a ~0.5 s baseline.
    """
    if df.empty:
        for c in ("sx", "sy", "speed_kph", "speed_px_s", "bearing_deg", "heading_px_deg", "accel_kph_s"):
            df[c] = pd.Series(dtype=float)
        return df

    tcfg = cfg["trajectory"]
    win = max(1, int(tcfg.get("smooth_window", 9)))
    stride = max(1, int(cfg["video"].get("frame_stride", 1)))
    eff_fps = fps / stride
    lag = max(1, int(round(float(tcfg.get("speed_window_s", 0.5)) * eff_fps)))
    half = max(1, lag // 2)

    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    # Drop flicker tracks: too short to yield a trustworthy speed or heading.
    min_frames = int(tcfg.get("min_track_frames", 8))
    counts = df.groupby("track_id")["frame"].transform("size")
    df = df[counts >= min_frames].reset_index(drop=True)
    if df.empty:
        for c in ("sx", "sy", "speed_kph", "speed_px_s", "bearing_deg", "heading_px_deg", "accel_kph_s"):
            df[c] = pd.Series(dtype=float)
        return df

    # Pixel-space smoothing is always available. It smooths the CHOSEN contact
    # anchor: the mask point where one exists and config asks for it, otherwise the
    # box bottom-centre. `x`/`y` themselves are never overwritten, so the box
    # baseline survives in the CSV for the A/B comparison.
    xsrc, ysrc = "x", "y"
    if contact_use_seg(cfg) and "x_seg" in df.columns and df["x_seg"].notna().any():
        df["_cx"] = df["x_seg"].where(df["x_seg"].notna(), df["x"])
        df["_cy"] = df["y_seg"].where(df["y_seg"].notna(), df["y"])
        xsrc, ysrc = "_cx", "_cy"

    g = df.groupby("track_id", sort=False)

    df["px_s"] = g[xsrc].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    df["py_s"] = g[ysrc].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())

    if calibrated:
        df["sx"] = g["world_x"].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
        df["sy"] = g["world_y"].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    else:
        df["sx"] = np.nan
        df["sy"] = np.nan

    g = df.groupby("track_id", sort=False)

    def centred_delta(col: str) -> pd.Series:
        return g[col].shift(-half) - g[col].shift(half)

    dt = centred_delta("timestamp")
    dt = dt.where(dt > 1e-6)

    # Pixel velocity: always reported, needs no calibration.
    dpx, dpy = centred_delta("px_s"), centred_delta("py_s")
    df["speed_px_s"] = np.hypot(dpx, dpy) / dt
    df["heading_px_deg"] = (np.degrees(np.arctan2(dpx, -dpy)) + 360.0) % 360.0

    if calibrated:
        dx, dy = centred_delta("sx"), centred_delta("sy")
        df["speed_kph"] = (np.hypot(dx, dy) / dt) * 3.6
        df["bearing_deg"] = _bearing_from_delta(dx, dy)
    else:
        df["speed_kph"] = np.nan
        df["bearing_deg"] = np.nan

    # Fill the window edges from within each track so no observation is blank.
    g = df.groupby("track_id", sort=False)
    for col in ("speed_px_s", "heading_px_deg", "speed_kph", "bearing_deg"):
        df[col] = g[col].transform(lambda s: s.ffill().bfill())

    # Acceleration, used for the sudden-stop detector.
    g = df.groupby("track_id", sort=False)
    if calibrated:
        dv = g["speed_kph"].diff()
        dtt = g["timestamp"].diff().where(lambda s: s > 1e-6)
        df["accel_kph_s"] = (dv / dtt).rolling(win, center=True, min_periods=1).mean()
    else:
        df["accel_kph_s"] = np.nan

    df = df.drop(columns=[c for c in ("px_s", "py_s", "_cx", "_cy") if c in df.columns])
    return df


def resolve_track_classes(df: pd.DataFrame, cfg: dict, calibrated: bool) -> pd.DataFrame:
    """
    Decide one class per track (majority vote) and apply the LGV/HGV size split.

    Majority voting over a track's whole life is much more stable than trusting
    any single frame, and it is what makes the counts double-count-free.

    The LGV/HGV split is a CALIBRATION-DERIVED SIZE HEURISTIC applied only to the
    model's `truck` detections - COCO has no LGV/HGV classes. Without telemetry
    the split is skipped entirely and trucks stay labelled `truck`.
    """
    if df.empty:
        return pd.DataFrame(columns=["track_id", "class", "class_source", "footprint_p90_m"])

    votes = (
        df.groupby(["track_id", "class"]).size().rename("n").reset_index()
        .sort_values(["track_id", "n"], ascending=[True, False])
        .drop_duplicates("track_id")[["track_id", "class"]]
    )
    votes["class_source"] = "model"

    fp = (
        df.groupby("track_id")["footprint_m"].quantile(0.9).rename("footprint_p90_m").reset_index()
        if calibrated
        else pd.DataFrame({"track_id": votes["track_id"], "footprint_p90_m": np.nan})
    )
    out = votes.merge(fp, on="track_id", how="left")

    if calibrated:
        split = float(cfg["analytics"].get("lgv_hgv_split_m", 7.0))
        is_truck = out["class"].eq("truck") & out["footprint_p90_m"].notna()
        out.loc[is_truck & (out["footprint_p90_m"] >= split), "class"] = "hgv"
        out.loc[is_truck & (out["footprint_p90_m"] < split), "class"] = "lgv"
        out.loc[is_truck, "class_source"] = "size_heuristic_from_calibration"

    return out


def apply_track_classes(df: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """Replace the noisy per-frame class with the resolved per-track class."""
    if df.empty:
        return df
    df = df.rename(columns={"class": "class_raw"}).merge(
        classes[["track_id", "class", "class_source"]], on="track_id", how="left"
    )
    df["class"] = df["class"].fillna(df["class_raw"])
    return df
