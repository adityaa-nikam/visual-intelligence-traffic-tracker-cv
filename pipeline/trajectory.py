"""
Trajectory data model - the reusable core of the whole system.

Everything downstream (counts, speeds, queues, congestion, conflicts, anomalies,
and whatever later levels ask for) reads from the trajectory table produced here.
Detection and tracking never appear again after this module, which is what makes
new analytics cheap to add.

An object's position is the BOTTOM-CENTRE of its bounding box - the point where
it contacts the road:

        +--------------+
        |     CAR      |
        |     #17      |
        +------o-------+
               ^ road-contact point

Per-observation schema written to outputs/trajectories.csv:
    track_id, class, frame, timestamp, x, y, confidence,
    world_x, world_y, footprint_m,
    sx, sy,                       smoothed position (metres if calibrated)
    speed_kph, speed_px_s, bearing_deg, heading_px_deg, accel_kph_s
`world_*`, `speed_kph`, `footprint_m` and `bearing_deg` are only populated when
telemetry calibration succeeded; otherwise they are left empty and the pipeline
falls back to pixel-space kinematics rather than inventing metric values.
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
    "depth_m",
    "box_w",
    "box_h",
]


class TrajectoryStore:
    """Accumulates one observation row per tracked object per frame."""

    def __init__(self, projector=None):
        self.projector = projector
        self._rows: list[dict] = []

    def add_frame(self, frame: int, t: float, tracks: list[dict]) -> None:
        if not tracks:
            return

        # Four probe points per box:
        #   bottom-centre  -> the road-contact position
        #   bottom-left/right -> the ground footprint ACROSS the box
        #   top-centre     -> the ground extent ALONG the viewing direction
        # The bottom edge lies in the road plane, so its projection is exact. The
        # top-centre probe is the box's far edge projected onto the same plane; for
        # a tall vehicle it lands beyond the real bodywork, which is why the pair is
        # solved for length and width in classify.py rather than read off directly.
        pts = []
        for tr in tracks:
            x1, y1, x2, y2 = tr["xyxy"]
            cx = (x1 + x2) / 2.0
            pts.append([cx, y2])
            pts.append([x1, y2])
            pts.append([x2, y2])
            pts.append([cx, y1])

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
                "depth_m": np.nan,
                "box_w": round(float(x2 - x1), 1),
                "box_h": round(float(y2 - y1), 1),
            }
            if ground is not None:
                centre, left, right, top = ground[4 * i:4 * i + 4]
                if not np.isnan(centre).any():
                    row["world_x"] = round(float(centre[0]), 3)
                    row["world_y"] = round(float(centre[1]), 3)
                if not (np.isnan(left).any() or np.isnan(right).any()):
                    row["footprint_m"] = round(float(np.linalg.norm(right - left)), 2)
                if not (np.isnan(top).any() or np.isnan(centre).any()):
                    row["depth_m"] = round(float(np.linalg.norm(top - centre)), 2)
            self._rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=OBS_COLUMNS)
        return pd.DataFrame(self._rows, columns=OBS_COLUMNS)


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

    g = df.groupby("track_id", sort=False)

    # Pixel-space smoothing is always available.
    df["px_s"] = g["x"].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    df["py_s"] = g["y"].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())

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
        raw_accel = dv / dtt
        # Smooth inside each identity only. A global rolling window would mix
        # the final acceleration samples of one track into the next track.
        df["accel_kph_s"] = raw_accel.groupby(df["track_id"], sort=False).transform(
            lambda s: s.rolling(win, center=True, min_periods=1).mean()
        )
    else:
        df["accel_kph_s"] = np.nan

    df = df.drop(columns=["px_s", "py_s"])
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
    # Drop any columns a previous resolution pass already added, otherwise the
    # rename below produces two `class_raw` columns and the fillna gets a
    # DataFrame instead of a Series. This happens on --from-trajectories, where
    # the input CSV has already been through here once.
    df = df.drop(columns=[c for c in ("class_raw", "class_source") if c in df.columns])
    df = df.rename(columns={"class": "class_raw"}).merge(
        classes[["track_id", "class", "class_source"]], on="track_id", how="left"
    )
    df["class"] = df["class"].fillna(df["class_raw"])
    return df
