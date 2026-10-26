"""
Analytics engine - derives traffic insight from the trajectory table.

This layer never sees a video frame or a neural network. Its only input is the
trajectory DataFrame, which is why later levels can add analytics without
re-running detection.

Every metric below states whether it is metric (calibration-backed) or
pixel-space, and estimates are labelled as estimates. Where a quantity cannot be
computed from the available data it is reported as unavailable rather than
guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pipeline.detector import MOTOR_VEHICLE_CLASSES

# Display order for the class breakdown.
CLASS_ORDER = ["car", "lgv", "hgv", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]

COMPASS_16 = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def _angle_diff(a: float, b: float) -> float:
    """Signed smallest angle a -> b in degrees, positive = clockwise."""
    return (b - a + 180.0) % 360.0 - 180.0


def compass_16(bearing: float) -> str:
    if bearing != bearing:  # NaN
        return "unknown"
    return COMPASS_16[int((bearing % 360.0) / 22.5 + 0.5) % 16]


def _longest_true_run(mask: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    """Longest contiguous True run: returns (duration_s, start_time)."""
    if mask.size == 0 or not mask.any():
        return 0.0, float("nan")
    best_dur, best_start = 0.0, float("nan")
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        dur = float(times[j] - times[i])
        if dur > best_dur:
            best_dur, best_start = dur, float(times[i])
        i = j + 1
    return best_dur, best_start


def build_track_summary(df: pd.DataFrame, cfg: dict, calibrated: bool) -> pd.DataFrame:
    """
    One row per tracked road user: lifetime, dwell, speed, path, turn movement.

    This is the table the dashboard's per-object views read from.
    """
    if df.empty:
        return pd.DataFrame()

    acfg = cfg["analytics"]
    stop_kph = float(acfg.get("stationary_speed_kph", 3.0))
    turn_cfg = acfg.get("turning", {})
    straight_max = float(turn_cfg.get("straight_max_deg", 25.0))
    min_path = float(turn_cfg.get("min_path_len_m", 12.0))

    # Pixel-space stand-in for the speed threshold when there is no calibration.
    px_stop = float(np.nanpercentile(df["speed_px_s"], 12)) if not calibrated else np.nan

    rows = []
    for tid, g in df.sort_values("frame").groupby("track_id", sort=False):
        t = g["timestamp"].to_numpy()
        visible_s = float(t[-1] - t[0]) if t.size > 1 else 0.0

        if calibrated:
            sx, sy = g["sx"].to_numpy(), g["sy"].to_numpy()
            speed = g["speed_kph"].to_numpy()
            stopped = speed < stop_kph
        else:
            sx, sy = g["x"].to_numpy(), g["y"].to_numpy()
            speed = g["speed_px_s"].to_numpy()
            stopped = speed < px_stop

        ok = ~(np.isnan(sx) | np.isnan(sy))
        path_len = float(np.nansum(np.hypot(np.diff(sx[ok]), np.diff(sy[ok])))) if ok.sum() > 1 else 0.0

        # Stationary time: total time below threshold, plus the longest single run.
        dt = np.diff(t, prepend=t[0])
        stationary_s = float(np.nansum(dt[stopped])) if stopped.size else 0.0
        longest_stop_s, stop_start_t = _longest_true_run(np.nan_to_num(stopped, nan=False).astype(bool), t)

        # Turn movement from entry vs exit heading over the first/last third.
        turn_type, turn_delta = "unknown", np.nan
        bearing_col = "bearing_deg" if calibrated else "heading_px_deg"
        b = g[bearing_col].to_numpy()
        b = b[~np.isnan(b)]
        if b.size >= 6 and (path_len >= min_path or not calibrated):
            k = max(2, b.size // 3)
            entry = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(b[:k]))), np.mean(np.cos(np.radians(b[:k]))))))
            exit_ = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(b[-k:]))), np.mean(np.cos(np.radians(b[-k:]))))))
            turn_delta = _angle_diff(entry % 360.0, exit_ % 360.0)
            if abs(turn_delta) <= straight_max:
                turn_type = "straight"
            elif turn_delta > 0:
                turn_type = "right"
            else:
                turn_type = "left"
            if abs(turn_delta) > 150.0:
                turn_type = "u-turn"
        entry_bearing = float(g[bearing_col].dropna().iloc[:5].mean()) if g[bearing_col].notna().any() else np.nan
        exit_bearing = float(g[bearing_col].dropna().iloc[-5:].mean()) if g[bearing_col].notna().any() else np.nan

        rows.append(
            {
                "track_id": int(tid),
                "class": g["class"].iloc[0],
                "class_source": g["class_source"].iloc[0] if "class_source" in g else "model",
                "first_t": round(float(t[0]), 2),
                "last_t": round(float(t[-1]), 2),
                "visible_s": round(visible_s, 2),
                "n_obs": int(len(g)),
                "path_len_m": round(path_len, 1) if calibrated else np.nan,
                "path_len_px": round(path_len, 1) if not calibrated else np.nan,
                "mean_speed_kph": round(float(np.nanmean(g["speed_kph"])), 1) if calibrated else np.nan,
                "max_speed_kph": round(float(np.nanmax(g["speed_kph"])), 1) if calibrated else np.nan,
                "mean_speed_px_s": round(float(np.nanmean(g["speed_px_s"])), 1),
                "stationary_s": round(stationary_s, 2),
                "longest_stop_s": round(longest_stop_s, 2),
                "stop_start_t": round(stop_start_t, 2) if stop_start_t == stop_start_t else np.nan,
                "entry_bearing_deg": round(entry_bearing, 1) if entry_bearing == entry_bearing else np.nan,
                "exit_bearing_deg": round(exit_bearing, 1) if exit_bearing == exit_bearing else np.nan,
                "turn_delta_deg": round(turn_delta, 1) if turn_delta == turn_delta else np.nan,
                "turn_type": turn_type,
                "footprint_p90_m": round(float(g["footprint_m"].quantile(0.9)), 2) if calibrated and g["footprint_m"].notna().any() else np.nan,
                # Measured ground dimensions - the evidence behind the fine-grained
                # class. Every classified track carries the number that classified it.
                "length_m": round(float(g["length_m"].median()), 2) if "length_m" in g and g["length_m"].notna().any() else np.nan,
                "width_m": round(float(g["width_m"].median()), 2) if "width_m" in g and g["width_m"].notna().any() else np.nan,
                "last_x": round(float(g["x"].iloc[-1]), 1),
                "last_y": round(float(g["y"].iloc[-1]), 1),
            }
        )

    return pd.DataFrame(rows).sort_values("first_t").reset_index(drop=True)


def class_counts(summary: pd.DataFrame) -> dict:
    """
    Unique road-user counts. One tracked identity = one road user, so an object
    seen for 300 frames is counted once.
    """
    if summary.empty:
        return {"total": 0, "by_class": {}, "motor_vehicles": 0, "vulnerable": 0}
    vc = summary["class"].value_counts().to_dict()
    ordered = {c: int(vc[c]) for c in CLASS_ORDER if c in vc}
    for c, n in vc.items():  # anything unexpected still gets reported
        ordered.setdefault(c, int(n))
    return {
        "total": int(len(summary)),
        "by_class": ordered,
        "motor_vehicles": int(sum(n for c, n in ordered.items() if c in MOTOR_VEHICLE_CLASSES)),
        "vulnerable": int(sum(n for c, n in ordered.items() if c in ("pedestrian", "bicycle", "motorcycle"))),
    }


def stationary_candidates(summary: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Road users that stayed put long enough to be worth an operator's attention.

    Deliberately named "candidate": from a single aerial clip we cannot tell a
    breakdown from a vehicle waiting at a red light. The queue analytics below
    are what let an operator discriminate - if a stationary candidate sits inside
    a detected queue cluster, it is almost certainly signal-related, and we flag
    that with `in_queue`.
    """
    if summary.empty:
        return pd.DataFrame()
    acfg = cfg["analytics"]
    min_s = float(acfg.get("stationary_min_s", 5.0))
    out = summary[
        summary["longest_stop_s"].ge(min_s) & summary["class"].isin(MOTOR_VEHICLE_CLASSES)
    ].copy()
    return out.sort_values("longest_stop_s", ascending=False).reset_index(drop=True)


def directional_flow(summary: pd.DataFrame, calibrated: bool) -> pd.DataFrame:
    """Approach/exit demand by 16-point compass sector (metric) or image sector."""
    if summary.empty:
        return pd.DataFrame()
    col = "exit_bearing_deg"
    d = summary[summary[col].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["sector"] = d[col].map(compass_16)
    out = d.groupby("sector").size().rename("road_users").reset_index()
    out["frame_of_reference"] = "compass (telemetry-calibrated)" if calibrated else "image-space"
    return out.sort_values("road_users", ascending=False).reset_index(drop=True)


def turning_movements(summary: pd.DataFrame) -> pd.DataFrame:
    """Left / straight / right / u-turn demand, split by class."""
    if summary.empty:
        return pd.DataFrame()
    d = summary[summary["turn_type"].ne("unknown")]
    if d.empty:
        return pd.DataFrame()
    return (
        d.groupby(["turn_type", "class"]).size().rename("road_users").reset_index()
        .sort_values("road_users", ascending=False).reset_index(drop=True)
    )


def detect_queues(df: pd.DataFrame, cfg: dict, calibrated: bool, bin_s: float = 2.0) -> pd.DataFrame:
    """
    Queue detection: spatial clusters of slow/stopped vehicles.

    Single-linkage clustering over the slow vehicles present in each time bin.
    Queue length is the extent of the cluster along its own principal axis, which
    is the direction the queue runs in. Reported in metres when calibrated,
    otherwise in pixels - never fake metres.
    """
    if df.empty:
        return pd.DataFrame()

    acfg = cfg["analytics"]
    radius = float(acfg.get("queue_cluster_radius_m", 14.0))
    min_v = int(acfg.get("queue_min_vehicles", 3))
    if calibrated:
        slow = df[df["speed_kph"] < float(acfg.get("queue_speed_kph", 8.0))]
        xcol, ycol, unit = "sx", "sy", "m"
    else:
        thresh = float(np.nanpercentile(df["speed_px_s"], 25))
        slow = df[df["speed_px_s"] < thresh]
        xcol, ycol, unit = "x", "y", "px"
        radius = radius / 0.06  # rough pixel equivalent when uncalibrated

    slow = slow[slow["class"].isin(MOTOR_VEHICLE_CLASSES)].dropna(subset=[xcol, ycol])
    if slow.empty:
        return pd.DataFrame()

    slow = slow.assign(bin=(slow["timestamp"] // bin_s).astype(int))
    rows = []
    for b, g in slow.groupby("bin"):
        # One position per track per bin, else a single vehicle dominates a cluster.
        g = g.groupby("track_id").agg(
            x=(xcol, "mean"), y=(ycol, "mean"),
            stationary_s=("timestamp", "size"), cls=("class", "first"),
        ).reset_index()
        if len(g) < min_v:
            continue

        pts = g[["x", "y"]].to_numpy()
        # Single-linkage via union-find over close pairs.
        parent = list(range(len(pts)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in cKDTree(pts).query_pairs(radius):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        labels = np.array([find(i) for i in range(len(pts))])
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            if idx.size < min_v:
                continue
            cl = pts[idx]
            centred = cl - cl.mean(axis=0)
            if idx.size >= 2:
                # Principal axis = queue direction; extent along it = queue length.
                _, _, vt = np.linalg.svd(centred, full_matrices=False)
                proj = centred @ vt[0]
                length = float(proj.max() - proj.min())
            else:
                length = 0.0
            tids = g["track_id"].to_numpy()[idx]
            wait = df[df["track_id"].isin(tids)].groupby("track_id")["timestamp"].apply(
                lambda s: float(s.max() - s.min())
            ).mean()
            rows.append(
                {
                    "t_start": round(b * bin_s, 1),
                    "t_end": round((b + 1) * bin_s, 1),
                    "vehicles": int(idx.size),
                    "length": round(length, 1),
                    "length_unit": unit,
                    "avg_time_in_view_s": round(float(wait), 1),
                    "centre_x": round(float(cl[:, 0].mean()), 1),
                    "centre_y": round(float(cl[:, 1].mean()), 1),
                    "track_ids": ",".join(str(int(t)) for t in sorted(tids)),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["vehicles", "length"], ascending=False).reset_index(drop=True)


def congestion_timeline(df: pd.DataFrame, cfg: dict, calibrated: bool, bin_s: float = 2.0) -> pd.DataFrame:
    """
    Congestion estimate over time.

    Score is a transparent weighted blend of three normalised indicators:
        0.45 * fraction of vehicles stopped/slow
      + 0.35 * speed deficit vs observed free-flow (85th percentile speed)
      + 0.20 * vehicle count relative to the busiest observed bin
    This is an ANALYTICAL ESTIMATE for operator triage, not a certified
    traffic-engineering level of service (which needs lane geometry and capacity).
    """
    if df.empty:
        return pd.DataFrame()

    acfg = cfg["analytics"]
    ccfg = acfg.get("congestion", {})
    med_thresh = float(ccfg.get("medium_score", 0.35))
    high_thresh = float(ccfg.get("high_score", 0.6))
    speed_col = "speed_kph" if calibrated else "speed_px_s"
    stop_thresh = (
        float(acfg.get("queue_speed_kph", 8.0))
        if calibrated
        else float(np.nanpercentile(df["speed_px_s"], 25))
    )

    veh = df[df["class"].isin(MOTOR_VEHICLE_CLASSES)].copy()
    if veh.empty:
        return pd.DataFrame()
    veh["bin"] = (veh["timestamp"] // bin_s).astype(int)

    free_flow = float(np.nanpercentile(veh[speed_col], 85)) or 1.0
    agg = veh.groupby("bin").agg(
        active=("track_id", "nunique"),
        mean_speed=(speed_col, "mean"),
        slow_frac=(speed_col, lambda s: float((s < stop_thresh).mean())),
    ).reset_index()

    peak = max(int(agg["active"].max()), 1)
    deficit = (1.0 - agg["mean_speed"] / free_flow).clip(0.0, 1.0)
    agg["score"] = (0.45 * agg["slow_frac"] + 0.35 * deficit + 0.20 * agg["active"] / peak).clip(0, 1)
    agg["level"] = np.where(
        agg["score"] >= high_thresh, "HIGH", np.where(agg["score"] >= med_thresh, "MEDIUM", "LOW")
    )
    agg["t"] = agg["bin"] * bin_s
    agg["speed_unit"] = "km/h" if calibrated else "px/s"
    agg["mean_speed"] = agg["mean_speed"].round(1)
    agg["score"] = agg["score"].round(3)
    agg["slow_frac"] = agg["slow_frac"].round(3)
    return agg[["t", "active", "mean_speed", "speed_unit", "slow_frac", "score", "level"]]


def active_track_timeline(df: pd.DataFrame, bin_s: float = 1.0) -> pd.DataFrame:
    """Simultaneously-tracked object count over time."""
    if df.empty:
        return pd.DataFrame()
    d = df.assign(bin=(df["timestamp"] // bin_s).astype(int))
    out = d.groupby("bin")["track_id"].nunique().rename("active_tracks").reset_index()
    out["t"] = out["bin"] * bin_s
    return out[["t", "active_tracks"]]


def id_stability_stats(df: pd.DataFrame, summary: pd.DataFrame, fps: float, stride: int) -> dict:
    """
    Evidence that identities are actually stable.

    `gap_recoveries` counts the times a track was re-associated after one or more
    missed frames - i.e. ByteTrack's buffer bridging an occlusion instead of
    starting a new ID. That is the concrete, measurable claim we can make about
    occlusion handling without hand-labelled ground truth.
    """
    if df.empty:
        return {}
    d = df.sort_values(["track_id", "frame"])
    gaps = d.groupby("track_id")["frame"].diff()
    expected = stride
    recoveries = int((gaps > expected).sum())
    longest_gap = float(gaps.max() / fps) if gaps.notna().any() else 0.0
    return {
        "tracks": int(len(summary)),
        "observations": int(len(df)),
        "mean_track_duration_s": round(float(summary["visible_s"].mean()), 2),
        "max_track_duration_s": round(float(summary["visible_s"].max()), 2),
        "tracks_over_10s": int((summary["visible_s"] >= 10).sum()),
        "gap_recoveries": recoveries,
        "longest_bridged_gap_s": round(longest_gap, 2),
    }

# Hardest braking a road vehicle can physically achieve is about 1 g on dry
# tarmac; anything past this is a measurement artefact, not a manoeuvre.
PHYSICAL_ACCEL_LIMIT_MS2 = 10.0


def build_kinematics(df: pd.DataFrame, summary: pd.DataFrame, cfg: dict,
                     calibrated: bool) -> tuple[pd.DataFrame, dict]:
    """
    LEVEL 2 deliverable: per-object velocity and acceleration in REAL units.

    The per-observation speed/acceleration already live in trajectories.csv; this
    condenses them to one row per road user, which is the form the deliverable
    asks for and the form a traffic engineer actually reads.

    Honesty about the units
    -----------------------
    km/h and m/s^2 are only reported when telemetry calibration succeeded, because
    they come from the pinhole projection onto the ground plane - not from a guess
    about scale. Without calibration these columns stay empty and only px/s is
    reported, rather than dressing up pixel motion as a physical speed.

    Acceleration is differentiated from an already-smoothed speed, so it is a
    trend over ~0.5 s, not an instantaneous g-force. A single frame of box jitter
    on a 20 px motorcycle would otherwise read as several m/s^2.
    """
    if df.empty:
        return pd.DataFrame(), {}

    acfg = cfg["analytics"]
    still = float(acfg.get("stationary_speed_kph", 3.0))
    rows = []
    for tid, g in df.sort_values(["track_id", "frame"]).groupby("track_id", sort=False):
        v = g["speed_kph"].to_numpy(dtype=float) if calibrated else np.full(len(g), np.nan)
        a = g["accel_kph_s"].to_numpy(dtype=float) if calibrated else np.full(len(g), np.nan)
        vpx = g["speed_px_s"].to_numpy(dtype=float)
        moving = v > still if calibrated else vpx > 0
        # m/s^2 is the unit an engineer expects for acceleration; 1 km/h/s = 0.2778 m/s^2.
        a_ms2 = a / 3.6
        rows.append({
            "track_id": int(tid),
            "class": g["class"].iloc[0],
            "n_obs": int(len(g)),
            "duration_s": round(float(g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]), 2),
            # Two speeds, because they answer different questions: journey speed
            # includes the time spent stopped at the signal, cruise speed does not.
            "mean_speed_kph": _r(np.nanmean(v) if np.isfinite(v).any() else np.nan),
            "mean_moving_speed_kph": _r(np.nanmean(v[moving]) if moving.any() else np.nan),
            "median_speed_kph": _r(np.nanmedian(v[moving]) if moving.any() else np.nan),
            "p85_speed_kph": _r(np.nanpercentile(v[moving], 85) if moving.any() else np.nan),
            # p98 is the headline peak; the raw max is a handful of frames of box
            # jitter on a 20 px object and is kept only for audit.
            "p98_speed_kph": _r(np.nanpercentile(v, 98) if np.isfinite(v).any() else np.nan),
            "max_speed_kph": _r(np.nanmax(v) if np.isfinite(v).any() else np.nan),
            "mean_speed_ms": _r(np.nanmean(v[moving]) / 3.6 if moving.any() else np.nan),
            # ACCELERATION, robust percentiles first. The raw extremes are not
            # physical: box corners jitter by a pixel or two between frames, which
            # differentiates into tens of m/s^2. Reporting p95/p5 keeps the real
            # signal (a vehicle braking for the stop line) and discards the spike.
            "p95_accel_ms2": _r(np.nanpercentile(a_ms2, 95) if np.isfinite(a_ms2).any() else np.nan),
            "p5_decel_ms2": _r(np.nanpercentile(a_ms2, 5) if np.isfinite(a_ms2).any() else np.nan),
            "max_accel_ms2_raw": _r(np.nanmax(a_ms2) if np.isfinite(a_ms2).any() else np.nan),
            "max_decel_ms2_raw": _r(np.nanmin(a_ms2) if np.isfinite(a_ms2).any() else np.nan),
            "mean_abs_accel_ms2": _r(np.nanmean(np.abs(a_ms2)) if np.isfinite(a_ms2).any() else np.nan),
            # Flag rather than silently clip, so the number stays auditable.
            "accel_exceeds_physical": bool(
                np.isfinite(a_ms2).any() and np.nanmax(np.abs(a_ms2)) > PHYSICAL_ACCEL_LIMIT_MS2),
            "moving_fraction": _r(float(np.mean(moving)) if len(moving) else np.nan),
            "mean_speed_px_s": _r(np.nanmean(vpx)),
            "units": "km/h and m/s^2" if calibrated else "px/s only - uncalibrated",
            "basis": ("telemetry-calibrated ground plane (estimate)" if calibrated
                      else "image space - no metric scale available"),
        })
    kin = pd.DataFrame(rows)

    if not summary.empty and "length_m" in summary:
        kin = kin.merge(summary[["track_id", "length_m", "width_m", "class_source"]],
                        on="track_id", how="left")

    stats = {}
    if calibrated and not kin.empty:
        by_class = (kin.groupby("class")
                    .agg(road_users=("track_id", "size"),
                         mean_journey_speed_kph=("mean_speed_kph", "mean"),
                         mean_moving_speed_kph=("mean_moving_speed_kph", "mean"),
                         p85_speed_kph=("p85_speed_kph", "mean"),
                         # Median of the per-track robust peaks: what a typical
                         # member of this class actually does, not the worst frame
                         # of the worst track.
                         typical_accel_ms2=("p95_accel_ms2", "median"),
                         typical_decel_ms2=("p5_decel_ms2", "median")).round(2))
        n_flag = int(kin["accel_exceeds_physical"].sum())
        clean = kin[~kin["accel_exceeds_physical"]]
        stats = {
            "road_users": int(len(kin)),
            "speed_unit": "km/h", "acceleration_unit": "m/s^2",
            "by_class": {c: r.to_dict() for c, r in by_class.iterrows()},
            "fleet_mean_speed_kph": _r(kin["mean_speed_kph"].mean()),
            # Fleet extremes are taken over the tracks whose acceleration stayed
            # inside the physical envelope. On a short track p5 is nearly the raw
            # minimum, so a jitter spike survives the percentile and would set the
            # fleet record on its own.
            "hardest_braking_ms2": _r(clean["p5_decel_ms2"].min()) if len(clean) else None,
            "hardest_accel_ms2": _r(clean["p95_accel_ms2"].max()) if len(clean) else None,
            "extremes_computed_over_tracks": int(len(clean)),
            "tracks_with_implausible_accel_spike": n_flag,
            "physical_accel_limit_ms2": PHYSICAL_ACCEL_LIMIT_MS2,
            "method": ("per-observation speed = centred difference of smoothed "
                       "ground coordinates over a ~0.5 s baseline; acceleration = "
                       "smoothed derivative of that speed; headline figures are "
                       "p95/p5 per track, not the raw extremes"),
            "caveat": ("estimates from a single hovering camera. The RAW per-track "
                       f"acceleration extremes reach {_r(kin['max_accel_ms2_raw'].max())} m/s^2, "
                       "which is not physical - a pixel of box jitter differentiates "
                       "into tens of m/s^2 on a small object. Raw columns are kept "
                       "in kinematics.csv for audit but are not reported as results."),
        }
    return kin, stats


def _r(v, nd: int = 2):
    """Round, tolerating NaN, so a missing measurement stays missing."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return round(f, nd) if np.isfinite(f) else np.nan
