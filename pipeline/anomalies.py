"""
Event / anomaly engine.

Detectors read the trajectory table and emit rows into a single flat event
schema. Adding a new detector for a later level means writing one function and
appending it to DETECTORS - nothing upstream changes.

Event schema (outputs/events.csv):
    event_type, track_id, secondary_track_id, timestamp, duration,
    x, y, severity, value, unit, description

The detectors here are the ones that can be computed RELIABLY from a single
aerial clip. Wrong-way detection is included but is explicitly relative to the
observed dominant flow of the corridor, not to a legal carriageway direction -
we have no HD map, so we do not claim a traffic violation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.detector import MOTOR_VEHICLE_CLASSES

EVENT_COLUMNS = [
    "event_type",
    "track_id",
    "secondary_track_id",
    "timestamp",
    "duration",
    "x",
    "y",
    "severity",
    "value",
    "unit",
    "description",
]


def _event(**kw) -> dict:
    row = {c: kw.get(c, np.nan) for c in EVENT_COLUMNS}
    return row


def stationary_events(df, summary, queues, cfg, calibrated) -> list[dict]:
    """
    Sustained stationary road users.

    Cross-referenced against detected queue clusters: a vehicle stopped inside a
    queue is almost certainly waiting at a signal, so it is reported at low
    severity and described as signal-related rather than as an incident.
    """
    from pipeline.analytics import stationary_candidates

    cands = stationary_candidates(summary, cfg)
    if cands.empty:
        return []

    queued_ids: set[int] = set()
    if queues is not None and not queues.empty:
        for s in queues["track_ids"]:
            queued_ids.update(int(t) for t in str(s).split(",") if t)

    out = []
    for _, r in cands.iterrows():
        in_queue = int(r["track_id"]) in queued_ids
        dur = float(r["longest_stop_s"])
        if in_queue:
            sev, desc = "low", "Stationary within a detected queue - likely signal/queue related, not an incident"
        elif dur >= 3 * float(cfg["analytics"].get("stationary_min_s", 5.0)):
            sev, desc = "high", "Stationary vehicle candidate - long stop, outside any detected queue"
        else:
            sev, desc = "medium", "Stationary vehicle candidate - outside any detected queue"
        out.append(
            _event(
                event_type="stationary_vehicle_candidate",
                track_id=int(r["track_id"]),
                timestamp=r["stop_start_t"],
                duration=round(dur, 2),
                x=r["last_x"],
                y=r["last_y"],
                severity=sev,
                value=round(dur, 2),
                unit="s",
                description=f"{r['class']} #{int(r['track_id'])} stationary {dur:.1f}s. {desc}",
            )
        )
    return out


def unusual_dwell_events(df, summary, queues, cfg, calibrated) -> list[dict]:
    """
    Road users present far longer than typical for the scene.

    A bare "visible for > 25 s" test is close to useless on this footage: it fired on
    27 tracks, every one of them a car that was stationary for 100% of its life at the
    kerbside. Those are PARKED VEHICLES, not anomalies - and because the event
    duration equalled the whole clip, the annotated video kept all 27 highlighted for
    all 1798 frames, which made the overlay read as though the entire scene was
    anomalous.

    So the two cases are separated on the fraction of life spent stationary:

      * never meaningfully moved  -> `parked_vehicle_candidate`, informational.
        Honest label: from one clip we cannot distinguish legal parking from a
        vehicle abandoned in a live lane, so it stays a "candidate".
      * moved, then dwelled long  -> `unusual_dwell`, the case actually worth a
        look, because something interrupted a journey in progress.
    """
    if summary.empty:
        return []
    limit = float(cfg["anomalies"].get("unusual_dwell_s", 25.0))
    parked_frac = float(cfg["anomalies"].get("parked_stationary_fraction", 0.9))
    d = summary[summary["visible_s"] >= limit]
    out = []
    for _, r in d.iterrows():
        vis = float(r["visible_s"])
        stat = float(r.get("stationary_s", 0.0))
        frac = stat / vis if vis > 0 else 0.0
        parked = frac >= parked_frac
        moving_s = max(vis - stat, 0.0)
        out.append(_event(
            event_type="parked_vehicle_candidate" if parked else "unusual_dwell",
            track_id=int(r["track_id"]),
            timestamp=r["first_t"],
            # Parked vehicles get a short marker window instead of the whole clip:
            # a 60 s event window would light the object up for the entire video.
            duration=3.0 if parked else vis,
            x=r["last_x"],
            y=r["last_y"],
            severity="low",
            value=round(vis, 2),
            unit="s",
            description=(
                f"{r['class']} #{int(r['track_id'])} stationary {stat:.1f}s of "
                f"{vis:.1f}s in view ({frac:.0%}) - parked, or stopped for the whole "
                "observation. Not distinguishable from an obstruction in one clip."
                if parked else
                f"{r['class']} #{int(r['track_id'])} in view {vis:.1f}s, moving for "
                f"{moving_s:.1f}s then dwelling {stat:.1f}s - a journey interrupted"
            ),
        ))
    return out


def sudden_stop_events(df, summary, queues, cfg, calibrated) -> list[dict]:
    """Hard deceleration spikes - a braking event, often the tail of a conflict."""
    if not calibrated or df.empty or "accel_kph_s" not in df:
        return []
    limit = -abs(float(cfg["anomalies"].get("sudden_stop_decel_kph_s", 18.0)))
    hard = df[df["accel_kph_s"].le(limit) & df["class"].isin(MOTOR_VEHICLE_CLASSES)]
    if hard.empty:
        return []
    # One event per track: its single hardest braking moment.
    idx = hard.groupby("track_id")["accel_kph_s"].idxmin()
    out = []
    for _, r in hard.loc[idx].iterrows():
        out.append(
            _event(
                event_type="sudden_stop",
                track_id=int(r["track_id"]),
                timestamp=round(float(r["timestamp"]), 2),
                duration=np.nan,
                x=r["x"],
                y=r["y"],
                severity="medium" if r["accel_kph_s"] > 2 * limit else "high",
                value=round(float(r["accel_kph_s"]), 1),
                unit="km/h/s",
                description=(
                    f"{r['class']} #{int(r['track_id'])} decelerated "
                    f"{abs(float(r['accel_kph_s'])):.0f} km/h/s at t={float(r['timestamp']):.1f}s"
                ),
            )
        )
    return out


def wrong_way_events(df, summary, queues, cfg, calibrated) -> list[dict]:
    """
    Movement against the dominant flow of the same part of the road.

    The reference direction is learned from the data: the scene is divided into a
    coarse spatial grid and each cell's modal travel bearing becomes the local
    expected flow. A road user moving strongly against its own cell's flow is
    flagged. Without an HD map this is a FLOW ANOMALY, not a proven violation.
    """
    if df.empty:
        return []
    acfg = cfg["anomalies"]
    min_speed = float(acfg.get("wrong_way_min_speed_kph", 8.0))
    dev_limit = float(acfg.get("wrong_way_deviation_deg", 130.0))

    if calibrated:
        xcol, ycol, scol, bcol, cell = "sx", "sy", "speed_kph", "bearing_deg", 15.0
    else:
        xcol, ycol, scol, bcol, cell = "x", "y", "speed_px_s", "heading_px_deg", 160.0
        min_speed = float(np.nanpercentile(df["speed_px_s"], 60))

    d = df.dropna(subset=[xcol, ycol, scol, bcol])
    d = d[d[scol] >= min_speed & d["class"].isin(MOTOR_VEHICLE_CLASSES)] if False else d[
        (d[scol] >= min_speed) & (d["class"].isin(MOTOR_VEHICLE_CLASSES))
    ]
    if len(d) < 50:
        return []

    d = d.assign(
        gx=np.floor(d[xcol] / cell).astype(int),
        gy=np.floor(d[ycol] / cell).astype(int),
        _s=np.sin(np.radians(d[bcol])),
        _c=np.cos(np.radians(d[bcol])),
    )
    # Circular mean bearing per cell = local dominant flow.
    flow = d.groupby(["gx", "gy"]).agg(s=("_s", "mean"), c=("_c", "mean"), n=("_s", "size")).reset_index()
    flow = flow[flow["n"] >= 20]  # ignore cells with too little evidence
    if flow.empty:
        return []
    flow["flow_deg"] = (np.degrees(np.arctan2(flow["s"], flow["c"])) + 360.0) % 360.0

    d = d.merge(flow[["gx", "gy", "flow_deg"]], on=["gx", "gy"], how="inner")
    if d.empty:
        return []
    d["dev"] = np.abs((d[bcol] - d["flow_deg"] + 180.0) % 360.0 - 180.0)

    # Require a sustained majority of the track to be against the flow, so a
    # single noisy heading estimate cannot trigger it.
    frac = d.assign(bad=d["dev"] >= dev_limit).groupby("track_id")["bad"].agg(["mean", "size"])
    suspect = frac[(frac["mean"] >= 0.6) & (frac["size"] >= 15)]
    out = []
    for tid, r in suspect.iterrows():
        g = d[d["track_id"] == tid]
        out.append(
            _event(
                event_type="against_dominant_flow",
                track_id=int(tid),
                timestamp=round(float(g["timestamp"].min()), 2),
                duration=round(float(g["timestamp"].max() - g["timestamp"].min()), 2),
                x=round(float(g["x"].median()), 1),
                y=round(float(g["y"].median()), 1),
                severity="high",
                value=round(float(g["dev"].median()), 1),
                unit="deg from local flow",
                description=(
                    f"{g['class'].iloc[0]} #{int(tid)} moved {float(g['dev'].median()):.0f} deg against the "
                    f"dominant local flow for {r['size']} observations (flow anomaly, not a confirmed violation)"
                ),
            )
        )
    return out


def queue_events(df, summary, queues, cfg, calibrated) -> list[dict]:
    """Queue formation, reported once per queue at its largest observed extent."""
    if queues is None or queues.empty:
        return []
    # Collapse to the worst bin per spatial location so one queue = one event.
    q = queues.copy()
    q["loc"] = (q["centre_x"] // 20).astype(int).astype(str) + "_" + (q["centre_y"] // 20).astype(int).astype(str)
    best = q.sort_values("vehicles", ascending=False).drop_duplicates("loc")
    out = []
    for _, r in best.iterrows():
        out.append(
            _event(
                event_type="queue_formation",
                track_id=np.nan,
                timestamp=r["t_start"],
                duration=round(float(r["t_end"] - r["t_start"]), 2),
                x=r["centre_x"],
                y=r["centre_y"],
                severity="high" if r["vehicles"] >= 8 else "medium",
                value=r["length"],
                unit=r["length_unit"],
                description=(
                    f"Queue of {int(r['vehicles'])} vehicles, ~{r['length']:.0f} {r['length_unit']} long "
                    f"at t={r['t_start']:.0f}s"
                ),
            )
        )
    return out


def conflict_events(interactions, cfg) -> list[dict]:
    """Turn interaction rows into events."""
    if interactions is None or interactions.empty:
        return []
    out = []
    for _, r in interactions.iterrows():
        has_ttc = r["ttc_s"] == r["ttc_s"]
        grade = r.get("conflict_grade", "critical" if has_ttc else "proximity")
        # Only graded conflicts are safety events. A "close interaction" cleared
        # every geometric filter but is low-energy, i.e. routine dense traffic -
        # calling it a conflict would inflate the count by ~4x. It is still
        # emitted, at low severity, because it is a useful density signal.
        if grade in ("critical", "serious"):
            etype = "potential_conflict"
            desc = (
                f"Potential conflict ({grade}): {r['class_a']} #{int(r['track_a'])} <-> "
                f"{r['class_b']} #{int(r['track_b'])} - projected TTC {r['ttc_s']:.1f}s, "
                f"closest approach {r['d_cpa']:.1f}{r['unit']}, closing "
                f"{r['closing_speed']:.0f} km/h ({r['interaction_type']})"
            )
        elif grade == "close_interaction":
            etype = "close_interaction"
            desc = (
                f"Close interaction: {r['class_a']} #{int(r['track_a'])} <-> {r['class_b']} "
                f"#{int(r['track_b'])} - TTC {r['ttc_s']:.1f}s but closing only "
                f"{r['closing_speed']:.0f} km/h, so low-energy ({r['interaction_type']}). "
                f"Density signal, not a safety event."
            )
        else:
            etype = "near_miss_proximity"
            desc = (
                f"Sustained close proximity: {r['class_a']} #{int(r['track_a'])} <-> {r['class_b']} "
                f"#{int(r['track_b'])} - min separation {r['min_distance']:.1f}{r['unit']} ({r['interaction_type']})"
            )
        out.append(
            _event(
                event_type=etype,
                track_id=int(r["track_a"]),
                secondary_track_id=int(r["track_b"]),
                timestamp=r["t"],
                duration=np.nan,
                x=r["x"],
                y=r["y"],
                severity=r["severity"],
                value=r["ttc_s"] if has_ttc else r["min_distance"],
                unit="s (TTC)" if has_ttc else r["unit"],
                description=desc,
            )
        )
    return out


# Trajectory-only detectors. Add new ones here as levels unlock.
DETECTORS = [
    stationary_events,
    queue_events,
    sudden_stop_events,
    wrong_way_events,
    unusual_dwell_events,
]


def build_events(df, summary, queues, interactions, cfg, calibrated) -> pd.DataFrame:
    """Run every detector and return one sorted event table."""
    rows: list[dict] = []
    for fn in DETECTORS:
        try:
            rows.extend(fn(df, summary, queues, cfg, calibrated))
        except Exception as exc:  # one broken detector must not kill the run
            print(f"[anomalies] detector {fn.__name__} failed: {exc}")
    try:
        rows.extend(conflict_events(interactions, cfg))
    except Exception as exc:
        print(f"[anomalies] conflict_events failed: {exc}")

    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    order = {"high": 0, "medium": 1, "low": 2}
    return (
        ev.assign(_o=ev["severity"].map(order).fillna(3))
        .sort_values(["_o", "timestamp"])
        .drop(columns=["_o"])
        .reset_index(drop=True)
    )
