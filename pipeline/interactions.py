"""
Interaction analytics - the part the problem statement says existing systems do
worst.

Fixed cameras measure objects; a drone in a single common coordinate frame lets
us measure the RELATIONSHIP between objects. With metric calibration we can
compute genuine surrogate safety measures rather than pixel-overlap guesses.

Method: for every frame, for every pair of road users within a neighbourhood
radius, treat both as points moving at constant velocity and solve for the
closest point of approach (CPA):

    dp = p2 - p1,  dv = v2 - v1
    t_cpa = -(dp . dv) / |dv|^2
    d_cpa = |dp + t_cpa * dv|

If the pair is closing (t_cpa > 0), the CPA happens soon, and the projected
separation falls inside the combined physical envelope of the two road users,
that is a POTENTIAL CONFLICT. Each pair is reduced to its single worst moment,
so one near miss produces one event, not two hundred.

Language is deliberately hedged: "potential conflict", "near miss (sustained
proximity)". A constant-velocity projection cannot know that a driver was already
braking, so these are conflict *indicators* for human review - not collisions.

GRADING, and why a single TTC threshold is not enough
----------------------------------------------------
The traffic-conflict literature's ~1.5 s TTC threshold comes from lane-disciplined
traffic. Measured on this footage it flags ~90% of all interacting pairs, because
in dense mixed traffic low TTC is *normal operating behaviour*: two vehicles
creeping in a queue, converging at 9 km/h and passing 3 m apart, produce a sub-1 s
TTC while being in no danger whatsoever.

So conflicts are graded jointly on TTC **and closing speed** - the latter is the
proxy for how much kinetic energy the evasive manoeuvre has to absorb:

    critical  : TTC <= critical_ttc_s AND closing >= critical_closing_kph
    serious   : TTC <= critical_ttc_s AND closing >= serious_closing_kph
    close     : passed every geometric filter, but low-energy -> routine density
    proximity : sustained closeness with no projected conflict at all

Only critical + serious are reported as potential conflicts. "Close interaction"
is retained and counted, because in dense traffic it is a real congestion/density
signal - it just is not a safety event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Approximate physical half-width of each road user, in metres. Used to size the
# conflict envelope; a bus needs more clearance than a pedestrian.
CLASS_RADIUS_M = {
    "pedestrian": 0.5,
    "bicycle": 0.8,
    "motorcycle": 1.0,
    "car": 1.9,
    "lgv": 2.2,
    "truck": 2.5,
    "hgv": 2.6,
    "bus": 2.6,
}
DEFAULT_RADIUS_M = 2.0

# LATERAL half-width (real vehicle width / 2). This - not the omnidirectional
# radius above - is what sizes the conflict envelope.
#
# Why the distinction matters, measured on this footage: CLASS_RADIUS_M is really
# a half-*length*, so for two cars it gives a 3.8 m envelope. A traffic lane is
# only ~3.0-3.5 m wide, so two vehicles passing normally in adjacent opposing
# lanes (centre separation ~3.4 m) fell inside it and were reported as head-on
# conflicts. Sizing the envelope on half-width instead means a pair is only
# flagged when it is on a genuine collision course.
CLASS_HALF_WIDTH_M = {
    "pedestrian": 0.25,
    "bicycle": 0.30,
    "motorcycle": 0.40,
    "car": 0.90,
    "lgv": 1.00,
    "truck": 1.25,
    "hgv": 1.30,
    "bus": 1.30,
}
DEFAULT_HALF_WIDTH_M = 0.9


def _velocity_components(speed: np.ndarray, bearing_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Speed + compass bearing -> (east, north) velocity components."""
    rad = np.radians(bearing_deg)
    return speed * np.sin(rad), speed * np.cos(rad)


def find_interactions(df: pd.DataFrame, cfg: dict, calibrated: bool) -> pd.DataFrame:
    """
    Detect pairwise interactions across the whole clip.

    Returns one row per interacting pair (its worst moment), with columns:
        track_a, track_b, class_a, class_b, t, ttc_s, d_cpa, min_distance,
        approach_speed, interaction_type, involves_vulnerable, severity, x, y
    Distances are metres when calibrated, pixels otherwise (see `unit`).
    """
    if df.empty:
        return pd.DataFrame()

    icfg = cfg["interactions"]
    radius = float(icfg.get("neighbour_radius_m", 25.0))
    ttc_max = float(icfg.get("ttc_max_s", 4.0))
    ttc_min = float(icfg.get("ttc_min_s", 0.2))
    vuln_ttc = float(icfg.get("vulnerable_ttc_s", 5.0))
    vuln_classes = set(icfg.get("vulnerable_classes", ["pedestrian", "motorcycle", "bicycle"]))
    near_miss_d = float(icfg.get("near_miss_distance_m", 2.5))
    min_approach = float(icfg.get("min_approach_speed_kph", 4.0))
    min_closing = float(icfg.get("min_closing_speed_kph", 6.0)) / 3.6  # -> m/s
    min_persist = int(icfg.get("min_persistence_frames", 5))
    crit_ttc = float(icfg.get("critical_ttc_s", 1.5))
    crit_closing = float(icfg.get("critical_closing_kph", 20.0))
    ser_closing = float(icfg.get("serious_closing_kph", 12.0))
    vuln_closing = float(icfg.get("vulnerable_closing_kph", 8.0))
    env_margin = float(icfg.get("conflict_envelope_margin", 1.15))

    if calibrated:
        xcol, ycol, scol, bcol, unit = "sx", "sy", "speed_kph", "bearing_deg", "m"
        px_per_m = 1.0
    else:
        # Uncalibrated fallback: work in pixels and scale the thresholds by a
        # nominal pixels-per-metre so the geometry still means something.
        xcol, ycol, scol, bcol, unit = "x", "y", "speed_px_s", "heading_px_deg", "px"
        px_per_m = 16.0
        radius *= px_per_m
        near_miss_d *= px_per_m

    d = df.dropna(subset=[xcol, ycol, scol, bcol])
    if d.empty:
        return pd.DataFrame()

    # Physical envelope per observation.
    #   radius   -> omnidirectional, for the sustained-proximity (near miss) test
    #   halfwidth-> lateral clearance, for the collision-course (conflict) test
    d = d.assign(
        radius=d["class"].map(CLASS_RADIUS_M).fillna(DEFAULT_RADIUS_M) * px_per_m,
        halfwidth=d["class"].map(CLASS_HALF_WIDTH_M).fillna(DEFAULT_HALF_WIDTH_M)
        * env_margin * px_per_m,
        speed_ms=d[scol] / (3.6 if calibrated else 1.0),
    )
    vx, vy = _velocity_components(d["speed_ms"].to_numpy(), d[bcol].to_numpy())
    d = d.assign(vx=vx, vy=vy)

    # Worst moment per pair, keyed by the ordered id pair.
    worst: dict[tuple[int, int], dict] = {}
    # How many frames each pair actually satisfied a trigger. Real conflicts
    # persist for ~1 s; tracker jitter fires for one or two frames and vanishes.
    hits: dict[tuple[int, int], int] = {}

    for _, g in d.groupby("frame", sort=True):
        if len(g) < 2:
            continue
        pts = g[[xcol, ycol]].to_numpy()
        vel = g[["vx", "vy"]].to_numpy()
        ids = g["track_id"].to_numpy()
        clss = g["class"].to_numpy()
        rads = g["radius"].to_numpy()
        halfw = g["halfwidth"].to_numpy()
        spd = g[scol].to_numpy()
        bear = g[bcol].to_numpy()
        t = float(g["timestamp"].iloc[0])

        for i, j in cKDTree(pts).query_pairs(radius):
            dp = pts[j] - pts[i]
            dv = vel[j] - vel[i]
            dist = float(np.hypot(*dp))
            envelope = float(rads[i] + rads[j])          # proximity test
            conflict_env = float(halfw[i] + halfw[j])    # collision-course test
            rel_speed = float(np.hypot(*dv))

            is_vuln = clss[i] in vuln_classes or clss[j] in vuln_classes
            ttc_limit = vuln_ttc if is_vuln else ttc_max

            # Both essentially stopped -> not a conflict, just proximity.
            if max(spd[i], spd[j]) < min_approach:
                continue

            # Closing rate along the line of centres. Two cars travelling in a
            # line at the same speed have a large |dv| from noise but a closing
            # rate near zero, so this is the filter that kills fake conflicts.
            closing = -float(np.dot(dp, dv)) / dist if dist > 1e-6 else 0.0

            ttc = np.nan
            d_cpa = dist
            if rel_speed > 1e-3 and closing >= min_closing:
                t_cpa = -float(np.dot(dp, dv)) / (rel_speed ** 2)
                if t_cpa > 0:
                    d_cpa = float(np.hypot(*(dp + t_cpa * dv)))
                    # Lateral envelope, not the omnidirectional one: otherwise
                    # normal adjacent-lane oncoming traffic reads as head-on.
                    if ttc_min <= t_cpa <= ttc_limit and d_cpa <= conflict_env:
                        ttc = t_cpa

            # A near miss needs both users moving AND genuinely close.
            sustained_near_miss = dist <= max(near_miss_d, envelope) and min(spd[i], spd[j]) >= min_approach
            if not (ttc == ttc) and not sustained_near_miss:
                continue

            key = (int(min(ids[i], ids[j])), int(max(ids[i], ids[j])))
            hits[key] = hits.get(key, 0) + 1

            # Interaction geometry from the relative heading of the two paths.
            hdiff = abs((float(bear[j]) - float(bear[i]) + 180.0) % 360.0 - 180.0)
            if hdiff < 25:
                itype = "following"
            elif hdiff > 150:
                itype = "head-on"
            elif hdiff < 60:
                itype = "merging"
            else:
                itype = "crossing"
            if is_vuln:
                itype = f"{itype}/vulnerable-road-user"

            # Rank by TTC first (a projected conflict beats mere proximity).
            score = (0 if ttc == ttc else 1, ttc if ttc == ttc else dist)
            prev = worst.get(key)
            if prev is None or score < prev["_score"]:
                worst[key] = {
                    "_score": score,
                    "track_a": key[0],
                    "track_b": key[1],
                    "class_a": clss[i] if ids[i] == key[0] else clss[j],
                    "class_b": clss[j] if ids[j] == key[1] else clss[i],
                    "t": round(t, 2),
                    "ttc_s": round(ttc, 2) if ttc == ttc else np.nan,
                    "d_cpa": round(d_cpa, 2),
                    "min_distance": round(dist, 2),
                    "closing_speed": round(closing * (3.6 if calibrated else 1.0), 1),
                    "approach_speed": round(rel_speed * (3.6 if calibrated else 1.0), 1),
                    "interaction_type": itype,
                    "involves_vulnerable": bool(is_vuln),
                    "unit": unit,
                    "x": round(float((pts[i][0] + pts[j][0]) / 2), 1),
                    "y": round(float((pts[i][1] + pts[j][1]) / 2), 1),
                }

    # Persistence filter: drop pairs that never sustained the trigger.
    worst = {k: v for k, v in worst.items() if hits.get(k, 0) >= min_persist}
    for k, v in worst.items():
        v["frames_triggered"] = hits[k]

    if not worst:
        return pd.DataFrame()

    out = pd.DataFrame(list(worst.values())).drop(columns=["_score"])

    # Grade jointly on TTC and closing speed. See the module docstring: TTC alone
    # flags routine dense-traffic proximity, because a queue creeping forward at
    # 9 km/h with a 3 m gap is geometrically a "conflict" and behaviourally normal.
    def grade(r) -> str:
        if r["ttc_s"] != r["ttc_s"]:            # NaN -> no projected conflict
            return "proximity"
        if r["ttc_s"] > crit_ttc:
            return "close_interaction"
        closing = float(r["closing_speed"])
        vuln = bool(r["involves_vulnerable"])
        # A pedestrian or rider struck at low speed is still injured, so the
        # energy bar is lower when a vulnerable road user is involved.
        if closing >= crit_closing or (vuln and closing >= ser_closing):
            return "critical"
        if closing >= ser_closing or (vuln and closing >= vuln_closing):
            return "serious"
        return "close_interaction"

    out["conflict_grade"] = out.apply(grade, axis=1)

    # Severity drives the dashboard and events.csv. Only graded conflicts are
    # "high" - close interactions are density signal, not safety events.
    sev_of = {"critical": "high", "serious": "medium",
              "close_interaction": "low", "proximity": "low"}
    out["severity"] = out["conflict_grade"].map(sev_of)

    order = {"critical": 0, "serious": 1, "close_interaction": 2, "proximity": 3}
    return (
        out.assign(_o=out["conflict_grade"].map(order))
        .sort_values(["_o", "ttc_s", "min_distance"])
        .drop(columns=["_o"])
        .reset_index(drop=True)
    )


def following_headways(df: pd.DataFrame, interactions: pd.DataFrame, calibrated: bool) -> pd.DataFrame:
    """
    Time headway for car-following pairs: gap distance / follower speed.

    Only meaningful with metric calibration, so it returns empty otherwise.
    """
    if not calibrated or interactions.empty or df.empty:
        return pd.DataFrame()
    foll = interactions[interactions["interaction_type"].str.startswith("following")]
    if foll.empty:
        return pd.DataFrame()

    speeds = df.groupby("track_id")["speed_kph"].mean()
    rows = []
    for _, r in foll.iterrows():
        v_kph = float(speeds.get(r["track_a"], np.nan))
        if not (v_kph == v_kph) or v_kph < 3.0:
            continue
        rows.append(
            {
                "leader": int(r["track_b"]),
                "follower": int(r["track_a"]),
                "gap_m": r["min_distance"],
                "follower_speed_kph": round(v_kph, 1),
                "time_headway_s": round(r["min_distance"] / (v_kph / 3.6), 2),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("time_headway_s").reset_index(drop=True)


def interaction_summary(interactions: pd.DataFrame, duration_s: float | None = None,
                        n_road_users: int | None = None) -> dict:
    """
    Headline interaction numbers.

    `potential_conflicts` counts only critical + serious grades. The raw count of
    everything that passed the geometric filters is reported separately as
    `close_interactions`, so the reduction is visible rather than hidden.
    """
    if interactions.empty:
        return {"pairs": 0, "potential_conflicts": 0, "by_type": {}, "by_severity": {}}

    graded = interactions["conflict_grade"] if "conflict_grade" in interactions else None
    if graded is None:                          # older CSV without grading
        conflicts = interactions[interactions["ttc_s"].notna()]
        n_crit = n_ser = None
    else:
        conflicts = interactions[graded.isin(["critical", "serious"])]
        n_crit = int((graded == "critical").sum())
        n_ser = int((graded == "serious").sum())

    out = {
        "pairs": int(len(interactions)),
        "potential_conflicts": int(len(conflicts)),
        "critical": n_crit,
        "serious": n_ser,
        "close_interactions": int((graded == "close_interaction").sum()) if graded is not None else None,
        "geometric_candidates_before_grading": int(interactions["ttc_s"].notna().sum()),
        "vulnerable_involved": int(conflicts["involves_vulnerable"].sum()) if not conflicts.empty else 0,
        "min_ttc_s": round(float(conflicts["ttc_s"].min()), 2) if not conflicts.empty else None,
        "by_type": conflicts["interaction_type"].value_counts().to_dict() if not conflicts.empty else {},
        "by_severity": interactions["severity"].value_counts().to_dict(),
        "grading": (
            "graded jointly on TTC and closing speed; a single TTC threshold flags "
            "routine dense-traffic proximity as a conflict"
        ),
    }
    # Rates make the count comparable across clips of different length/demand.
    if duration_s and duration_s > 0:
        out["conflicts_per_minute"] = round(len(conflicts) * 60.0 / duration_s, 1)
    if n_road_users:
        out["conflicts_per_100_road_users"] = round(len(conflicts) * 100.0 / n_road_users, 1)
    return out
