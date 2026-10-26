"""
Optional size refinement of the aerial detector's per-track classes.

Why this module exists
----------------------
The original COCO class head did not work reliably from 70 m looking down. In an
archived Level 1 run, across 69,505 detections its labels were:

    car 63,931 | truck 2,565 | pedestrian 1,625 | motorcycle 817 | bus 567

92% of every detection came back "car". Per-track majority voting was NOT the
problem - median vote purity was 1.00, i.e. the model was confidently and
consistently wrong. YOLO was trained on ground-level photographs where a bus is a
tall slab of windows; from directly above it is a long rectangle, which is out of
distribution, and the class head collapses onto its dominant prior.

That finding motivated a measured-size fallback/refinement. The active upload
model is VisDrone-trained and contributes aerial-specific class labels; calibrated
size is used only as a second signal, and only when enough valid geometry exists.
The original thresholds remain footage-calibrated heuristics, not independently
validated vehicle dimensions or a replacement for a labeled classifier.

How the dimensions are recovered
--------------------------------
YOLO gives an AXIS-ALIGNED box, so its width is not the vehicle's width - it is a
mixture of length and width that depends on which way the vehicle is pointing. For
a vehicle of ground length L and width W at image heading phi:

    footprint_m = L*|cos phi| + W*|sin phi|      (ground-projected bottom edge)
    depth_m     = L*|sin phi| + W*|cos phi|      (ground-projected vertical extent)

Two equations, two unknowns, solved per observation:

    det = cos^2(phi) - sin^2(phi) = cos(2*phi)

The solve degenerates at phi = 45 and 135 degrees, where the box is square and
carries no orientation information, so those observations are discarded
(|cos 2phi| < 0.35) rather than fitted. Each track then takes the MEDIAN over its
surviving observations, which is robust to per-frame box jitter.

What the numbers do and do not mean
-----------------------------------
`depth_m` is the box's far edge projected onto the road plane. For a tall vehicle
that lands beyond the real bodywork, because the roof is not on the ground - so
measured length is biased UPWARDS with vehicle height, by roughly
height * tan(27 deg) at this gimbal angle. We do not correct for it (we cannot
measure height from one view) and we do not hide it: the thresholds below were
calibrated against the measured distribution on this footage, not copied from a
vehicle catalogue, and every row carries the measurement that produced its label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Length thresholds in metres on the MEASURED length, calibrated against the
# distribution this footage actually produces (per-track medians):
#
#     what COCO called it | measured length | measured width | tracks
#     pedestrian          |      1.55       |     0.52       |   20
#     motorcycle          |      1.92       |     0.73       |    8
#     car                 |      2.19       |     1.20       |  193
#     truck               |      4.09       |     1.92       |   11
#     bus                 |      4.76       |     2.94       |    2
#
# These are NOT catalogue dimensions - de-rotating an axis-aligned box compresses
# the scale, so a 4.4 m car measures ~2.2 m. What matters for classification is
# that the ordering is monotonic and the bands are drawn where this footage puts
# them, which is why the thresholds are calibrated rather than copied.
SIZE_BANDS = [
    (3.0, "car"),
    (4.5, "lgv"),            # van / light goods
    (float("inf"), "large"),  # bus vs HGV decided by the model, see below
]

# Classes the model is left in charge of. COCO is genuinely reliable on people and
# two-wheelers from above - a rider is a distinctive shape, and the size feature is
# weakest exactly here (motorcycle 1.92 m vs car 2.19 m is only a 1.14x gap, while
# car vs truck is 1.87x). Use each signal where it is strong.
MODEL_OWNED = ("pedestrian", "bicycle", "motorcycle")

# Within the large band, size cannot separate a 12 m bus from a 12 m truck - they
# are the same object to a tape measure. This is the one place the model's opinion
# is genuinely informative, so it is used as a tie-break and labelled as such.
LARGE_DEFAULT = "hgv"


def measure_ground_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-observation `length_m` / `width_m` by de-rotating the axis-aligned box.

    Returns the frame with two new columns; both are NaN where the geometry is
    uninformative (square box, missing projection, no heading yet).
    """
    if df.empty or "depth_m" not in df or "heading_px_deg" not in df:
        df["length_m"] = np.nan
        df["width_m"] = np.nan
        return df

    phi = np.radians(df["heading_px_deg"].to_numpy(dtype=float))
    c, s = np.abs(np.cos(phi)), np.abs(np.sin(phi))
    det = c * c - s * s

    fp = df["footprint_m"].to_numpy(dtype=float)
    dp = df["depth_m"].to_numpy(dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        a = (fp * c - dp * s) / det
        b = (dp * c - fp * s) / det

    # Reject the degenerate band around 45 deg, and any non-physical solve.
    bad = (np.abs(det) < 0.35) | ~np.isfinite(a) | ~np.isfinite(b) | (a <= 0) | (b <= 0)
    a, b = np.where(bad, np.nan, a), np.where(bad, np.nan, b)

    # A vehicle is longer than it is wide; ordering the pair removes the ambiguity
    # about which axis the solve assigned to which dimension.
    df["length_m"] = np.round(np.fmax(a, b), 2)
    df["width_m"] = np.round(np.fmin(a, b), 2)
    return df


def _model_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Confidence-weighted per-track class vote, plus the winner's vote share."""
    w = df.assign(_w=df["confidence"].fillna(0.0).clip(lower=0.01))
    tot = w.groupby("track_id")["_w"].sum().rename("_tot")
    per = w.groupby(["track_id", "class"])["_w"].sum().rename("_s").reset_index()
    per = per.merge(tot, on="track_id")
    per["share"] = per["_s"] / per["_tot"]
    win = per.sort_values(["track_id", "_s"], ascending=[True, False]).drop_duplicates("track_id")
    return win.rename(columns={"class": "model_class", "share": "model_vote_share"})[
        ["track_id", "model_class", "model_vote_share"]
    ]


def classify_tracks(df: pd.DataFrame, cfg: dict, calibrated: bool) -> pd.DataFrame:
    """
    One resolved class per track, from measured size where possible.

    Columns returned:
        track_id, class, class_source, model_class, model_vote_share,
        length_m, width_m, n_size_obs
    """
    if df.empty:
        return pd.DataFrame(columns=["track_id", "class", "class_source", "model_class",
                                     "model_vote_share", "length_m", "width_m", "n_size_obs"])

    out = _model_votes(df)
    out["class"] = out["model_class"]
    out["class_source"] = "model"
    out["length_m"] = np.nan
    out["width_m"] = np.nan
    out["n_size_obs"] = 0

    if not calibrated or "length_m" not in df:
        return out

    ccfg = cfg.get("classification", {}) or {}
    min_obs = int(ccfg.get("min_size_observations", 8))
    bands = [(float(v), k) for v, k in
             (ccfg.get("size_bands") or [[u, n] for u, n in SIZE_BANDS[:-1]])]
    bands.append((float("inf"), "large"))
    model_owned = tuple(ccfg.get("model_owned_classes") or MODEL_OWNED)

    dims = (
        df.groupby("track_id")
        .agg(length_m=("length_m", "median"),
             width_m=("width_m", "median"),
             n_size_obs=("length_m", "count"))
        .reset_index()
    )
    out = out.drop(columns=["length_m", "width_m", "n_size_obs"]).merge(dims, on="track_id", how="left")

    # Size arbitrates only where it is the stronger signal. Pedestrians and
    # two-wheelers stay with the model (see MODEL_OWNED).
    sized = (out["n_size_obs"].fillna(0) >= min_obs) & out["length_m"].notna() \
        & ~out["model_class"].isin(model_owned)

    def band(length: float) -> str:
        for upper, name in bands:
            if length < upper:
                return name
        return "large"

    resolved, source = [], []
    fp_split = float((cfg.get("analytics", {}) or {}).get("lgv_hgv_split_m", 7.0))
    fp90 = (df.groupby("track_id")["footprint_m"].quantile(0.9)
            if "footprint_m" in df else pd.Series(dtype=float))
    for _, r in out.iterrows():
        if r["model_class"] in model_owned:
            resolved.append(r["model_class"])
            source.append("model_reliable_for_this_class")
            continue
        if not sized.loc[r.name]:
            # Too few usable solves (a box that stayed near 45 deg the whole time).
            # `truck` must still not escape: it is not in the challenge vocabulary,
            # so fall back to the Level 1 footprint heuristic rather than emitting
            # a label we cannot defend.
            if r["model_class"] == "truck":
                f = float(fp90.get(r["track_id"], np.nan))
                resolved.append("hgv" if np.isfinite(f) and f >= fp_split else "lgv")
                source.append("footprint_fallback_insufficient_geometry")
            else:
                resolved.append(r["model_class"])
                source.append("model_only_insufficient_geometry")
            continue
        b = band(float(r["length_m"]))
        if b == "large":
            # Size says "large vehicle"; only the model can say bus or truck.
            if r["model_class"] == "bus":
                resolved.append("bus")
                source.append("measured_size_large_model_says_bus")
            else:
                resolved.append(LARGE_DEFAULT)
                source.append("measured_size_large")
        else:
            resolved.append(b)
            source.append("measured_size"
                          if b == r["model_class"] else "measured_size_overrides_model")
    out["class"] = resolved
    out["class_source"] = source
    return out


def classification_report(classes: pd.DataFrame) -> dict:
    """
    Evidence about the classifier itself, for the dashboard and the write-up.

    The interesting number is how often measured size DISAGREES with the model.
    A high override rate is the finding, not a bug: it is the size of the error
    the COCO class head was making from this viewpoint.
    """
    if classes.empty:
        return {}
    n = len(classes)
    overrides = classes[classes["class"] != classes["model_class"]]
    by_class = classes["class"].value_counts().to_dict()
    dims = (
        classes.dropna(subset=["length_m"])
        .groupby("class")[["length_m", "width_m"]]
        .median().round(2)
    )
    return {
        "tracks": int(n),
        "resolved_by_class": by_class,
        "model_would_have_said": classes["model_class"].value_counts().to_dict(),
        "size_overrode_model": int(len(overrides)),
        "size_overrode_model_pct": round(100.0 * len(overrides) / n, 1),
        "override_flow": (
            overrides.groupby(["model_class", "class"]).size()
            .sort_values(ascending=False).head(8)
            .rename_axis(["from", "to"]).reset_index(name="n")
            .apply(lambda r: f"{r['from']} -> {r['to']}: {r['n']}", axis=1).tolist()
            if len(overrides) else []
        ),
        "class_source_counts": classes["class_source"].value_counts().to_dict(),
        "median_measured_dimensions_m": {
            c: {"length": float(r["length_m"]), "width": float(r["width_m"])}
            for c, r in dims.iterrows()
        },
        "method": ("length/width solved from the axis-aligned box and the image heading, "
                   "then median per track; thresholds calibrated on the measured "
                   "distribution for this footage"),
        "caveat": ("measured length is biased upwards by vehicle height (the box's far "
                   "edge is projected onto the road plane), so these are consistent "
                   "relative sizes rather than catalogue dimensions"),
    }
