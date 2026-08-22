"""
Tests for the MobileSAM contact-point refinement.

Deliberately runs in a BARE environment: no ultralytics, no torch, no cv2, no
video. `contact_points` imports cv2 defensively and the estimators below are pure
numpy, so the part of this feature that decides where a vehicle touches the road
is testable without a 40 MB download or a GPU.

Run either way:
    python -m pytest tests/test_contact_points.py -q
    python tests/test_contact_points.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.contact_points import (  # noqa: E402
    _contact_from_mask,
    _pair_masks_to_boxes,
    _vet_mask,
    contact_metrics,
)
from pipeline.trajectory import apply_contact_refinement  # noqa: E402

H, W = 200, 200

# One vehicle: rows 40..80, cols 30..70. Its tyre line is row 80, its true
# contact-line centre is col ~49.5, its contact width ~40 px.
BODY = (40, 81, 30, 70)
BOX = (28.0, 38.0, 72.0, 82.0)


def _rect(*, r0, r1, c0, c1, base=None) -> np.ndarray:
    m = np.zeros((H, W), bool) if base is None else base.copy()
    m[r0:r1, c0:c1] = True
    return m


def _body() -> np.ndarray:
    r0, r1, c0, c1 = BODY
    return _rect(r0=r0, r1=r1, c0=c0, c1=c1)


def test_recovers_a_known_contact_line():
    est = _contact_from_mask(_body(), BOX)
    assert est is not None
    u_c, v_c, l_x, r_x = est

    # The tyre line is row 80; p92 of a uniform block sits just above it.
    assert 74.0 <= v_c <= 80.0, v_c
    # Horizontal centre of the contact band, not of the box.
    assert abs(u_c - 49.5) <= 1.5, u_c
    # p5..p95 of the band spans most of the true 40 px width.
    assert 30.0 <= (r_x - l_x) <= 40.0, (l_x, r_x)
    assert l_x < u_c < r_x


def test_thin_shadow_tail_stays_on_the_tyre_line():
    """
    The failure mode the percentile exists for: a cast shadow leaking the mask
    downward. A realistic thin tail must not move the contact point at all.
    """
    m = _rect(r0=81, r1=111, c0=55, c1=59, base=_body())   # 30x4 px shadow tail

    est = _contact_from_mask(m, BOX)
    assert est is not None
    u_c, v_c, _, _ = est

    assert m.nonzero()[0].max() == 110          # the tail really is in the mask
    assert 74.0 <= v_c <= 82.0, v_c             # ...and the estimate ignores it
    assert abs(u_c - 49.5) <= 2.0, u_c          # not dragged toward the tail's x


def test_fat_shadow_tail_degrades_but_beats_max_v():
    """
    A tail large enough to reach past the 92nd percentile DOES move the estimate.
    Recorded here on purpose: the percentile bounds the error to the tail SAM
    mislabelled, it does not eliminate it. `max(v)` is unbounded by comparison.
    """
    m = _rect(r0=81, r1=131, c0=55, c1=63, base=_body())   # 50x8 px, ~20% of area

    est = _contact_from_mask(m, BOX)
    assert est is not None
    _, v_c, _, _ = est

    true_line = 80.0
    err_pct = abs(v_c - true_line)
    err_max = abs(float(m.nonzero()[0].max()) - true_line)
    assert err_pct < err_max, (v_c, err_pct, err_max)
    assert err_pct <= 0.7 * err_max


def test_vetting_rejects_degenerate_masks():
    ok, why = _vet_mask(_body(), BOX)
    assert ok and why == "ok"

    ok, why = _vet_mask(np.zeros((H, W), bool), BOX)
    assert not ok and why == "empty_mask"

    # A windscreen-sized fragment.
    ok, why = _vet_mask(_rect(r0=50, r1=55, c0=48, c1=52), BOX)
    assert not ok and why == "mask_too_small_vs_box"

    # SAM grabbed the whole carriageway.
    ok, why = _vet_mask(_rect(r0=0, r1=H, c0=0, c1=W), BOX)
    assert not ok and why == "mask_too_large_vs_box"

    # A mask that mostly lives OUTSIDE its own prompt: part of the vehicle plus a
    # long shadow across the carriageway. Sized to stay inside the area bounds, so
    # containment is what does the rejecting here.
    m = _rect(r0=60, r1=80, c0=40, c1=60)                      # 400 px inside the box
    m = _rect(r0=85, r1=105, c0=100, c1=150, base=m)           # 1000 px outside it
    ok, why = _vet_mask(m, BOX)
    assert why == "mask_escapes_box", why
    assert not ok


def test_contact_estimate_needs_enough_pixels():
    assert _contact_from_mask(_rect(r0=79, r1=81, c0=49, c1=51), BOX) is None
    assert _contact_from_mask(np.zeros((H, W), bool), BOX) is None


class _Row:
    """Stand-in for a boxes.csv itertuples row."""

    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


def test_masks_pair_positionally_then_by_iou():
    rows = [_Row(0, 0, 20, 20), _Row(100, 100, 140, 140)]
    a = _rect(r0=2, r1=18, c0=2, c1=18)
    b = _rect(r0=102, r1=138, c0=102, c1=138)

    # Normal path: equal counts, prompt order preserved.
    assert [i for i, _ in _pair_masks_to_boxes([a, b], rows)] == [0, 1]

    # Count mismatch: match by overlap rather than guess by position. The single
    # mask belongs to the SECOND box, and mispairing here would silently corrupt
    # two trajectories at once.
    pairs = _pair_masks_to_boxes([b], rows)
    assert [i for i, _ in pairs] == [1]

    # A mask overlapping nothing is dropped, not assigned.
    assert _pair_masks_to_boxes([_rect(r0=180, r1=195, c0=180, c1=195)], rows) == []


def _obs(track_id: int, frames, xs, ys) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": track_id,
            "class": "car",
            "frame": frames,
            "timestamp": [f / 30.0 for f in frames],
            "x": xs,
            "y": ys,
            "confidence": 0.9,
            "world_x": np.nan,
            "world_y": np.nan,
            "footprint_m": np.nan,
            "x_seg": np.nan,
            "y_seg": np.nan,
            "footprint_seg_m": np.nan,
            "contact_source": "bbox",
        }
    )


def test_missing_masks_leave_the_bbox_point_untouched():
    raw = _obs(1, [0, 1, 2], [100.0, 101.0, 102.0], [200.0, 201.0, 202.0])
    out, stats = apply_contact_refinement(raw.copy(), pd.DataFrame(), projector=None)

    assert stats["applied"] is False
    assert out["x_seg"].isna().all()
    assert (out["contact_source"] == "bbox").all()
    pd.testing.assert_series_equal(out["x"], raw["x"])
    pd.testing.assert_series_equal(out["y"], raw["y"])


def test_gaps_interpolate_the_offset_not_the_position():
    """
    A rejected or unsegmented frame must not snap back to the box point: that
    discontinuity is what a centred difference turns into a phantom speed spike.
    The OFFSET is bridged instead, and tagged so the CSV stays honest.
    """
    frames = [0, 1, 2, 3, 4, 5]
    xs = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    ys = [200.0] * 6
    raw = _obs(7, frames, xs, ys)

    # Masks on the first and last-but-one frame only; offset +2 px then +6 px.
    refined = pd.DataFrame(
        {
            "frame": [0, 4],
            "track_id": [7, 7],
            "x_seg": [102.0, 146.0],
            "y_seg": [198.0, 194.0],
            "seg_l_x": [92.0, 136.0],
            "seg_r_x": [112.0, 156.0],
        }
    )
    out, stats = apply_contact_refinement(raw.copy(), refined, projector=None)
    out = out.sort_values("frame").reset_index(drop=True)

    assert stats["from_mask"] == 2
    assert stats["from_interpolated_offset"] == 4
    assert stats["kept_bbox"] == 0

    src = list(out["contact_source"])
    assert src == ["mask", "mask_interpolated", "mask_interpolated",
                   "mask_interpolated", "mask", "mask_interpolated"]

    # The x offset runs 2 -> 6 linearly across frames 0..4, then holds for frame 5.
    dx = (out["x_seg"] - out["x"]).to_numpy()
    assert np.allclose(dx, [2.0, 3.0, 4.0, 5.0, 6.0, 6.0]), dx

    # The box point itself is never overwritten - that is what keeps the A/B honest.
    assert np.allclose(out["x"].to_numpy(), xs)
    assert np.allclose(out["y"].to_numpy(), ys)

    # Interpolated frames keep a contact-line width, so footprint_m stays derivable.
    assert np.allclose((out["seg_r_x"] - out["seg_l_x"]).to_numpy(), 20.0)


def test_a_track_with_no_masks_keeps_bbox_while_its_neighbour_is_refined():
    raw = pd.concat(
        [
            _obs(1, [0, 1], [100.0, 101.0], [200.0, 201.0]),
            _obs(2, [0, 1], [300.0, 301.0], [400.0, 401.0]),
        ],
        ignore_index=True,
    )
    refined = pd.DataFrame(
        {"frame": [0, 1], "track_id": [1, 1], "x_seg": [103.0, 104.0],
         "y_seg": [199.0, 200.0], "seg_l_x": [93.0, 94.0], "seg_r_x": [113.0, 114.0]}
    )
    out, stats = apply_contact_refinement(raw.copy(), refined, projector=None)

    t1 = out[out["track_id"] == 1]
    t2 = out[out["track_id"] == 2]
    assert (t1["contact_source"] == "mask").all()
    assert (t2["contact_source"] == "bbox").all()
    assert t2["x_seg"].isna().all()
    assert stats["kept_bbox"] == 2


def test_use_bbox_records_the_contact_point_without_applying_it():
    raw = _obs(1, [0, 1], [100.0, 101.0], [200.0, 201.0])
    refined = pd.DataFrame(
        {"frame": [0, 1], "track_id": [1, 1], "x_seg": [105.0, 106.0],
         "y_seg": [198.0, 199.0], "seg_l_x": [95.0, 96.0], "seg_r_x": [115.0, 116.0]}
    )
    out, stats = apply_contact_refinement(raw.copy(), refined, projector=None, use="bbox")

    assert stats["applied"] is False
    assert "NOT used" in stats["outcome"]
    assert out["x_seg"].notna().all()          # recorded, so the A/B is still possible
    assert out["world_x"].isna().all()         # but nothing metric was changed


def test_metrics_report_the_row_band_trend():
    """
    The falsifiable prediction: shift grows toward the top of the frame. Feed a
    synthetic set that HAS that structure and confirm the metric surfaces it.
    """
    n = 240
    y = np.linspace(50.0, 950.0, n)            # top of frame -> bottom
    shift = 30.0 * (1.0 - (y - 50.0) / 900.0)  # large at the top, ~0 at the bottom
    obs = pd.DataFrame(
        {
            "track_id": np.arange(n) % 12,
            "class": "car",
            "frame": np.arange(n),
            "x": 500.0,
            "y": y,
            "x_seg": 500.0 + shift,
            "y_seg": y,
            "contact_source": "mask",
        }
    )
    m = contact_metrics(obs, projector=None, n_row_bands=4)

    assert m["refined"] == n
    bands = list(m["by_image_row_band"].values())
    assert len(bands) == 4
    vals = [b["median_shift_px"] for b in bands]
    assert vals == sorted(vals, reverse=True), vals   # decreasing top -> bottom
    assert "falsifies" in m["row_band_note"]


def test_metrics_survive_an_unrefined_run():
    obs = pd.DataFrame({"track_id": [1], "class": ["car"], "frame": [0], "x": [1.0], "y": [2.0]})
    m = contact_metrics(obs, projector=None)
    assert m["refined"] == 0
    assert "median_shift_px" not in m


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
