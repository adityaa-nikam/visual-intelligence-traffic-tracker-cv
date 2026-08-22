# Issues found and how we solved them

Every problem hit during the build, the diagnosis, and the fix. This is the
engineering record - each entry is something we found by interrogating our own
output rather than trusting it.

---

## 1. The source video was unusable as-is

**Problem.** 4K, 6.5 GB, ~400 s. At `imgsz=1280` on CPU a single frame took ~0.9 s,
and decoding 4K alone was a bottleneck. A full run would have taken hours.

**Diagnosis.** Resolution was buying us nothing: at ~70 m altitude a car is already
~70 px long at 1920 wide, well inside YOLO's usable range.

**Fix.** Cut a 60 s clip and downscale 4K -> 1920 with ffmpeg.

```bash
ffmpeg -ss 120 -i source.MP4 -t 60 -vf scale=1920:-2 -c:v libx264 -preset ultrafast -crf 26 -an -sn data/intersection.mp4
```

**Why this is safe, not a shortcut.** Focal length *in pixels* scales with frame
width: `f_px = (focal_35mm / 36) * image_width`. Halve the width and both `f_px`
and the pixel coordinates halve, so the ground projection is **mathematically
identical**. Downscaling costs detection recall on small objects, not metric accuracy.

---

## 2. No GPU, and we could not install one

**Problem.** `torch.cuda.is_available()` returned `False`. CPU inference ran at
**1.13 fps** (0.89 s/frame).

**Attempted fix.** Install a CUDA build:
`pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126`

**It failed.**
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```
`shutil.disk_usage` showed only **6.5 GB free of 264 GB** on C:. Worse, pip had
already partially upgraded torch to `2.13.0+cpu` (`cuda compiled: None`).

**Decision: stay on CPU.** Rolling the dice on a torch reinstall with no disk
headroom risked breaking a *working* install mid-hackathon. Two things made this
an acceptable call rather than a compromise:

1. **Architecture absorbs it.** Detection is the only expensive stage and runs
   **once**. Every analytic reads `trajectories.csv`, so re-analysis is seconds
   (see issue 8). Levels 2-5 do not re-pay the detection cost.
2. **Colab covers the GPU need at zero risk.** `Traffic_Intelligence_Colab.ipynb`
   runs the identical code on a free T4 at ~20-25 fps.

**Side lesson.** A first attempt to measure disk usage with `du -sh` on the Temp
directory hung past a 120 s timeout. `shutil.disk_usage` answered instantly - use
the syscall, not a directory walk.

---

## 3. COCO has no LGV or HGV class

**Problem.** The deliverable asks for LGV vs HGV. COCO gives `truck` and `bus`,
full stop. Claiming otherwise would be fabricating a model capability.

**Fix.** Split `truck` detections by **measured ground footprint**, which the
telemetry calibration gives us for free: project both bottom corners of the box to
the ground plane and take the p90 of the resulting width across the track's life.
Above `lgv_hgv_split_m: 7.0` -> HGV, below -> LGV.

**The honesty mechanism.** Every such track carries
`class_source=size_heuristic_from_calibration` in `track_summary.csv`, and the
dashboard prints the caveat next to the class chart. It is never presented as a
model output. Result on the 60 s clip: 18 LGV, 3 HGV.

---

## 4. Per-frame class flicker double-counted road users

**Problem.** YOLO changes its mind between frames - the same vehicle is `car` in
one frame and `truck` in the next. Counting per-frame detections inflates totals
and produces nonsense class breakdowns.

**Fix.** Class is resolved **per track, once**, by majority vote across every
observation of that track (`resolve_track_classes`), then written back to all rows.
One physical road user = one track = one class = one count.

---

## 5. Conflict detection fired 12 times in 2 seconds, with an impossible TTC

**Problem.** The first smoke test reported 12 "potential conflicts" in a 2-second
window, with `min_ttc_s: 0.01` - physically impossible.

**Diagnosis.** Constant-velocity CPA amplifies speed noise. Cars travelling in a
line have a large relative-velocity magnitude `|dv|` purely from jitter, while
their actual **closing rate along the line of centres** is near zero. The CPA
formula happily returns a tiny `t_cpa` for that.

**Fix, in two parts:**

1. **Require genuine closing.** Project the relative velocity onto the line of
   centres and require it to exceed `min_closing_speed_kph: 6.0`. Plus a
   `ttc_min_s: 0.2` floor below which a value is noise by definition.
   ```python
   closing = -np.dot(dp, dv) / dist
   if rel_speed > 1e-3 and closing >= min_closing:
       t_cpa = -np.dot(dp, dv) / rel_speed**2
   ```
2. **Require persistence.** A real conflict develops over ~1 s; jitter fires for
   one frame. Count trigger frames per pair and drop anything under
   `min_persistence_frames: 5`.

**Result.** 12 -> 10 conflicts, `min_ttc` 0.01 -> 0.23 s, `frames_triggered` 6-50.

**Then we swept the threshold to check the survivors were not artifacts.** Conflict
count stayed stable from `ttc_max` 2.5 s all the way down to 1.5 s - meaning the
survivors are genuinely tight encounters, not pairs sitting just under a threshold.
Locked in `ttc_max_s: 2.5`, `min_approach_speed_kph: 10.0`.

**The sweep cost zero GPU time** - it re-ran `find_interactions` against the
existing `trajectories.csv`. That is the architecture paying off in practice.

---

## 6. On the full clip, 357 of 390 pairs were flagged as conflicts (91%)

**Problem.** The filters from issue 5 held up on 2 seconds but not on 60. Claiming
357 near-collisions in one minute is not credible.

**Diagnosis.** We looked at the distribution instead of guessing. 73% of conflicts
had TTC < 1.0 s - **not** the signature of threshold noise, which would pile up
*near* the threshold. So these were geometrically real. But:

| | median |
|---|---|
| closing speed of "critical" conflicts | **9.6 km/h** |
| miss distance at CPA | **3.34 m** |

Two vehicles converging at 2.7 m/s that will pass 3.3 m apart are **creeping in a
queue**, not in conflict. The ~1.5 s TTC threshold from the traffic-conflict
literature assumes lane-disciplined traffic; in dense mixed traffic low TTC is
*normal operating behaviour*.

**Fix: grade jointly on TTC and closing speed.** Closing speed is the proxy for how
much kinetic energy an evasive manoeuvre must absorb.

```
critical : TTC <= 1.5s AND closing >= 20 km/h
serious  : TTC <= 1.5s AND closing >= 12 km/h   (>= 8 km/h if a vulnerable user)
close    : passed every geometric filter but low-energy -> routine density
```

Only critical + serious are reported as conflicts. "Close interaction" is still
counted and exported, because in dense traffic it is a genuine *density* signal -
it just is not a safety event. We also report **rates** (per minute, per 100 road
users) so the number is comparable across clips.

**Result.** 357 -> 129 conflicts.

---

## 7. The remaining conflicts were normal oncoming traffic

**Problem.** After issue 6, one pattern stood out in the critical list: track `855`
appeared in four separate "head-on" conflicts and `663` in three. A vehicle having
four head-on conflicts in a minute is not plausible.

**First hypothesis - wrong-way driving - was wrong.** We checked the bearings:
tracks 855 and 663 average ~290 deg, and the scene's dominant flows are 270 deg and
90 deg (a two-way east-west road). 290 deg sits *inside* the main 270 deg stream.
They were not driving against traffic; they were just fast (25 and 32 km/h) and
therefore met a lot of oncoming vehicles.

**Real diagnosis.** We compared miss distance by interaction geometry:

| geometry | median `d_cpa` |
|---|---|
| crossing | 1.88 m |
| following | 1.97 m |
| merging | 2.04 m |
| **head-on** | **3.39 m** |

33 of 51 head-on conflicts had `d_cpa > 2.5 m`. A traffic lane is ~3.0-3.5 m wide
and a car is ~1.8 m wide, so **3.4 m is exactly the centre-to-centre separation of
two vehicles passing normally in adjacent opposing lanes.**

**Root cause.** The conflict envelope was built from `CLASS_RADIUS_M`, an
*omnidirectional* radius of 1.9 m for a car - effectively a half-*length*. Summed
for two cars that is a 3.8 m envelope, **wider than a lane**. Normal oncoming
traffic fell inside it.

**Fix.** Size the conflict envelope on **lateral half-width** (`CLASS_HALF_WIDTH_M`,
0.9 m for a car) with a 1.15 noise margin -> ~2.1 m for a car pair. That admits
genuine collision courses and rejects adjacent-lane passing. The omnidirectional
radius is retained for the separate sustained-proximity test, where it is correct.

**Result.** 129 -> **51 conflicts (18 critical + 33 serious)**, and every surviving
`d_cpa` is now <= 2.21 m across all four geometries. The final funnel:

```
390 pairs within 25 m
  -> 138 pass TTC + lateral-envelope test
  -> 51 graded conflicts (18 critical, 33 serious)
  -> 87 reclassified as close interactions (density, not safety)
```

The survivors are convincing: a rear-end risk at 0.86 m miss sustained 68 frames,
a bus merging at 1.0 m, a car/motorcycle crossing at 0.83 m.

---

## 8. Re-running analytics required a 25-minute re-detection

**Problem.** Fixing issues 6 and 7 changed only the analytics layer, but
`run_pipeline.py` always started from YOLO. Re-deriving the numbers meant 25 min of
CPU detection for a change that touched no pixels.

**Fix.** Added `--from-trajectories`, which skips detection and tracking entirely
and re-derives every analytic from a saved `trajectories.csv`. Bounding boxes are
now persisted to `boxes.csv` too, so even the annotated video can be re-rendered
without re-detecting.

```bash
python run_pipeline.py --from-trajectories outputs/trajectories.csv --tag graded
```

Runtime: **seconds instead of 25 minutes.** This is the trajectory-first
architecture made executable - and it is how Levels 2-5 will add analytics without
rebuilding detection.

---

## 9. A 67.7 km/h top speed at an intersection averaging 17 km/h

**Problem.** `max_speed` reported 67.7 km/h while p85 was 23.6 km/h. Either the
calibration was broken or a track was corrupted.

**Diagnosis.** Neither - it is a tail artifact, and the numbers proved it:

| percentile | speed |
|---|---|
| p50 | 3.1 km/h (includes stopped vehicles) |
| p85 | 20.4 km/h |
| p99 | 32.8 km/h |
| p99.9 | 37.2 km/h |
| max | 67.7 km/h |

Only **13 observations out of 69,067 (0.019%)**, across 4 of 292 tracks, exceeded
40 km/h - a few frames of box jitter on fast-moving vehicles.

**Fix.** Report `p85_speed` as the headline figure (which is also what traffic
engineering actually uses) and add `robust_max_speed_p99_5`. The raw max is retained
in `summary.json` with an explicit
`max_speed_caveat: "raw max is jitter-sensitive; use p85 / p99.5 for reporting"`.
We did not silently clip the data - we changed which statistic we quote.

---

## 10. Track IDs did not survive occlusion

**Problem.** Default ByteTrack settings dropped IDs when a vehicle passed behind a
tree or bus, or sat still long enough at a red light to be treated as gone. Each
recovery became a *new* ID, inflating counts and fragmenting trajectories.

**Fix.** The single highest-value tuning change was `track_buffer: 90` in
`config/bytetrack.yaml` - ~3 s at 30 fps - plus a lowered `track_low_thresh: 0.08`
so weak detections still feed re-association.

**Measured result** on the 60 s clip: **2,688 gap recoveries** (a track re-linked
after missed detections rather than reborn), longest bridged gap **3.07 s**, mean
track life 9.85 s, longest 59.96 s - i.e. tracks surviving the entire clip.

**Why we report this metric.** We have no hand-labelled ground truth, so we cannot
compute MOTA or IDF1, and we do not quote one. Gap recoveries are the honest,
directly measurable evidence for stable identity.

---

## 11. Stationary vehicles were being reported as incidents

**Problem.** A vehicle stopped for 20 s at a red light is not a breakdown, but a
naive stationary detector reports both identically.

**Fix.** Stationary detection cross-references the queue detector. If a stopped
vehicle is inside a detected queue cluster, the event description says so
explicitly. The event type is `stationary_vehicle_candidate` - never
"stalled vehicle" or "incident". One clip cannot distinguish the two causes, so we
surface the evidence and let a human decide.

---

## 12. Wrong-way detection with no HD map

**Problem.** Detecting illegal wrong-way driving requires knowing the legal
direction of each carriageway. We have no HD map.

**Fix.** Learn the dominant flow **from the data**: bin the scene into a grid and
compute the circular mean bearing of all traffic per cell. Flag tracks deviating
more than `wrong_way_deviation_deg: 130` from their own cell's dominant flow.

**The honest framing.** The event type is `against_dominant_flow`, not
"wrong_way_violation". It is a flow anomaly. This also correctly avoids false
positives on legitimate turning traffic, which follows the local flow.

---

## 13. `half=True` spammed a deprecation warning every frame

**Problem.** Newer Ultralytics warns `'half' is deprecated ... Use 'quantize'
instead` on every `model.track()` call - 1,798 lines of noise, hiding real output.

**Fix.** Only pass the key when it is actually enabled, so CPU runs never trigger it:
```python
if det_settings.get("half"):
    kw["half"] = True
```

---

## 14. RuntimeWarnings from short tracks

**Problem.** `Mean of empty slice` / `All-NaN axis encountered` from
`np.nanmean`/`np.nanmax` on tracks too short to have valid smoothed kinematics.

**Assessment.** Cosmetic - the NaN propagates correctly and those tracks are
filtered by `min_track_frames: 8` downstream. Suppressed at the run level with
`python -W ignore` rather than spending build time on it. Logged here because it
is a real (if harmless) rough edge, not something we pretend is clean.

---

## 15. A background run looked hung, and wasn't

**Problem.** The full run's log file sat at 0 bytes for minutes. It looked stalled.

**Diagnosis.** Not a hang - `grep -vE` in the pipeline was **buffering** stdout.
`tasklist | grep python` showed a live process at 245,800 K RSS, working normally.

**Lesson.** Verify with process state, not log output, before killing a long job.

---

## 16. The annotated video would not play in a browser

**Problem.** OpenCV writes `mp4v`, which browsers and the Streamlit player refuse.
The 1920p output was also 382 MB.

**Fix.** Re-encode to H.264 once, and have the dashboard prefer `annotated_web.mp4`
when it exists, falling back to the raw file and showing the exact ffmpeg command
if it does not.

```bash
ffmpeg -i outputs/annotated.mp4 -vcodec libx264 -crf 28 -pix_fmt yuv420p outputs/annotated_web.mp4
```

---

## 17. Every position in the system belonged to no part of the vehicle

**Problem.** Nothing crashed and no number looked wrong. But `calibration.py` had a
comment conceding *"a box bottom is still an approximation"*, and that approximation was
the anchor for **every** metre, km/h and bearing the system reported.

**Diagnosis.** Computed from the telemetry rather than assumed. The gimbal sits at
**-63 deg pitch, not nadir**. With `f_px = (24/36) * 1920 = 1280`, the vertical FOV is
45.7 deg and the optical axis is 27 deg off nadir, so the frame spans **4.1 deg off-nadir
at the bottom edge to 49.9 deg at the top**. An elevated point projects displaced
radially outward from the nadir by `height * tan(off-nadir)`:

| object | h | bottom (4 deg) | centre (27 deg) | top (50 deg) |
|---|---|---|---|---|
| car | 1.5 m | 0.11 m | 0.76 m | 1.78 m |
| pedestrian | 1.7 m | 0.12 m | 0.87 m | 2.02 m |
| bus / hgv | 3.2 m | 0.23 m | 1.63 m | 3.80 m |

An axis-aligned box wraps the whole silhouette **including the roof**. Its bottom edge
`y2` is the lowest silhouette pixel - roughly the near tyre line, about right - but its
horizontal centre `(x1+x2)/2` is the centre of a silhouette stretched by that radial roof
displacement. So `(cx, y2)` is a mongrel point: its `v` comes from the wheels and its `u`
from a roof-contaminated average.

Three things follow, and none of them depend on the exact magnitude:

1. The bias grows **~17x from the bottom of the frame to the top**, so it does not cancel
   as a vehicle crosses the scene. A monotonically growing position error **is** a
   velocity error.
2. It is **orientation-dependent**, so a turning vehicle's box centre moves even when its
   true footprint centre does not - spurious motion exactly where turning counts and
   conflict detection are most sensitive.
3. The conflict envelope for two cars is **2.07 m**. In the upper half of the frame the
   roof-displacement scale is the same order as the whole envelope, which puts it squarely
   in the way of the conflict work in #5/#6/#7.

**Fix.** `pipeline/contact_points.py`: prompt the MobileSAM weights *already* in the
project with detection boxes, and take the tyre-contact line of the resulting instance
mask. Contact row = **92nd percentile** of mask rows, not `max(v)`, because the dominant
failure mode on sunlit aerial imagery is a cast shadow leaking the mask downward -
`max(v)` follows the shadow tip an unbounded distance, a percentile bounds the error to
the tail SAM actually mislabelled. `u` is the median of a 6 px band above that row; the
band's p5/p95 give a true contact-line width, which also replaces the box-corner proxy
behind `footprint_m` and the LGV/HGV split from #3.

Three decisions worth recording:

- **A second pass over `boxes.csv`, not a call inside the tracker.** Costs one extra video
  decode, and in exchange `trajectories.csv` stays the single contract, an existing run
  can be re-anchored with **no re-detection** (the payoff from #8, applied to
  segmentation), and a SAM failure cannot destroy a finished detection run.
- **Both anchors are kept.** `x, y` are never overwritten; `x_seg, y_seg` sit beside them
  and one config key chooses which feeds the world coordinates. Without this the
  "improvement" would be unfalsifiable - you cannot compare against a baseline you have
  overwritten.
- **Rejected frames interpolate the box->mask OFFSET, they do not revert.** Snapping one
  rejected frame back to the box point would reintroduce a 0.1-1.8 m discontinuity
  mid-track, and a discontinuity is exactly what a centred difference turns into a
  phantom speed spike. Mixing anchors within one track is worse than using either
  consistently.

**Falsifiable check, built in from the start.** The geometry predicts the measured shift
grows monotonically from the bottom of the frame to the top and scales with object
height. `contact_metrics` reports the shift per image-row band and per class, and the
dashboard states plainly whether the prediction held - a flat profile means the estimator
is wrong, not that the feature is fine. `path_quality` (speed jitter, footprint CV,
straight-path RMS lateral deviation on raw coordinates) is computed on **every** run, so
the bbox-vs-mask A/B is a diff of two `summary.json` files rather than bespoke tooling.

**Lesson.** The strongest version of "interrogate your own output" is to interrogate an
input that never produced a visible error. This one was found by reading a comment, doing
the trigonometry it implied, and discovering the quantity the whole system rests on
corresponds to no physical feature of the object being measured.

---

## The pattern behind all of these

Six of the seventeen issues above (4, 5, 6, 7, 9, 11) were found by **interrogating
our own output** - looking at distributions, cross-tabulating by category, and
asking whether a number was physically plausible - not by seeing a crash. Issue 17
goes one step further: it was found by interrogating an *input* that never produced a
visible error at all.

The conflict detector is the clearest example. It never threw an error. It produced
confident, well-formatted, plausible-looking output at every stage, and it was
wrong three times in a row for three completely different reasons: speed noise
(issue 5), then dense-traffic norms (issue 6), then a geometry error that made
normal oncoming traffic look like head-on collisions (issue 7). Each one was only
findable by asking "is this number actually believable?" and then checking a
distribution instead of trusting the answer.
