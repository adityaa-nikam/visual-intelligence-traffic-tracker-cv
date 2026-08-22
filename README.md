# Drone Traffic Intelligence - Level 1

**Detection and tracking of every road user in drone footage, with stable identities
through occlusion, and a reusable trajectory layer that all analytics are built on.**

Aerial video is not just "CCTV from higher up". A drone sees a whole intersection in
one frame, in one coordinate system - so it can measure the *relationships* between
road users, which is exactly what fixed cameras cannot do. This build leans into that.

The differentiator: **DJI aircraft embed per-frame flight telemetry in the MP4 as a
subtitle track.** Altitude, focal length and gimbal angles are enough to project any
pixel onto the road plane, so this system reports **real metres, real km/h and real
compass bearings** instead of pixel velocities.

---

## Quick start

```bash
pip install -r requirements.txt
```

Prepare a clip (downscaling 4K -> 1920 costs no metric accuracy - see ISSUES.md #1):

```bash
ffmpeg -ss 120 -i "source.MP4" -t 60 -vf scale=1920:-2 -c:v libx264 -preset ultrafast -crf 26 -an -sn data/intersection.mp4
```

Extract the telemetry (this is what unlocks metric output):

```bash
ffmpeg -i "source.MP4" -map 0:s:0 data/intersection_full.srt
```

Run the pipeline, then the dashboard:

```bash
python run_pipeline.py
```

```bash
streamlit run app.py
```

**No GPU?** Open `Traffic_Intelligence_Colab.ipynb` in Google Colab and pick a T4
runtime - the same code runs ~20x faster. It reads the video straight from Google
Drive, so there is nothing to upload.

Useful flags:

```bash
python run_pipeline.py --max-frames 60 --tag smoke      # fast smoke test
python run_pipeline.py --no-video                       # analytics only
python run_pipeline.py --from-trajectories outputs/trajectories.csv --tag retune
```

That last one is the important one - see [Architecture](#architecture) below.

---

## Architecture

```
                      drone video (MP4)
                             |
        +--------------------+--------------------+
        |                                         |
  tx3g subtitle track                        video frames
        |                                         |
  pipeline/calibration.py                 pipeline/detector.py   (YOLOv8)
  altitude, focal length,                 pipeline/tracker.py    (ByteTrack)
  gimbal angles                                   |
        |                                    stable track IDs
  pinhole -> ground plane                         |
  projection (metres)                             |
        +--------------------+--------------------+
                             |
                pipeline/contact_points.py   (MobileSAM instance masks:
                  bbox bottom-centre  ->      second pass over boxes.csv,
                  true tyre-contact line      before any kinematics)
                             |
                  pipeline/trajectory.py
              ***  trajectories.csv  ***   <-- the contract
                             |
        +----------------+---+-----------+-------------+
        |                |               |             |
   analytics.py    interactions.py   anomalies.py  visualize.py
   counts, queues,  TTC, conflicts,   events.csv   annotated.mp4
   congestion,      headways                       trajectory map
   turning, dwell
```

**`trajectories.csv` is the architectural contract.** Detection and tracking are the
only expensive stages and they run exactly once. Every analytic is a pure function of
the trajectory table - none of them import YOLO, and none can see a video frame.

Three concrete consequences:

1. **Adding an analytic for Level 2+ never touches detection or tracking.** Write a
   function that takes the trajectory DataFrame; register it in `anomalies.DETECTORS`.
2. **Retuning is free.** `--from-trajectories` re-derives every analytic in *seconds*
   instead of 25 minutes. The conflict-threshold sweep in ISSUES.md #5 and the two
   conflict fixes in #6 and #7 were all done this way, with zero GPU time.
3. **A CPU-only machine is not a blocker.** You pay detection once.

---

## Metric calibration - how real-world units are obtained

`ffprobe` on the source reveals a `tx3g` subtitle stream. Each record looks like:

```
[focal_len: 24.00] [latitude: 18.566227] [longitude: 73.771846]
[rel_alt: 70.472 abs_alt: 607.273] [gb_yaw: -125.5 gb_pitch: -63.1 gb_roll: 0.0]
```

`pipeline/calibration.py` turns that into a ground projection:

1. **Focal length in pixels** from the 35 mm-equivalent value:
   `f_px = (focal_35mm / 36.0) * image_width`
2. **Camera rotation** from gimbal yaw/pitch/roll, in an ENU world frame
   (X east, Y north, Z up), with the aircraft at `Z = rel_alt`.
3. **Ray-cast** each pixel through the pinhole model onto the plane `Z = 0`.

A ray pointing above the horizon returns `NaN`, never a fabricated coordinate.

Every road user's **bottom-centre bbox point** is used as the road-contact point,
which is where the vehicle actually meets the ground plane.

### Validating calibration without ground truth

There is no survey to check against, so we validate by **physical plausibility** -
three independent checks that would all fail on a broken projection:

| Check | Measured | Verdict |
|---|---|---|
| **Hover stability** | altitude spread 0.023 m, pitch 0.0 deg, yaw 0.4 deg across 6,824 records | Aircraft is genuinely static -> a single fixed camera pose is justified |
| **Ground scale** | 0.0617 m/px at centre, 97.6 m scene width at 70.47 m altitude | Correct order of magnitude for a 24 mm-equivalent lens at that height |
| **Speed distribution** | mean 16.7, median 16.8, p85 23.6 km/h | Exactly right for a busy urban intersection. A broken projection gives 300 km/h or 2 km/h |

**Without telemetry the system degrades honestly.** It falls back to pixel-space
analytics and **refuses to print km/h at all** rather than inventing a scale factor.

---

## Detection and tracking

**Detector:** YOLOv8s (COCO), `imgsz=1280`, `conf=0.25`, restricted to the six
relevant COCO classes - person, bicycle, car, motorcycle, bus, truck. No custom
training: within a 4-hour budget a pretrained model with tuned inference beats a
half-trained custom one.

**Tracker:** ByteTrack (bundled with Ultralytics) via `model.track(persist=True)`.
It keeps low-confidence detections as association candidates instead of discarding
them, which is precisely what carries an ID through partial occlusion.

**The tuning that mattered most** is `track_buffer: 90` (~3 s at 30 fps) in
`config/bytetrack.yaml` - long enough to survive a bus passing in front of a car, or
a vehicle waiting at a red light.

### Identity stability, measured

| Metric | 60 s clip |
|---|---|
| Road users tracked | 292 |
| Observations | 69,505 |
| **Gap recoveries** (re-linked after missed detections) | **2,688** |
| Longest bridged occlusion | 3.07 s |
| Mean track life | 9.85 s |
| Longest track | 59.96 s (the full clip) |
| Tracks over 10 s | 99 |

We have no hand-labelled ground truth, so **we do not quote MOTA or IDF1** - we
cannot compute them. Gap recoveries are the honest, directly measurable evidence
that identities survive occlusion.

---

## Trajectory representation

One row per road user per frame in `trajectories.csv`:

| column | meaning |
|---|---|
| `track_id`, `class`, `frame`, `timestamp` | identity and time |
| `x`, `y` | bottom-centre of bbox, image space - **always** the box point |
| `x_seg`, `y_seg` | MobileSAM tyre-contact point (see below); `seg_l_x`/`seg_r_x` bound the contact line |
| `contact_source` | `mask`, `mask_interpolated` or `bbox` - which anchor produced this row |
| `world_x`, `world_y` | ground-plane projection of the **chosen** anchor, metres |
| `sx`, `sy` | smoothed ground coordinates (centred moving average) |
| `speed_kph`, `speed_px_s` | centred difference over a ~0.5 s baseline |
| `bearing_deg` | true compass bearing (metric); `heading_px_deg` in pixel space |
| `accel_kph_s` | for sudden-stop detection |
| `footprint_m` | measured ground width of the contact line (bbox bottom edge if unrefined) |
| `footprint_seg_m`, `box_w`, `box_h` | mask contact width, and the raw box dimensions |

Speed uses a **centred** difference over a ~0.5 s baseline: long enough to suppress
per-frame jitter, short enough to preserve real acceleration.

### Ground-contact point: bounding box vs instance mask

The bbox bottom-centre was the original anchor, and `calibration.py` conceded the
limitation in a comment: *"a box bottom is still an approximation."* On this footage it
is worse than it sounds. **The gimbal sits at -63 deg pitch, not nadir.** With
`f_px = 1280` on 1920x1080 the vertical FOV is 45.7 deg, so the frame spans **4.1 deg
off-nadir at the bottom edge to 49.9 deg at the top**, and an elevated point projects
displaced radially outward from the nadir by `height * tan(off-nadir)`:

| object | h | bottom of frame (4 deg) | centre (27 deg) | top (50 deg) |
|---|---|---|---|---|
| car | 1.5 m | 0.11 m | 0.76 m | 1.78 m |
| pedestrian | 1.7 m | 0.12 m | 0.87 m | 2.02 m |
| lgv | 2.5 m | 0.18 m | 1.27 m | 2.97 m |
| bus / hgv | 3.2 m | 0.23 m | 1.63 m | 3.80 m |

An axis-aligned box wraps the **whole silhouette, roof included**. Its bottom edge `y2`
is the lowest silhouette pixel (roughly the near tyre line, about right), but its
horizontal centre `(x1+x2)/2` is the centre of a silhouette *stretched by that radial
roof displacement*. So `(cx, y2)` **corresponds to no physical feature of the vehicle** -
its `v` comes from the wheels and its `u` from a roof-contaminated average. Three
consequences follow from the geometry regardless of the exact magnitude:

1. The bias grows **~17x from the bottom of the frame to the top**, so it does not cancel
   as a vehicle crosses the scene - it **injects false velocity** into `speed_kph` and
   `bearing_deg`.
2. It is **orientation-dependent**, so a *turning* vehicle's box centre shifts even when
   its true footprint centre does not - spurious motion exactly where turning counts and
   conflict detection are most sensitive.
3. For scale, the conflict envelope for two cars is **2.07 m**. In the upper half of the
   frame the roof-displacement scale is the same order as the entire envelope.

`pipeline/contact_points.py` replaces the anchor with the real tyre-contact line of a
**MobileSAM instance mask** - the same 40 MB model already used for the road area, but
prompted with detection *boxes* instead of points:

```
m      = mask pixels inside the prompt box
v_c    = percentile(v of m, 92)        # robust bottom, NOT max(v)
band   = m where v >= v_c - 6px        # the tyre-contact band
u_c    = median(u of band)             # contact point
l, r   = p5, p95 of u over band        # true contact-line width -> footprint_m
```

The percentile is the load-bearing choice: the dominant SAM failure mode on sunlit
aerial imagery is a **cast shadow leaking the mask downward**. `max(v)` would follow the
shadow tip an unbounded distance; a percentile bounds the error to the tail SAM actually
mislabelled. A thin tail is ignored entirely, a fat one degrades the estimate but still
beats `max(v)` - both cases are pinned in `tests/test_contact_points.py`.

**It runs as a second pass over `boxes.csv`, not inside the tracker.** That is what keeps
`trajectories.csv` the single contract (no analytic learns SAM exists), lets an
*existing* run be re-anchored with **no re-detection**, and means a SAM failure cannot
destroy a completed detection run. One `SAM(img, bboxes=[...])` call per frame carries
every box in that frame: one image encode, N cheap decodes.

**Both anchors are kept.** `x, y` are never overwritten; `segmentation.contact.use`
chooses which feeds `world_x/world_y`. So the improvement is *measured* rather than
asserted, every row records its own `contact_source`, and a bad result on other footage
is one config key from being reverted. Masks that fail vetting (area vs the prompt box,
containment inside it) do **not** snap back to the box point mid-track - the box->mask
*offset* is interpolated instead and tagged `mask_interpolated`, because a 0.1-1.8 m
discontinuity is precisely what a centred difference turns into a phantom speed spike.
Mixing anchors within one track is worse than using either anchor consistently.

The A/B, costing one detection run in total:

```bash
python run_pipeline.py --no-refine-contacts --tag bbox
```

```bash
python run_pipeline.py --from-trajectories outputs/trajectories_bbox.csv --refine-contacts --tag seg
```

`summary.json` then carries `contact_refinement` and `path_quality` for both runs, and
the dashboard renders the comparison. The **falsifiable** check is the one that matters:
the measured shift must grow monotonically from the bottom of the frame to the top and
scale with object height. A flat profile falsifies the method, and the dashboard says so
in that case rather than quietly showing a bar chart.

---

## Analytics

All in `pipeline/analytics.py`, `interactions.py`, `anomalies.py` - each a pure
function of the trajectory table.

**Counts and classification.** Class is resolved **per track by majority vote**, not
per frame, so detector flicker cannot double-count a vehicle.

**Stationary and dwell.** Sustained low-speed runs, cross-referenced against the
queue detector so a red light is not reported as an incident.

**Queues.** Single-linkage spatial clustering (`cKDTree` + union-find) of slow
vehicles per time bin. Length is the cluster's extent along its own principal axis
(via SVD), in metres.

**Congestion.** A transparent weighted score, stated in the output:
`0.45 * slow_fraction + 0.35 * speed_deficit + 0.20 * relative_density`.

**Turning movements.** Entry vs exit heading per trajectory. Tracks too short to
judge are excluded rather than forced into a bucket.

**Directional flow.** True 16-point compass distribution when calibrated.

**Interactions (surrogate safety).** For every pair within 25 m, solve for the
closest point of approach under constant velocity:

```
dp = p2 - p1,  dv = v2 - v1
t_cpa = -(dp . dv) / |dv|^2
d_cpa = |dp + t_cpa * dv|
```

Getting this to produce trustworthy output took **three separate fixes** - it is the
most interesting engineering in the project, and all three are documented in
[ISSUES.md](ISSUES.md) #5, #6 and #7. In short, a naive CPA implementation is wrong
in three independent ways:

1. **Speed noise** inflates `|dv|` for vehicles moving in a line -> require a real
   closing rate along the line of centres, plus multi-frame persistence.
2. **Dense-traffic norms**: a bare TTC <= 1.5 s test flags ~90% of all pairs, because
   queue creep is geometrically a "conflict" -> grade jointly on TTC **and closing
   speed**, which proxies the energy an evasive manoeuvre must absorb.
3. **Envelope geometry**: an omnidirectional radius (~half vehicle *length*, 3.8 m
   for two cars) is **wider than a traffic lane**, so normal oncoming traffic read as
   head-on collisions -> size the envelope on **lateral half-width** instead.

The final funnel on the 60 s clip:

```
390 pairs within 25 m
  -> 138 pass the TTC + lateral-envelope test
  -> 51 graded conflicts  (18 critical, 33 serious)
  -> 87 reclassified as close interactions (density signal, not safety events)
```

**Events.** `events.csv` carries a uniform schema across all detectors:
`event_type, track_id, secondary_track_id, timestamp, duration, x, y, severity,
value, unit, description`.

---

## Results (60 s, 1798 frames, urban intersection)

| | |
|---|---|
| Road users | **292** - 231 car, 18 LGV, 3 HGV, 2 bus, 9 motorcycle, 29 pedestrian |
| Motor vehicles / vulnerable users | 263 / 38 |
| Speed | mean 16.7, median 16.8, **p85 23.6** km/h |
| Queues | 91 detections, largest 22 vehicles, longest 78.7 m |
| Turning movements | 86 straight, 14 left, 14 right |
| Potential conflicts | **51** (18 critical, 33 serious) |
| Congestion | peak score 0.686 |
| Events | 387 across 8 types |

These are from the **bbox-anchored** run, i.e. `--no-refine-contacts`. Re-anchoring on
mask contact points moves positions by design, so every figure here will shift - the
conflict counts in particular, since they were earned through the three fixes in
ISSUES.md #5/#6/#7. They are regenerated from the refined run only *after* the
`contact_refinement` metrics confirm the shift moved in the predicted direction; a sharp
move in conflicts gets investigated rather than assumed to be the better number.

---

## Outputs

| File | Contents |
|---|---|
| `outputs/annotated.mp4` | Boxes, IDs, class, live speed, trails, conflict lines, HUD. Renders are never overwritten - later ones land as `_v2`, `_v3` |
| `outputs/trajectories.csv` | **The core artifact** - every observation of every road user |
| `outputs/segments.csv` | MobileSAM contact points per `(frame, track_id)`, with the contact-line bounds |
| `outputs/events.csv` | All detected events, uniform schema |
| `outputs/summary.json` | Headline metrics, calibration evidence, method notes, `contact_refinement` + `path_quality` |
| `outputs/track_summary.csv` | One row per road user |
| `outputs/trajectory_map.png` | Top-down trajectory plot with a metric scale bar |
| `outputs/road_mask.png` | Drivable-area overlay (MobileSAM + observed trajectories) |
| `queues / congestion / turning_movements / directional_flow / interactions / headways .csv` | Per-analytic tables |
| `outputs/boxes.csv` | Raw boxes, so video can be re-rendered - and trajectories re-anchored - without re-detecting |

---

## What we deliberately do not claim

Being explicit here is a design principle, not a disclaimer. Fabricated precision is
worse than an honest estimate.

| Area | We claim | We do **not** claim |
|---|---|---|
| **Classes** | The six COCO classes the detector actually outputs | That the model distinguishes LGV from HGV. **It does not.** That split is a footprint-size heuristic from calibration, tagged `class_source=size_heuristic_from_calibration` |
| **Speed** | "Estimated speed", from a telemetry-derived ground plane | Survey-grade. Assumes flat ground. Headline figure is p85, not the jitter-sensitive raw max |
| **Contact point** | The tyre-contact line of a MobileSAM instance mask where one passed vetting, tagged per observation in `contact_source`, with the measured shift reported in `summary.json` | That every object was segmented, or that the mask is exact. Masks that fail vetting keep an interpolated box->mask offset; a cast shadow reaching past the 92nd percentile of mask rows still moves the estimate, just far less than `max(v)` would |
| **Stationary** | "Stationary vehicle **candidate**", cross-referenced against queues | An incident. One clip cannot separate a breakdown from a red light |
| **Wrong way** | `against_dominant_flow` - movement against the *observed* flow of that part of the road, learned from data | A legal violation. We have no HD map of legal carriageway directions |
| **Conflicts** | "**Potential** conflict", graded, from a constant-velocity CPA projection | A collision, or a certified traffic-conflict-technique score. A constant-velocity model cannot know a driver was already braking |
| **Congestion** | A transparent weighted score for operator triage | A certified level of service - that needs lane geometry and capacity |
| **Coordinates** | Metres relative to the aircraft, in one common ground frame | Absolute survey coordinates. GPS is recorded, so results *could* be georeferenced |

Additional known limitations: the gimbal sits at -63 deg (oblique, not nadir), so
projection error grows toward the far field; the flat-ground assumption ignores road
camber and gradient; and detection recall drops for small or heavily occluded objects,
which no amount of downstream analytics can recover.

---

## Project layout

```
config/config.yaml        every tunable - nothing analytical is hard-coded
config/bytetrack.yaml     tracker params (track_buffer: 90 is the key one)
pipeline/calibration.py   DJI telemetry -> metric ground projection
pipeline/detector.py      YOLOv8 loading, class mapping, colours
pipeline/tracker.py       ByteTrack, yields per-frame records
pipeline/contact_points.py MobileSAM instance masks -> tyre-contact anchors
pipeline/road_mask.py     MobileSAM + observed traffic -> drivable area
pipeline/trajectory.py    kinematics, per-track class resolution
pipeline/analytics.py     counts, queues, congestion, turning, dwell, ID stability
pipeline/interactions.py  CPA / TTC, conflict grading, headways
pipeline/anomalies.py     event detectors -> events.csv
pipeline/visualize.py     annotated video + trajectory map
tests/                    contact-point estimator + merge tests (no SAM, no video)
run_pipeline.py           orchestration (7 stages)
app.py                     Streamlit dashboard
make_colab.py             regenerates the Colab notebook
ISSUES.md                 every problem hit, diagnosed and fixed
```

Tests need neither a GPU nor the video:

```bash
python -m pytest tests/ -q
```

**Extending for Level 2+:** write a function that takes the trajectory DataFrame and
returns event rows, register it in `anomalies.DETECTORS`, then run
`--from-trajectories`. No detection, no tracking, no re-processing.
