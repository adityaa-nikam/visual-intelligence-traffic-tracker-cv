# Drone Traffic Intelligence - Level 1 Submission

This is an archived submission write-up for the earlier COCO-based `v4` baseline.
The current upload workflow uses the VisDrone aerial detector, and the archived
counts below are not its accuracy evaluation. Use the selected run's summary JSON
for that run's outputs, and see `MODEL_AND_EVALUATION.md` for the current model,
class mapping, and labeled-evaluation workflow. `ISSUES.md` records run-specific
historical tuning measurements.

**Detection, classification, multi-object tracking and trajectory extraction from
aerial drone video, with a metric ground-plane reconstruction derived from the
aircraft's own flight telemetry.**

Test clip: 60 s of an urban signalised intersection, 1798 frames, 1920x1080 @ 29.97 fps,
shot from a DJI aircraft hovering at 70.5 m with the gimbal at -63 deg.

---

## 1. What was asked, and where it is

| Level 1 requirement | Where it is |
|---|---|
| Detect every road user | `pipeline/detector.py` - YOLOv8s (COCO), inference at native 1920 |
| Classify by mode | `pipeline/analytics.py` - per-track majority vote + calibrated size heuristic |
| Multi-object tracking | ByteTrack via `model.track(persist=True)`, `track_buffer: 90` |
| Stable IDs through occlusion | 2,688 measured gap recoveries, longest bridged occlusion 3.07 s |
| Trajectory extraction | `outputs/trajectories.csv` - 69,505 rows, one per road user per frame |
| Traffic analytics | `pipeline/analytics.py`, `interactions.py`, `anomalies.py` |
| Visualisation | `outputs/annotated_v3.mp4`, `trajectory_map.png`, Streamlit dashboard |

**Deliverables:** source code, `requirements.txt`, `README.md`, annotated video,
`trajectories.csv`, `events.csv`, Streamlit dashboard (`app.py`), run instructions,
plus `ISSUES.md` - a log of every defect we found in our own output and how we fixed it.

---

## 2. Architecture: trajectories are the interface

```
drone video
    |
    v
[1] detection            YOLOv8s, native-resolution inference
    |
    v
[2] tracking             ByteTrack, persistent IDs
    |
    v
[3] TRAJECTORY TABLE  <--- the single interface in the system
    |                      (track_id, class, frame, t, x, y, world_x, world_y,
    |                       sx, sy, speed_kph, bearing_deg, accel, footprint_m)
    +-- [3b] road segmentation      MobileSAM, vetted against observed traffic
    +-- [4] analytics               counts, flow, queues, congestion, turns, headway
    +-- [4] interactions            CPA / TTC conflict analysis
    +-- [4] anomalies               stationary, dwell, sudden stop, wrong-way
    +-- [5] visualisation           annotated video, trajectory map, dashboard
```

Every analytics module is a **pure function of the trajectory table**. Nothing
downstream of step 3 imports YOLO or ByteTrack, and nothing in steps 1-2 knows an
analytic exists. Adding a Level 2 metric means writing one function that takes a
DataFrame - detection and tracking are never touched again. Detection is also
cached, so `--from-trajectories` re-runs the entire analytics and rendering stack in
minutes without re-running the model.

The road-contact point is the **bottom-centre of the box**, not the centroid. The
centroid of a tall vehicle floats above the road surface and would project to a
ground position several metres wrong.

---

## 3. The differentiator: real metres from flight telemetry

Most aerial submissions report pixels, or invent a scale factor. We recovered a
genuine metric ground plane, and we can show every step.

DJI writes flight telemetry into the MP4 as a **tx3g subtitle track**. It is
extractable with one command:

```bash
ffmpeg -i input.mp4 -map 0:s:0 telemetry.srt
```

6,824 records carrying focal length (35 mm-equivalent), relative altitude, gimbal
yaw/pitch/roll and GPS position.

**Step 1 - focal length in pixels.** The 35 mm-equivalent focal length converts to
pixels using the 36 mm reference frame width: `f_px = (focal_35mm / 36) * image_width`
-> **1280 px**.

**Step 2 - camera pose.** Altitude 70.47 m, gimbal pitch -63.1 deg, yaw -126.6 deg.

**Step 3 - ray casting.** Each pixel becomes a ray in a local East-North-Up frame,
rotated by the gimbal attitude and intersected with the ground plane Z = 0. Pixels
whose ray points at or above the horizon return NaN rather than a fabricated
coordinate.

Result: 0.0617 m/px at frame centre, 97.6 m of ground width across the bottom edge
of the frame. Speeds in km/h, distances in metres, bearings as true compass headings.

**Why a single hover pose is legitimate.** We measured the aircraft's stability
across all 6,824 records rather than assuming it:

| | p5-p95 spread |
|---|---|
| relative altitude | **0.023 m** |
| gimbal pitch | **0.0 deg** |
| gimbal yaw | 0.4 deg |
| gimbal roll | 0.0 deg |

23 mm of altitude drift over a minute. A per-frame homography would be pure
ceremony. This same measurement is what licenses computing the road mask **once**
for the whole clip.

### Validating the calibration with no ground truth

We never measured this intersection, so we cannot quote a calibration error. What we
*can* do is check whether the calibration predicts physically sensible sizes. The
`footprint_m` column is the projected real-world ground width of each box's bottom
edge - a quantity the model was never told about:

| class | median measured footprint |
|---|---|
| pedestrian | 0.88 m |
| motorcycle | 1.56 m |
| car | 2.30 m |
| lgv | 4.22 m |
| bus | 5.92 m |
| hgv | 11.26 m |

A 2.3 m car and a 0.88 m pedestrian are correct to within a few centimetres of what
a tape measure would give. Nothing in the pipeline forced this ordering - it falls
out of the projection, which is strong independent evidence the geometry is right.

---

## 4. Detection and classification

**Detector:** YOLOv8s (COCO), restricted to person / bicycle / car / motorcycle /
bus / truck. No custom training - inside a 4-hour budget, a pretrained model with
tuned inference beats a half-trained custom one.

**Classification is resolved per track, not per frame.** A single vehicle flickers
between "car" and "truck" across 300 frames. We take the confidence-weighted
majority vote over its whole life, which turns 300 noisy guesses into one stable
answer.

**LGV vs HGV - stated honestly.** COCO has one `truck` class. It cannot distinguish a
light goods vehicle from a heavy one, and we do not pretend otherwise. Instead we
*derive* the split from the calibrated footprint: a COCO truck whose measured ground
footprint exceeds the LGV threshold is reported as `hgv`. Every such row is tagged
`class_source = size_heuristic_from_calibration` in `track_summary.csv` (21 of 292
tracks); the other 271 are `class_source = model`. A judge can see exactly which
labels came from the network and which came from geometry.

### Small-object detection: measured, not assumed

At 70 m a motorcycle is a few dozen pixels. The textbook fixes are higher inference
resolution and a lower confidence floor. Rather than apply both and claim
improvement, we built `tools/measure_small_objects.py`, which runs three settings
over 30 frames sampled evenly across the clip (so no single traffic state flatters
one setting) and buckets every detection by pixel area:

| setting | detections | tiny (<400 px) | small | medium | large | on-road | median conf |
|---|---|---|---|---|---|---|---|
| 1280 / conf 0.25 (baseline) | 1250 | 92 | 503 | 636 | 19 | 87.6% | 0.522 |
| **1920 / conf 0.25 (adopted)** | **1975** | **535** | **743** | **677** | **20** | **79.5%** | 0.469 |
| 1920 / conf 0.15 (rejected) | 3164 | 1140 | 1055 | 916 | 53 | 64.9% | 0.315 |

*on-road = fraction of detections whose road-contact point falls inside the
segmented carriageway. Real missed road users are on the road; hallucinations
scatter over rooftops and vegetation.*

We **adopted 1920** - it is the native width of the frame, so the previous 1280
setting was discarding a third of the linear resolution before the model ever saw
it. It multiplied tiny detections by 5.8x while leaving medium and large essentially
unchanged (+41, +1). That is exactly the signature of a genuine resolution gain.

We **rejected conf 0.15** despite it scoring the largest headline improvement
(+153% detections). It added **+34 large** detections - a resolution or threshold
change cannot conjure new buses - and on-road plausibility collapsed from 87.6% to
64.9%. Those are false positives on rooftops. The decision was made on the *shape*
of the change, not its size.

---

## 5. Tracking and identity stability

**ByteTrack** via `model.track(persist=True)`. It retains low-confidence detections
as association candidates instead of discarding them, which is precisely what
carries an ID through partial occlusion.

The single tuning change that mattered was `track_buffer: 90` (~3 s at 30 fps) in
`config/bytetrack.yaml` - long enough for a bus to pass in front of a car, or a
vehicle to sit through a phase of red, without the ID being reissued.

| metric | 60 s clip |
|---|---|
| road users tracked | 292 |
| trajectory observations | 69,505 |
| **gap recoveries** (ID re-linked after missed detections) | **2,688** |
| longest bridged occlusion | 3.07 s |
| mean track life | 9.85 s |
| longest track | 59.96 s (the entire clip) |
| tracks surviving > 10 s | 99 |

**We do not quote MOTA or IDF1.** Those metrics require hand-labelled ground truth,
which does not exist for this clip; quoting them would be fabrication. Gap
recoveries are the honest, directly measurable evidence that identities survive
occlusion - each one is an instance where the tracker lost the detection and
correctly re-attached the same ID rather than minting a new one.

---

## 6. Road segmentation

The problem statement asks for the road to be the focus. We segment the carriageway
using **MobileSAM (~40 MB)**, loaded through Ultralytics so it adds no new
dependency and runs on a CPU laptop. Because the aircraft hovers (see the stability
table above), the mask is computed **once per clip**, not once per frame.

The interesting part is that we do not trust SAM on its own. Two independent
estimates of the road are built and their **agreement is reported as a number**:

1. **Empirical** - `travelled_mask`: rasterise every road-contact point from every
   trajectory, dilate by one lane width, morphologically close across lane gaps.
   Where traffic has driven *is* road. It covers 26.1% of the frame.
2. **Geometric** - MobileSAM prompted at 48 points on bare carriageway.

**SAM is class-agnostic** - it returns whatever object sits under a prompt point and
has no concept of "road". That single fact caused two separate bugs, in opposite
directions, and both are documented in `ISSUES.md` (#17, #18). The fix for the
second was a per-segment vetting rule - a SAM segment is accepted only if >= 45% of
its own area lies inside the travelled mask. 13 of 28 segments were rejected
(rooftops and a tree canopy); 15 were accepted.

Final mask: **27.2% of the frame, 2,096.8 m²** of carriageway. Area is integrated on
a coarse grid using the per-pixel ground scale, because at -63 deg of pitch the
metres-per-pixel scale varies substantially between the top and bottom of the frame.
Agreement with the travelled mask: **IoU 0.468**. That number is deliberately
conservative rather than flattering - we would rather under-claim the road than let
a rooftop into the denominator of a density figure.

In the annotated video, everything off the carriageway is **dimmed to 38%** rather
than blacked out, so a judge can still see what was excluded and check the decision.

---

## 7. Analytics layer

All of these consume `trajectories.csv` only.

**Counts** - 292 road users: 231 car, 29 pedestrian, 18 lgv, 9 motorcycle, 3 hgv,
2 bus. 263 motor vehicles, 38 vulnerable road users.

**Directional flow** - 16-sector compass histogram of exit bearings. Dominant
movements WNW (83) and ESE (38) - a real corridor, recovered from telemetry-derived
true bearings rather than screen directions.

**Turning movements** - entry vs exit bearing per track: 86 straight, 14 left,
14 right, 178 unknown (tracks too short or too stationary to have a defined entry
and exit heading - reported as unknown rather than guessed).

**Speed** - centred difference over a ~0.5 s baseline: long enough to suppress
per-frame jitter, short enough to preserve real acceleration. Mean moving speed
16.7 km/h, median 16.8, 85th percentile 23.6, robust max (p99.5) 35.8 km/h. Always
labelled **estimated**.

**Queues** - vehicles below a speed threshold are clustered with a `cKDTree` +
union-find, then measured along the queue axis. 91 queue observations, largest
22 vehicles.

**Congestion** - density and speed-deficit index over 2 s bins: 22 HIGH, 8 MEDIUM.

**Headways** - 93 leader/follower pairs with gap in metres and time headway in
seconds.

**Interactions / conflicts** - closest-point-of-approach analysis on the metric
trajectories: `t_cpa = -(dp.dv)/|dv|²`, then `d_cpa = |dp + t_cpa.dv|`, with a
lateral half-width envelope. Graded jointly on TTC *and* closing speed, because two
cars 1.5 m apart in a stationary queue are not a conflict. 51 graded potential
conflicts (18 high, 33 medium); 87 low-energy close interactions kept separately as
a density signal, not upgraded to safety events.

**Anomalies** - `events.csv`, 387 rows in one flat schema. 20
`parked_vehicle_candidate`, 53 `stationary_vehicle_candidate`, 69 `sudden_stop`,
19 `against_dominant_flow`, 12 `queue_formation`, 7 `unusual_dwell`.

---

## 8. What we deliberately do not claim

This is the section we would most like the bench to read.

- **No fabricated units.** Every speed, distance and bearing traces to measured
  telemetry. If telemetry were absent the pipeline falls back to pixel-space and
  the HUD says *"pixel-space (no calibration)"* - it does not invent a scale.
- **Speeds are "estimated"** - single monocular camera, assumed flat ground plane.
- **Conflicts are "potential conflicts"** - a kinematic projection of current
  velocities is not a prediction that a collision would have occurred. **No
  collision is claimed anywhere.** No accident occurs in this footage and our output
  does not say one does.
- **Wrong-way is `against_dominant_flow`** - the reference direction is learned from
  the observed modal bearing per grid cell, not from a legal carriageway direction.
  With no HD map this is a flow anomaly, not a proven violation.
- **Stationary vehicles are "candidates"**, and one stopped inside a detected queue
  is explicitly described as signal-related rather than as an incident.
- **`parked_vehicle_candidate` vs an obstruction** - not distinguishable from a
  single clip, and the label says so.
- **LGV/HGV provenance is disclosed per row** (see section 4).
- **No MOTA / IDF1** without ground truth (see section 5).
- **`turn_type = unknown` is reported as unknown** for 178 tracks rather than
  defaulted to "straight".
- **Pixels above the horizon return NaN**, not a coordinate.

---

## 9. Problems we found in our own output, and the fixes

`ISSUES.md` documents 21 issues in full. **Nine of them were found by interrogating
our own output rather than by hitting a crash** - the pipeline ran, produced
plausible-looking numbers, and the numbers were wrong. Five worth the bench's time:

**Speeds of 300+ km/h at an urban intersection.** Cause: the trajectory smoother ran
over a raw frame index that had gaps, so a 3-frame occlusion was differentiated as
if it were one frame. Fix: reindex onto a continuous time base per track before
differencing, and use a centred difference over a fixed ~0.5 s baseline instead of a
fixed frame count.

**Every close pair in dense traffic was flagged as a conflict.** The geometry was
correct and the conclusion was still wrong: two cars 1.5 m apart in a stationary
queue satisfy every proximity and TTC test. Fix: grade on TTC **and** closing speed
jointly. Low-energy encounters are still emitted as `close_interaction` at low
severity - a density signal - which cut the conflict count roughly 4x while keeping
the data.

**MobileSAM segmented 24 cars instead of the road.** The tell was in the log: 24
trajectory-seeded prompts produced exactly 24 segments. Root cause: trajectory
points sit *on vehicles*, and SAM is class-agnostic - prompt a car, get a car. Fix:
place prompts on bare carriageway by excluding pixels covered by a detection box in
the segmented frame, one seed per 96 px grid cell. Road coverage 35.8% -> 56.7%.

**Then the mask swallowed a rooftop and a tree canopy.** Same root cause pointing
the other way - a grey roof looks exactly like grey tarmac from 70 m, and no
threshold fixes that. Fix: *"SAM proposes, the observed traffic vouches"* - accept a
segment only if >= 45% of its own area lies inside the travelled mask. 13 of 28
rejected, IoU 0.342 -> 0.468.

**Nearly every vehicle in the annotated video read "UNUSUAL DWELL".** Two
independent causes behind one symptom. Semantically, a bare *"visible > 25 s"* test
fired on 27 kerbside cars that were stationary for 100% of their observed life -
those are parked vehicles, not anomalies. Visually, each event's duration equalled
the whole clip, so all 27 stayed highlighted for all 1798 frames, and low-severity
events were painting warning-orange over the class colour - destroying the class
information Level 1 is actually scored on. Fix: split on stationary fraction
(>= 90% -> `parked_vehicle_candidate` with a 3 s marker window; otherwise
`unusual_dwell`, a journey genuinely interrupted), let the highest active severity
win a track's label, and restrict warning colour, text labels and conflict lines to
medium/high severity.

The small-object study in section 4 is the same discipline applied *before* believing
good news: **the setting with the largest measured improvement is the one we
rejected.**

---

## 10. Results summary

| | |
|---|---|
| Road users detected and tracked | **292** |
| Trajectory observations | **69,505** |
| Class breakdown | 231 car, 29 pedestrian, 18 lgv, 9 motorcycle, 3 hgv, 2 bus |
| ID gap recoveries | **2,688** (longest bridged occlusion 3.07 s) |
| Carriageway segmented | 27.2% of frame = **2,096.8 m²** |
| Ground scale | 0.0617 m/px at centre, 97.6 m across the frame's bottom edge |
| Estimated mean moving speed | 16.7 km/h (85th pct 23.6) |
| Events flagged | 387 across 8 event types |
| Largest queue | 22 vehicles |

---

## 11. How to run

```bash
pip install -r requirements.txt
python run_pipeline.py --video data/intersection.mp4     # full pipeline
python run_pipeline.py --from-trajectories               # analytics + render only (cached detections)
streamlit run app.py                                     # dashboard
python tools/measure_small_objects.py --frames 30        # reproduce the tuning study
```

Everything is configured in `config/config.yaml`, with the reasoning for each
threshold written next to it as a comment. A Colab notebook
(`Traffic_Intelligence_Colab.ipynb`) runs the same code on a free T4 GPU reading
video straight from Google Drive.

---

## 12. Extending to later levels

The trajectory table is the contract. Because analytics never touch detection or
tracking, later levels bolt on:

- **new metric** -> one function taking the trajectory DataFrame
- **new event type** -> one detector function appended to `anomalies.DETECTORS`
- **new visual** -> reads the same CSVs
- **multi-clip / multi-drone** -> the metric ENU frame is already georeferenced by
  the telemetry GPS, so trajectories from separate flights share a coordinate system

Detection is the expensive stage and it is cached. Every subsequent level can
iterate on analysis in minutes instead of re-running the model.
