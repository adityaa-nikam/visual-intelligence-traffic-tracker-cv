"""
Generates Traffic_Intelligence_Colab.ipynb.

Kept as a generator script rather than a hand-edited .ipynb so the notebook can
be regenerated after pipeline changes without fighting JSON escaping.
"""

import json

MD = "markdown"
CODE = "code"

CELLS = [
    (MD, """# Drone Traffic Intelligence - Level 1
### Detection -> Tracking -> Trajectories -> Traffic Analytics

Runs the full pipeline on a Colab GPU (T4 does ~20-25 fps vs ~1.2 fps on a laptop CPU),
so you can process the **entire** video rather than a 60-second clip.

**Pipeline stages**

| Stage | What it does |
|---|---|
| 1. Calibration | Reads DJI telemetry embedded in the MP4 (altitude, focal length, gimbal angles) and builds a pixel -> metres ground projection |
| 2. Detection | YOLOv8 finds road users |
| 3. Tracking | ByteTrack assigns stable IDs that survive occlusion |
| 4. Trajectories | Every object becomes a metric path: position, speed (km/h), heading |
| 5. Analytics | Counts, dwell, stationary, queues, congestion, turning movements |
| 6. Interactions | Pairwise time-to-collision -> potential conflicts |
| 7. Anomalies | Event engine writing `events.csv` |

Run the cells top to bottom."""),

    (MD, """## 1. Check the GPU

`Runtime -> Change runtime type -> T4 GPU` if this reports no GPU."""),

    (CODE, """import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)

import torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU')"""),

    (MD, """## 2. Install dependencies

Colab already has torch, OpenCV, pandas, scipy and ffmpeg. We only need Ultralytics
(YOLO + ByteTrack) and `lap` for the tracker's assignment step."""),

    (CODE, """!pip install -q ultralytics lap
import ultralytics; print('ultralytics', ultralytics.__version__)"""),

    (MD, """## 3. Get the pipeline code

**Option A (recommended) - GitHub.** Set `REPO_URL` below. After that, whenever the code
changes locally you just `git push` on your machine and re-run **this one cell** in Colab
to pull it. No re-uploading.

**Option B - zip upload.** Leave `REPO_URL` empty and upload `traffic_ai_code.zip`
(a few tens of KB, produced locally by `python make_code_zip.py`)."""),

    (CODE, """REPO_URL = ""   # e.g. "https://github.com/<you>/traffic-ai.git"

import os, zipfile, subprocess

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    return r.returncode

if REPO_URL:
    if os.path.isdir('traffic-ai/.git'):
        sh('cd traffic-ai && git pull --ff-only')     # already cloned -> just update
    else:
        sh(f'git clone {REPO_URL} traffic-ai')
    os.chdir('/content/traffic-ai')
elif not os.path.exists('pipeline'):
    print('Select traffic_ai_code.zip ...')
    from google.colab import files
    up = files.upload()
    with zipfile.ZipFile(list(up.keys())[0]) as z:
        z.extractall('.')

for d in ('data', 'outputs', 'models'):
    os.makedirs(d, exist_ok=True)

print('\\ncwd     :', os.getcwd())
print('pipeline:', sorted(os.listdir('pipeline')))"""),

    (MD, """## 4. Get the drone video from Google Drive

Two options - use whichever matches how the video is shared.

**Option A - a shared Drive link.** Take the FILE ID out of the URL:
`https://drive.google.com/file/d/`**`1AbCdEf...`**`/view` and paste it below.

**Option B - the file is in your own Drive.** Set `USE_DRIVE_MOUNT = True` and give
the path inside the mounted drive.

The videos are ~5-6.5 GB, so the download/copy is the slowest cell in the notebook."""),

    (CODE, """# ---- configure this cell -------------------------------------------------
USE_DRIVE_MOUNT = False

# Option A: shared-link file id
DRIVE_FILE_ID = ""          # e.g. "1AbCdEfGhIjKlMnOpQrSt"

# Option B: path inside your mounted Drive
DRIVE_PATH = "/content/drive/MyDrive/Intersection_Merged.MP4"

SOURCE_VIDEO = "data/source.MP4"
# --------------------------------------------------------------------------

import os

if os.path.exists(SOURCE_VIDEO) and os.path.getsize(SOURCE_VIDEO) > 10_000_000:
    print('already downloaded:', SOURCE_VIDEO, round(os.path.getsize(SOURCE_VIDEO)/1e9, 2), 'GB')
elif USE_DRIVE_MOUNT:
    from google.colab import drive
    drive.mount('/content/drive')
    assert os.path.exists(DRIVE_PATH), f'not found: {DRIVE_PATH}'
    # Symlink instead of copying: avoids duplicating several GB on disk.
    if os.path.lexists(SOURCE_VIDEO):
        os.remove(SOURCE_VIDEO)
    os.symlink(DRIVE_PATH, SOURCE_VIDEO)
    print('linked', DRIVE_PATH)
else:
    assert DRIVE_FILE_ID, 'Set DRIVE_FILE_ID (or switch USE_DRIVE_MOUNT to True)'
    !pip install -q --upgrade gdown
    !gdown --id $DRIVE_FILE_ID -O $SOURCE_VIDEO

print('size:', round(os.path.getsize(SOURCE_VIDEO)/1e9, 2), 'GB')"""),

    (MD, """## 5. Inspect the video and extract the DJI telemetry

This is the step that makes real-world measurement possible.

DJI aircraft write a per-frame flight log into the MP4 as a `tx3g` **subtitle track**.
Each record carries:

```
[focal_len: 24.00] [latitude: 18.566227] [longitude: 73.771846]
[rel_alt: 70.472 abs_alt: 607.273] [gb_yaw: -125.5 gb_pitch: -63.1 gb_roll: 0.0]
```

Altitude + focal length + gimbal angles are exactly enough to project any pixel onto
the road plane and get **metres**. Without this we could only report pixel velocities,
and quoting km/h would be fabrication."""),

    (CODE, """!ffprobe -v error -select_streams v:0 \\
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration \\
  -of default=noprint_wrappers=1 $SOURCE_VIDEO

print('--- streams (look for a subtitle stream = telemetry) ---')
!ffprobe -v error -show_entries stream=index,codec_type,codec_tag_string -of csv=p=0 $SOURCE_VIDEO

SRT = 'data/telemetry.srt'
!ffmpeg -v error -i $SOURCE_VIDEO -map 0:s:0 -y $SRT

import os
if os.path.exists(SRT) and os.path.getsize(SRT) > 0:
    print('\\n--- telemetry extracted, first record ---')
    print(open(SRT).read(600))
else:
    print('\\nNo telemetry track. Pipeline will fall back to pixel-space analytics')
    print('and will NOT report km/h (by design - we do not fabricate metric values).')"""),

    (MD, """## 6. Prepare the working clip

Downscaling 4K -> 1920 roughly quarters the decode/inference cost. Two things make this safe:

* At ~70 m altitude a car is still ~70 px long at 1920 wide, well within YOLO's range.
* Focal length in **pixels** scales with frame width, so the metric calibration stays
  exact after resizing - no accuracy is lost.

On a T4 you can afford a much longer clip than on CPU. Set `CLIP_DURATION = 0` for the whole video."""),

    (CODE, """CLIP_START    = 120    # seconds into the source video
CLIP_DURATION = 120    # seconds; 0 = process the entire video
TARGET_WIDTH  = 1920

CLIP = 'data/clip.mp4'

dur_arg = '' if CLIP_DURATION == 0 else f'-t {CLIP_DURATION}'
!ffmpeg -v error -ss $CLIP_START -i $SOURCE_VIDEO $dur_arg \\
  -vf scale=$TARGET_WIDTH:-2 -c:v libx264 -preset ultrafast -crf 26 \\
  -an -sn -y $CLIP

import cv2, os
cap = cv2.VideoCapture(CLIP)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
print(f'clip: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} '
      f'@ {fps:.2f}fps  {n} frames  {n/fps:.0f}s  {os.path.getsize(CLIP)/1e6:.0f}MB')
cap.release()"""),

    (MD, """## 7. Point the config at this clip

`config/config.yaml` holds every tunable - thresholds, model size, window lengths.
Nothing analytical is hard-coded in the detection or tracking code.

`clip_start_s` matters: it tells the calibration code where this clip sits inside the
original video so the telemetry lines up with the right frames."""),

    (CODE, """import yaml

cfg = yaml.safe_load(open('config/config.yaml'))
cfg['video']['path']         = CLIP
cfg['video']['clip_start_s'] = float(CLIP_START)
cfg['video']['frame_stride'] = 1        # GPU is fast enough for every frame
cfg['video']['max_frames']   = None
cfg['telemetry']['srt_path'] = SRT
cfg['detector']['device']    = 'auto'   # -> cuda
cfg['detector']['half']      = True     # fp16 on the T4
cfg['detector']['weights']   = 'models/yolov8s.pt'

yaml.safe_dump(cfg, open('config/config.yaml', 'w'), sort_keys=False)
print(yaml.safe_dump(cfg, sort_keys=False))"""),

    (MD, """## 8. Sanity check before the long run

Sixty frames takes seconds on a GPU and validates the entire path: calibration ->
detection -> tracking -> trajectories -> analytics -> events -> video. Far better than
discovering a bug 20 minutes into a full run.

**Read the calibration line carefully.** If altitude and metres-per-pixel look wrong,
every downstream metric is wrong."""),

    (CODE, """!python run_pipeline.py --max-frames 60 --tag smoke

import json
s = json.load(open('outputs/summary_smoke.json'))
print('\\nCALIBRATION:', json.dumps(s['calibration'], indent=1))
print('SPEED      :', s['speed'])
print('COUNTS     :', s['counts'])"""),

    (MD, """### How to tell the calibration is actually right

There is no ground-truth survey here, so we validate by **physical plausibility** -
three independent checks that would all fail if the geometry were wrong:

1. **Hover stability** - `rel_alt` spread across thousands of records should be a few
   centimetres. If so, modelling the flight as one fixed pose is justified.
2. **Ground scale** - at ~70 m with a 24 mm-equivalent lens, expect roughly
   0.05-0.07 m/px and a footprint on the order of 100 m across.
3. **Speeds** - urban intersection traffic should land around 15-30 km/h. Getting
   300 km/h or 2 km/h would immediately expose a broken projection.

Speeds are still reported as *estimates*: flat-ground assumption, box-bottom as the
contact point, and positions relative to the aircraft rather than a survey datum."""),

    (MD, """## 9. Full run

Writes `annotated.mp4`, `trajectories.csv`, `events.csv`, `summary.json`,
`trajectory_map.png` and the per-analytic side tables."""),

    (CODE, """import time
t0 = time.time()
!python run_pipeline.py
print(f'\\ntotal wall clock: {time.time()-t0:.0f}s')"""),

    (MD, """## 10. Results"""),

    (CODE, """import json, pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 90)

s = json.load(open('outputs/summary.json'))
print('=' * 70)
print('ROAD USERS :', s['counts']['total'], s['counts']['by_class'])
print('SPEED      :', s['speed'])
print('CONGESTION :', s['congestion']['levels'], 'peak', s['congestion']['peak_score'])
print('QUEUES     :', s['queues'])
print('TURNING    :', s['turning_movements'])
print('INTERACTION:', s['interactions'])
print('ID STABILTY:', s['id_stability'])
print('EVENTS     :', s['events'])
print('=' * 70)"""),

    (CODE, """ev = pd.read_csv('outputs/events.csv')
print(f'{len(ev)} events\\n')
display(ev[['event_type', 'severity', 'timestamp', 'track_id',
            'secondary_track_id', 'value', 'unit', 'description']].head(30))"""),

    (CODE, """from IPython.display import Image, display
display(Image('outputs/trajectory_map.png', width=900))"""),

    (MD, """### Play the annotated video

`mp4v` (what OpenCV writes) does not play in a browser, so we re-encode to H.264 first."""),

    (CODE, """from IPython.display import HTML
from base64 import b64encode

!ffmpeg -v error -i outputs/annotated.mp4 -vcodec libx264 -crf 28 -preset veryfast -y outputs/annotated_web.mp4

data = b64encode(open('outputs/annotated_web.mp4', 'rb').read()).decode()
HTML(f'<video width=960 controls><source src="data:video/mp4;base64,{data}" type="video/mp4"></video>')"""),

    (MD, """## 11. Re-run analytics without re-running YOLO

This is the core architectural point of the project, and the reason later levels are cheap.

Detection and tracking are the only expensive stages, and they happen **once**.
Everything else reads `trajectories.csv`. So you can retune a threshold, or add an
entirely new analytic for a new level, and get an answer in seconds instead of
re-processing the video.

The cell below re-derives conflicts at several thresholds - no GPU work at all."""),

    (CODE, """import copy, sys, yaml, pandas as pd
sys.path.insert(0, '.')
from pipeline import interactions, analytics

base = yaml.safe_load(open('config/config.yaml'))
obs  = pd.read_csv('outputs/trajectories.csv')
calibrated = obs['sx'].notna().any()
print('observations:', len(obs), '| tracks:', obs.track_id.nunique(), '| metric:', calibrated, '\\n')

for ttc in (1.5, 2.0, 2.5, 3.0, 4.0):
    cfg = copy.deepcopy(base)
    cfg['interactions']['ttc_max_s'] = ttc
    r = interactions.interaction_summary(interactions.find_interactions(obs, cfg, calibrated))
    print(f"ttc_max={ttc}s -> conflicts={r['potential_conflicts']:>3}  "
          f"pairs={r['pairs']:>3}  min_ttc={r['min_ttc_s']}  {r['by_type']}")"""),

    (MD, """## 12. Get the outputs off Colab

Three ways, pick whichever suits you.

**A - Download a zip.** Save it into `hackathon/colab_outputs/` locally.

**B - Write straight to Google Drive.** If you run Google Drive for Desktop, the files
appear on your local disk automatically with no manual download step.

**C - Copy-paste the text summary.** Enough to diagnose almost anything without files."""),

    (CODE, """# --- A: download a zip -----------------------------------------------------
import shutil
from google.colab import files

shutil.make_archive('traffic_ai_outputs', 'zip', 'outputs')
files.download('traffic_ai_outputs.zip')"""),

    (CODE, """# --- B: copy outputs into Google Drive -------------------------------------
SAVE_TO_DRIVE = False
DRIVE_OUT_DIR = '/content/drive/MyDrive/traffic_ai_outputs'

if SAVE_TO_DRIVE:
    import os, shutil
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
    if os.path.isdir(DRIVE_OUT_DIR):
        shutil.rmtree(DRIVE_OUT_DIR)
    shutil.copytree('outputs', DRIVE_OUT_DIR)
    print('copied to', DRIVE_OUT_DIR)
    print(sorted(os.listdir(DRIVE_OUT_DIR)))
else:
    print('set SAVE_TO_DRIVE = True to use this')"""),

    (MD, """### C - one-block text summary

Run this and paste the whole output into chat. It carries every headline number plus
the top events, which is enough to review the run without transferring any files."""),

    (CODE, """import json, pandas as pd

print('===== TRAFFIC INTELLIGENCE RUN SUMMARY =====')
print(json.dumps(json.load(open('outputs/summary.json')), indent=1))

for name in ('events', 'track_summary', 'queues', 'congestion',
             'turning_movements', 'directional_flow', 'interactions'):
    p = f'outputs/{name}.csv'
    try:
        d = pd.read_csv(p)
    except Exception as e:
        print(f'\\n----- {name}: {e}')
        continue
    print(f'\\n----- {name}.csv  ({len(d)} rows) -----')
    print(d.head(15).to_string() if not d.empty else '(empty)')"""),

    (MD, """## Known limitations (stated deliberately)

Being explicit about these is a scoring point, not a weakness - it shows we know what
the data can and cannot support.

| Area | Limitation |
|---|---|
| **LGV vs HGV** | COCO has no LGV/HGV classes. We split the model's `truck` detections by measured ground footprint and record `class_source=size_heuristic_from_calibration`. Never presented as a model output. |
| **Speed** | Estimate. Assumes flat ground and uses the box bottom as the contact point. Reported only when telemetry calibration succeeds - otherwise px/s, never fake km/h. |
| **Stationary vehicles** | Labelled *candidates*. A single clip cannot separate a breakdown from a red light, so we cross-reference detected queues and say so in the event description. |
| **Wrong way** | Relative to the *observed dominant flow* of that part of the road, learned from the data. With no HD map we report a flow anomaly, not a legal violation. |
| **Conflicts** | Constant-velocity projection; it cannot know a driver was already braking. Hence "potential conflict", filtered by closing rate and temporal persistence. |
| **Congestion** | A transparent weighted score for operator triage, not a certified level of service (that needs lane geometry and capacity). |
| **Coordinates** | Metres relative to the aircraft, not a survey datum. GPS is recorded so results could be georeferenced. |"""),
]


def to_cell(kind: str, src: str) -> dict:
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "code":
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": source}
    return {"cell_type": "markdown", "metadata": {}, "source": source}


nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [to_cell(k, s) for k, s in CELLS],
}

with open("Traffic_Intelligence_Colab.ipynb", "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)

print(f"wrote Traffic_Intelligence_Colab.ipynb ({len(nb['cells'])} cells)")
