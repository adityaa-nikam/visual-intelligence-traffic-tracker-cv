# ERRORS.md

Concrete failures hit while building this project: the **exact error**, the
**diagnosis**, and the **solution**.

This is the companion to [ISSUES.md](ISSUES.md), and the split is deliberate:

- **ISSUES.md** = the 29 times the code *ran fine and produced a wrong answer.*
  Silent analytical failures, found by interrogating our own output.
- **ERRORS.md** (this file) = the times something *broke loudly* - tracebacks,
  API rejections, environment failures, tooling that lied to us.

The second list is shorter to explain but was most of the wall-clock cost. Where an
error also changed a design decision, the ISSUES.md entry is cross-referenced.

Error text is quoted verbatim where captured. Where it is paraphrased, it says so.

---

## 1. `build_projector()` rejected its own config

```
TypeError: build_projector() missing 1 required positional argument: 'image_h'
```

**Diagnosis.** A standalone validation script called `build_projector(cfg)`, assuming
the function read the frame size out of the config dict. The real signature is
`build_projector(cfg: dict, image_w: int, image_h: int)` - the projector is built per
frame size, because the ground-plane homography depends on the image dimensions, not
on the config.

**Solution.** `build_projector(cfg, 1920, 1080)`. No library change - the caller was
wrong, and the error was correct.

---

## 2. `fillna` was handed a DataFrame instead of a Series

```
TypeError: "value" parameter must be a scalar, dict or Series, but you passed a "DataFrame"
```

Raised inside `apply_track_classes` (`pipeline/trajectory.py`), and only ever on the
`--from-trajectories` path.

**Diagnosis.** The function does
`df.rename(columns={"class": "class_raw"})` and then
`df["class"].fillna(df["class_raw"])`. A *freshly detected* frame has one `class`
column, so this is fine. But a **saved** `trajectories.csv` has already been through
this function once, so it already contains a `class_raw` column. Renaming `class` to
`class_raw` therefore produced **two columns with the same name**, and
`df["class_raw"]` selected both of them - a DataFrame, not a Series.

**Solution.** Drop any columns a previous pass added, before the rename:

```python
df = df.drop(columns=[c for c in ("class_raw", "class_source") if c in df.columns])
```

**Why it mattered.** This is what broke re-rendering a video from saved tracking
data, which is the whole reason the reuse path exists.

---

## 3. `--tag v4` silently switched the video off

No exception. The pipeline printed a one-line note and finished successfully, having
produced **no video at all** - which is how the user found it ("i want the video
atleast i dont see the video int hte output folder").

**Diagnosis.** `--tag` names the **output** files, but the code also used it to
resolve **inputs**: `in_tag = args.tag`, so `in_path("boxes")` looked for
`outputs/boxes_v4.csv`. That file never existed - the boxes were saved as
`outputs/boxes.csv` by the original untagged run. The reuse branch found no boxes,
and its error handling was to set `write_video = False` and carry on quietly.

**Solution.** Fall back to the untagged file when the tagged one is absent:

```python
boxes_path = in_path("boxes")
if not os.path.exists(boxes_path):
    alt = os.path.join(out_dir, "boxes.csv")   # --tag names the OUTPUT
    if os.path.exists(alt):
        boxes_path = alt
```

**Transferable lesson.** "Degrade gracefully" and "fail silently" are the same code
path with different intentions. A missing input that disables a headline deliverable
must be loud.

---

## 4. Re-classification reported that it changed nothing

Not an exception. The run completed and printed:

```
measured size overrode the model on 0 tracks (0.0%)
```

We had measured 25% offline. **The contradiction was the only symptom.**

**Diagnosis.** Re-running classification over a saved `trajectories.csv` re-derives
the previous answer, because that file's `class` column *already holds the resolved
class* - the model's raw per-frame output was moved to `class_raw` on the way out. So
the voting stage was voting on its own previous conclusion and, unsurprisingly,
agreed with itself.

**Solution.** Restore the raw labels before re-classifying:

```python
if "class_raw" in obs.columns:
    obs["class"] = obs["class_raw"]
    obs = obs.drop(columns=["class_raw", ...])
```

**Why it is in this file and not just ISSUES.md.** There was no bug in the logic and
no error to read. The only thing that flagged it was a number that disagreed with a
number we already knew. Had we not measured the 25% first, this would have shipped.

---

## 5. `DataFrame columns must be unique`

```
ValueError: DataFrame columns must be unique for orient='records'.
```

Raised building the LLM evidence brief, in the "notable vehicles" helper.

**Diagnosis.** The helper sorts a table by a column and then selects a fixed
keep-list of interesting columns. For `_top(tracks, "length_m", 4)` the sort column
`length_m` is *already in* the keep-list, so the selection contained it twice, and
`to_json(orient="records")` refuses duplicate names.

**Solution.** De-duplicate while preserving order:

```python
keep = list(dict.fromkeys([c for c in ["track_id", "class", col, ...] if c in df]))
```

**Related, same root cause.** The dashboard's vehicle register merges
`track_summary` with `kinematics`, and both legitimately carry `class`, `n_obs`,
`length_m`, `width_m` and `mean_speed_kph` - they were designed to each stand alone
as an artifact. A plain merge produced `mean_speed_kph_x` / `mean_speed_kph_y`, and
the display list matched neither, so **the speed columns silently vanished from the
table.** Fixed by taking only what the left table does not already have:
`["track_id"] + [c for c in kin.columns if c not in tracks.columns]`.
See ISSUES.md #27.

---

## 6. The video player rendered but showed nothing

No error anywhere. `st.video()` produced a player that sat at `0:00` with a blank
frame and a dead scrubber.

**Diagnosis.** Two bugs stacked:

1. OpenCV's writer produces **mpeg4** (`mp4v`), which no browser can decode.
   Confirmed with `ffprobe`: `annotated_v4.mp4 -> mpeg4,yuv420p` versus
   `annotated_v4_web.mp4 -> h264,yuv420p`. The H.264 transcode existed the whole time.
2. The lookup for it asked `suffixed("annotated_web", ".mp4", tag)`, which builds
   `annotated_web_v4.mp4`. The real filename is `annotated_v4_web.mp4` - **the tag
   comes first and `_web` last.** The lookup missed, and the fallback handed
   `st.video()` the raw mpeg4 file.

There was a third, quieter problem: `_vN` is overloaded. It marks a re-render of one
run (`annotated_v3.mp4`) *and* is used as a run tag (`annotated_v4.mp4`), so the
default run was displaying v4's footage under its own heading.

**Solution.** `playable_video()` returns `(file_to_play, newest_raw_render)`:
prefers the H.264 sibling of the newest render, falls back to the newest H.264 file
in the family, and only then to an unplayable raw. A run that has its own
`summary_*.json` owns its files, which resolves the `_vN` ambiguity. When the newest
render has no transcode, the dashboard **says so and prints the ffmpeg command**
rather than quietly showing older footage.

---

## 7. The LLM gateway rejected assistant prefill

```
Error code: 400 - {'error': {'message': 'Anthropic Claude bad request: This model
does not support assistant message prefill. The conversation must end with a user
message.', 'type': '<nil>'}, 'type': 'error'}
```

**Diagnosis.** To guarantee parseable output, the interpretation call ended the
message list with `{"role": "assistant", "content": "{"}` - prefilling an opening
brace so the model can only continue as JSON. This is a legitimate and reliable
technique against the Anthropic API directly, but the configured gateway rejects
assistant-final conversations outright.

**Solution.** End on the user turn and locate the object in the reply instead.
`_extract_json()` strips a markdown fence if present, then finds the first `{` and
last `}`. Verified against 7 reply shapes - bare, fenced with and without a language
tag, prose before, prose after, nested braces, and a reply containing no JSON at all,
which degrades to a `_parse_failed` flag instead of raising.

**Note for whoever restarts this.** `pipeline.interpret` is cached in `sys.modules`,
so a Streamlit rerun keeps serving the old code. The server must be restarted or the
identical 400 comes back and looks like the fix failed.

---

## 8. A 401 whose error format was not Anthropic's

Testing the API path with a deliberately invalid key returned:

```
Error code: 401 - {'error': {'message': '?????', 'type': 'new_api_error'}}
```

Neither the error `type` nor the request-id format matches the Anthropic API.

**Diagnosis.** This machine has `ANTHROPIC_BASE_URL=https://agentrouter.org` set in
the environment, and the Anthropic SDK honours that variable **automatically**. So
`anthropic.Anthropic(api_key=key)` - the obvious one-line constructor - routes both
the API key and the request body to a third-party proxy, with nothing in the code or
on screen indicating it. The dashboard asks the user to paste a key into a text box,
so this is a credential-disclosure path, and it is silent by construction: if the
proxy works, everything looks fine.

**Solution.** Pin and display the destination. `resolve_endpoint()` returns the base
URL that will actually be used; the sidebar shows it and warns when it is not
`api.anthropic.com`, with a checkbox to force the real endpoint; the CLI prints the
same warning and accepts `--base-url`. The key is session-scoped - never written to
disk, never logged, never committed. See ISSUES.md #28.

---

## 9. Detection was going to take 100 minutes

Not an error - a measurement that made the plan impossible. Detection ran at
**0.26 fps** on CPU at native 1920, so one pass over the clip was ~73-100 minutes,
against a 4-hour budget for 5 levels.

**Diagnosis.** An RTX 2050 is present, but `torch` was the **CPU-only build**
(`2.13.0+cpu`), and only 6.1 GB was free on `C:` - not enough headroom to install
CUDA torch safely mid-hackathon.

**Solution.** Two independent wins instead of a risky one:

1. **ONNX Runtime** rather than PyTorch: `onnxruntime 1.29.0`, model exported with
   `model.export(format="onnx", imgsz=(1088,1920), opset=12)`. Measured **0.41 vs
   0.26 fps for byte-identical output** - 1,812 detections from both backends over
   the same 15 frames. 1.6x faster, zero accuracy trade.
2. **`frame_stride: 3`**, with the justification recorded in the config: at 20 km/h a
   vehicle moves ~6 px between processed frames against a ~40 px box, so ByteTrack's
   IoU association is unaffected, and the 90-frame track buffer becomes 6 s of real
   time instead of 3 s - bridging occlusions *better*, not worse.

**Follow-on gotcha.** The ONNX graph is exported at a **fixed** input shape, so
`imgsz` must be an explicit `[h, w]` list (`[1088, 1920]`) that matches the export
exactly - 1088 being 1080 padded up to a multiple of 32. A scalar `imgsz` silently
mismatches the graph.

---

## 10. Physically impossible acceleration

`max_accel_ms2` came out at **45.25 m/s2** - 4.6 g. No error; the number was simply
absurd.

**Diagnosis.** Acceleration is a *second* derivative. At 0.0617 m/px and 30 fps, a
1-pixel bounding-box wobble differentiates into tens of m/s2. The pipeline was
faithfully reporting tracker jitter as vehicle dynamics.

**Solution.** Report robust statistics as the headline (`p95_accel_ms2`,
`p5_decel_ms2` per track), keep raw extremes for audit only, and **flag rather than
silently clip** anything beyond `PHYSICAL_ACCEL_LIMIT_MS2 = 10.0` via an
`accel_exceeds_physical` column. Fleet extremes are computed over the 266 unflagged
tracks. Result: per-class typical 0.3-2.0 m/s2, fleet -8.46 / +7.97 m/s2.
See ISSUES.md #25.

---

## 11. Every parked vehicle had a NaN average speed

**Diagnosis.** `mean_speed_kph` was averaged over *moving* observations only, which
is undefined for a vehicle that never moved - a correct answer to a badly posed
question.

**Solution.** Report both, because they answer different questions and collapsing
them hides the queueing: `mean_speed_kph` (journey speed, includes stops) and
`mean_moving_speed_kph` (cruise speed), plus `moving_fraction`.

---

## 12. `truck` leaked into the output vocabulary

`truck: 11` appeared in the resolved class counts. `truck` is a COCO class, not one
of ours.

**Diagnosis.** Tracks with too few usable geometry solves fall back to the model's
raw label. For most classes that is the right call, but `truck` is not in the
challenge vocabulary at all, so it escaped as-is.

**Solution.** A dedicated fallback branch for `truck` that splits on the Level 1
footprint p90 against `lgv_hgv_split_m`, tagged
`class_source = "footprint_fallback_insufficient_geometry"` so the weaker evidence is
visible in the output. Truck count is now 0.

---

## 13. Tooling that lied to us

Five environment failures that cost real time and produced no useful error.

**a. Bash heredoc refused a multi-line append.**

```
bash: unexpected EOF while looking for matching `''
```

Appending prose containing apostrophes and backticks to `ISSUES.md` via a heredoc
kept terminating early. **Solution:** write a throwaway Python script with the `Write`
tool, run it, delete it. Quoting rules stop being a problem when the content never
passes through the shell.

**b. `nohup ... &` produced no log file.**

```
tail: cannot open 'outputs/run_v4.log' for reading: No such file or directory
```

**Solution:** use the harness's own background execution rather than shell
backgrounding.

**c. Background logs came back empty.** A trailing `| tail -N` in the command
buffers all output until the process exits, so polling the log showed nothing while
the job ran. **Solution:** `python -u` redirected straight to a file, no pipe, then
poll with `tail` / `stat -c%s`.

**d. `pkill -f streamlit` silently did nothing on Windows.** It reported success,
the process kept running, and the replacement server failed to bind:

```
Port 8503 is not available
```

The old instance kept serving, so **the browser showed a version of the app that
predated the changes** - which looks exactly like "the fix didn't work".
**Solution:** find the real PID and kill it properly:

```bash
netstat -ano | grep ":8503.*LISTENING"   # -> PID
taskkill //PID <pid> //F
```

**e. Windows console encoding broke error printing.**

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 41-45
```

Printing an API error containing non-ASCII characters crashed the *diagnostic*, not
the code under test. **Solution:** `PYTHONIOENCODING=utf-8`, or
`.encode("ascii", "replace")` when printing untrusted error text.

---

## 14. A hypothesis that measurement killed

Not an error, but it belongs here because it *would* have been a wasted hour.

The suggestion was that YOLO11 detects vehicles better from a drone, and it was
worth switching to fix both "everything is a car" and the missing two-wheelers.

**Diagnosis by measurement, not opinion.** On 10 frames spread across the clip:

| Model | called "car" | motorcycles found |
|---|---|---|
| yolov8s | 71% | 76 |
| yolo11s | **76%** | **7** |

YOLO11 was **worse on both** reported problems.

**Conclusion.** "Everything is a car" is a **viewpoint** failure, not an
architecture failure - a top-down vehicle at 70 m has no distinguishing silhouette
for any COCO class head. No amount of model-swapping fixes it, which is why the
solution was to measure the vehicle instead (ISSUES.md #24). We kept yolov8s.

---

## The pattern

Of the 14 entries above, only **five** were exceptions with a stack trace
(1, 2, 5, 13a, 13e). The rest either succeeded while producing the wrong thing
(3, 4, 6, 10, 11, 12) or were environment and tooling failures that reported success
(13b, 13c, 13d).

Three of those - 3, 4 and 6 - are the same shape: **the code completed, printed
nothing alarming, and quietly withheld or replaced a deliverable.** A missing video, a
re-classification that changed nothing, a player showing older footage. Every one was
caught by comparing the output against a number or a file we already knew about, not
by reading an error.

And 13d is the version of that failure which attacks *debugging itself*: a stale
server serving stale code makes a working fix look broken, which is the fastest way
to abandon a correct solution.
