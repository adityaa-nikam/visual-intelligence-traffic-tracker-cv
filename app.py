"""
Traffic Intelligence dashboard.

Reads only what the pipeline wrote to outputs/ - it runs no models and does no
detection, so it starts instantly and can be re-opened without touching the GPU.

    streamlit run app.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

st.set_page_config(page_title="Drone Traffic Intelligence", page_icon="::", layout="wide")

from pipeline import interpret as interp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AERIAL_CHECKPOINT = os.path.join(BASE_DIR, "models", "best.pt")


def _session_isolation_enabled() -> bool:
    value = os.environ.get("ISOLATE_SESSION_OUTPUTS", "")
    try:
        value = value or st.secrets.get("ISOLATE_SESSION_OUTPUTS", "")
    except Exception:
        # No local Streamlit secrets file is normal during development.
        pass
    return str(value).lower() in {"1", "true", "yes"}

# A shared public deployment must not expose one visitor's uploaded footage or
# results in another visitor's run selector. Local development keeps using the
# repository's outputs/ directory; hosted deployments can isolate files by
# Streamlit session and use ephemeral container storage.
if _session_isolation_enabled():
    _session_id = st.session_state.setdefault("_output_session_id", uuid.uuid4().hex)
    OUT_DIR = os.path.join(tempfile.gettempdir(), "drone_traffic_sessions", _session_id)
else:
    OUT_DIR = os.path.join(BASE_DIR, "outputs")

# Dark operations-console styling; the video and charts carry the colour.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: #171a1f; border: 1px solid #262b33;
        border-radius: 10px; padding: 12px 14px;
      }
      div[data-testid="stMetricLabel"] p {font-size: .72rem; letter-spacing: .06em;
        text-transform: uppercase; color: #8b95a5;}
      .caveat {font-size: .78rem; color: #8b95a5; border-left: 2px solid #3d4450;
        padding-left: .7rem; margin: .3rem 0 1rem 0;}
      .pill {display:inline-block; padding:2px 9px; border-radius:999px;
        font-size:.7rem; font-weight:600; letter-spacing:.04em;}
      h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_COLOR = {
    "car": "#50c878", "lgv": "#c85adc", "hgv": "#963cbe", "truck": "#c85adc",
    "bus": "#f0783c", "motorcycle": "#ffbe3c", "bicycle": "#64dcf0",
    "pedestrian": "#5a82ff",
}
SEV_COLOR = {"high": "#ff4d5a", "medium": "#ffa03c", "low": "#7d8794"}


# --------------------------------------------------------------------------- io
def available_tags() -> list[str]:
    """
    Discover pipeline runs by looking for summary JSONs, newest first.

    Ordered by write time so the dashboard opens on the most recent run rather
    than on whichever tag happens to sort first - a stale default is the fastest
    way to demo the wrong numbers.
    """
    found = []
    for p in glob.glob(os.path.join(OUT_DIR, "summary*.json")):
        base = os.path.basename(p)[len("summary"):-len(".json")]
        found.append((os.path.getmtime(p), base.lstrip("_")))
    return [t for _, t in sorted(found, reverse=True)]


def suffixed(name: str, ext: str, tag: str) -> str:
    return os.path.join(OUT_DIR, f"{name}{'_' + tag if tag else ''}{ext}")


def latest_video(name: str, tag: str) -> str | None:
    """
    Newest render of a video, honouring the `_vN` versioning.

    Renders are never overwritten, so a run directory accumulates annotated.mp4,
    annotated_v2.mp4, ... The dashboard should always show the most recent one.
    """
    base = suffixed(name, ".mp4", tag)
    root = base[:-4]
    cands = [base] + glob.glob(f"{root}_v*.mp4")
    cands = [c for c in cands if os.path.exists(c)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def playable_video(name: str, tag: str) -> tuple[str | None, str | None]:
    """
    Pick the file a browser can actually decode, and report the newest raw render.

    OpenCV's writer produces mp4v/mpeg4, which no browser will play - the player
    renders but sits at 0:00. Every render therefore gets an H.264 sibling named
    `<render>_web.mp4`, and that is what must be handed to st.video().

    The naming is `annotated_<tag>[_vN][_web].mp4`, so the web transcode of run v4
    is `annotated_v4_web.mp4`. Asking suffixed() for "annotated_web" instead builds
    `annotated_web_v4.mp4`, which does not exist - the lookup silently missed, fell
    back to the raw mpeg4 render, and the video appeared blank.

    Returns (file_to_play, newest_raw_render) so the caller can say when the newest
    render has no transcode yet.
    """
    base = suffixed(name, ".mp4", tag)
    root = base[:-4]
    family = [base, f"{root}_web.mp4"]
    family += glob.glob(f"{root}_v[0-9]*.mp4") + glob.glob(f"{root}_v[0-9]*_web.mp4")
    family = [c for c in dict.fromkeys(family) if os.path.exists(c)]

    # `_vN` is overloaded: it marks a re-render of one run (annotated_v3.mp4) AND it
    # is used as a run tag (annotated_v4.mp4). Filename alone cannot tell them apart,
    # so a run that has its own summary JSON owns its files - otherwise the default
    # run would show v4's video, which is the wrong footage under the right heading.
    known = {t for t in available_tags() if t}
    def _own(path: str) -> bool:
        rel = os.path.basename(path)[len(os.path.basename(root)):]
        rel = rel[:-len(".mp4")].removesuffix("_web").lstrip("_")
        return not rel or rel.split("_")[0] not in known
    family = [c for c in family if _own(c)]
    if not family:
        return None, None

    raws = [c for c in family if not c.endswith("_web.mp4")]
    webs = [c for c in family if c.endswith("_web.mp4")]
    newest_raw = max(raws, key=os.path.getmtime) if raws else None

    # The transcode of the newest render is the ideal; otherwise the newest
    # transcode of any render still beats an unplayable file.
    if newest_raw:
        sibling = newest_raw[:-4] + "_web.mp4"
        if os.path.exists(sibling):
            return sibling, newest_raw
    if webs:
        return max(webs, key=os.path.getmtime), newest_raw
    return newest_raw, newest_raw


def ffmpeg_executable() -> str | None:
    """Find system FFmpeg, or use the binary bundled with imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


@st.cache_resource(show_spinner="Checking the aerial detector (first launch may take a minute)...")
def ensure_aerial_model() -> tuple[bool, str]:
    """Download and verify the pinned model once when a deployment lacks it."""
    if os.path.isfile(AERIAL_CHECKPOINT):
        return True, ""
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "tools", "download_aerial_model.py")],
            cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Model setup failed.").strip()
        return False, detail[-3000:]
    return os.path.isfile(AERIAL_CHECKPOINT), result.stdout.strip()


def transcode_for_browser(source: str) -> tuple[str | None, str | None]:
    """Create a browser-compatible H.264 copy without changing the raw render."""
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return None, "FFmpeg is unavailable. Install requirements.txt or system FFmpeg."
    target = source[:-4] + "_web.mp4"
    result = subprocess.run(
        [ffmpeg, "-y", "-i", source, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "28", "-movflags", "+faststart", target],
        cwd=BASE_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(target):
        return None, result.stderr[-1200:] or "FFmpeg did not create the browser video."
    return target, None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ load a run
st.title("Drone Traffic Intelligence")
st.caption("Upload an aerial traffic video to detect, track, and analyze road users.")

model_ready, model_setup_detail = ensure_aerial_model()
if not model_ready:
    st.error("The aerial detector could not be installed, so video analysis is unavailable.")
    if model_setup_detail:
        with st.expander("Model setup details"):
            st.code(model_setup_detail, language="text")

with st.expander("Project overview · tech stack · what I learned"):
    st.markdown(
        "**A computer vision learning project** that turns aerial traffic video into "
        "road-user detections, persistent tracks, trajectories, and traffic summaries. "
        "It is an analytical prototype, not a certified traffic-safety system."
    )
    overview_col, learning_col = st.columns(2, gap="large")
    with overview_col:
        st.markdown("**Tech stack**")
        st.markdown(
            "- **Python** — pipeline and data processing\n"
            "- **PyTorch + Ultralytics YOLOv8n** — aerial road-user detection\n"
            "- **ByteTrack** — tracking detections across frames\n"
            "- **OpenCV + FFmpeg** — video/frame handling and browser-compatible output\n"
            "- **NumPy, Pandas, SciPy** — trajectories and analytics\n"
            "- **Streamlit + Plotly** — interactive dashboard and charts\n"
            "- **DJI SRT telemetry (optional)** — camera/altitude data for ground-plane estimates"
        )
    with learning_col:
        st.markdown("**What I learned**")
        st.markdown(
            "- How object detection differs from multi-object tracking\n"
            "- Why stable IDs, occlusion handling, and frame sampling affect results\n"
            "- How to turn detections into trajectories and derive traffic measures\n"
            "- How camera geometry and telemetry affect pixel-to-metre estimates\n"
            "- How to expose uncertainty and validate a vision pipeline instead of "
            "treating every model output as ground truth"
        )

with st.expander("Upload a video and run analysis", expanded=not bool(available_tags())):
    st.markdown(
        "Choose a video to run the detection and tracking pipeline. DJI telemetry is "
        "optional; without a matching subtitle file, motion is reported in pixel-space. "
        "Processing time depends on video length and available hardware."
    )
    with st.form("upload_run_form"):
        uploaded_video = st.file_uploader(
            "Traffic video", type=["mp4", "mov", "m4v", "avi", "mkv"],
            help="Use an aerial traffic clip. Large or long videos may take a while to process.",
        )
        uploaded_srt = st.file_uploader(
            "Matching DJI telemetry subtitles (optional)", type=["srt"],
            help="Only attach a subtitle file recorded for this exact video. Enables metric estimates.",
        )
        clip_start = st.number_input(
            "Clip start time in the original DJI recording (seconds)",
            min_value=0.0, value=0.0, step=1.0,
            help="Use 0 when the subtitle file starts with the uploaded clip.",
        ) if uploaded_srt else 0.0
        stride = st.select_slider(
            "Process every Nth frame", options=[1, 2, 3, 5], value=3,
            help="Higher values are faster but may miss brief events and small objects.",
        )
        use_segmentation = st.checkbox(
            "Use the single-frame road mask", value=False,
            help="Enable only for a fixed or hovering camera. Moving cameras need frame-by-frame stabilization.",
        )
        # Keep this enabled: controls inside a Streamlit form do not trigger a
        # rerun as files are selected, so a disabled expression here can remain
        # stuck at its initial (no upload) state.
        start_run = st.form_submit_button(
            "Analyze video", type="primary", disabled=not model_ready
        )

    if start_run and uploaded_video is None:
        st.error("Choose a traffic video before starting the analysis.")
    elif start_run:
        ext = Path(uploaded_video.name).suffix.lower()
        if ext not in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
            st.error("Unsupported video format.")
        else:
            tag = "upload_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            uploads_dir = os.path.join(OUT_DIR, "uploads")
            configs_dir = os.path.join(OUT_DIR, "upload_configs")
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(configs_dir, exist_ok=True)
            video_path = os.path.join(uploads_dir, tag + ext)
            with open(video_path, "wb") as video_file:
                video_file.write(uploaded_video.getbuffer())

            config_path = os.path.join(configs_dir, tag + ".yaml")
            with open(os.path.join(BASE_DIR, "config", "config.yaml"), "r", encoding="utf-8") as config_file:
                run_config = yaml.safe_load(config_file)
            run_config["video"]["path"] = video_path
            run_config["video"]["clip_start_s"] = float(clip_start)
            run_config["video"]["frame_stride"] = int(stride)
            run_config["telemetry"]["srt_path"] = None
            run_config["segmentation"]["enabled"] = bool(use_segmentation)
            run_config["output"]["dir"] = OUT_DIR
            # Uploaded footage is aerial: require and explicitly record the
            # VisDrone checkpoint rather than silently falling back to COCO.
            run_config["detector"].update({
                "weights": "models/best.pt",
                "classes": list(range(10)),
                "imgsz": 1280,
                "conf": 0.10,
                "strict_conf": 0.20,
                "small_conf": 0.08,
            })
            with open(config_path, "w", encoding="utf-8") as config_file:
                yaml.safe_dump(run_config, config_file, sort_keys=False)
            srt_path = None
            if uploaded_srt is not None:
                srt_path = os.path.join(uploads_dir, tag + ".srt")
                with open(srt_path, "wb") as srt_file:
                    srt_file.write(uploaded_srt.getbuffer())

            command = [sys.executable, "-u", "run_pipeline.py", "--config", config_path,
                       "--video", video_path, "--stride", str(stride), "--tag", tag]
            if srt_path:
                command.extend(["--srt", srt_path])
            log_box = st.empty()
            progress = st.progress(0, text="Starting analysis…")
            log_lines = []
            try:
                process = subprocess.Popen(
                    command, cwd=BASE_DIR, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    bufsize=1,
                )
                for line in process.stdout:
                    line = line.rstrip()
                    if not line:
                        continue
                    log_lines.append(line)
                    log_lines = log_lines[-12:]
                    frame_match = re.search(r"(\d+) frames\s+([\d.]+) fps", line)
                    if frame_match:
                        progress.progress(35, text=f"Detection and tracking: {frame_match.group(1)} frames")
                    elif "[3/6]" in line or "[3b/6]" in line:
                        progress.progress(55, text="Building trajectories and road mask…")
                    elif "[4/6]" in line:
                        progress.progress(70, text="Calculating traffic analytics…")
                    elif "[5/6]" in line:
                        progress.progress(82, text="Rendering trajectory map…")
                    elif "[6/6]" in line:
                        progress.progress(90, text="Rendering annotated video…")
                    log_box.code("\n".join(log_lines), language="text")
                return_code = process.wait()
            except Exception as exc:
                st.error(f"Could not start analysis: {exc}")
                return_code = -1

            if return_code == 0:
                progress.progress(100, text="Analysis complete")
                video_out = os.path.join(OUT_DIR, f"annotated_{tag}.mp4")
                if os.path.exists(video_out):
                    _, transcode_error = transcode_for_browser(video_out)
                    if transcode_error:
                        st.warning("Analysis finished, but browser video conversion failed. "
                                   "The CSV and summary outputs are available. " + transcode_error)
                st.success("Analysis complete. Opening the new run…")
                st.cache_data.clear()
                st.session_state["selected_run_tag"] = tag
                st.rerun()
            else:
                progress.empty()
                st.error("Analysis failed. The pipeline output above should show the cause.")

tags = available_tags()
if not tags:
    st.info("Upload a video above to create your first analysis run.")
    st.stop()

with st.sidebar:
    st.markdown("### Run")
    default_tag = st.session_state.pop("selected_run_tag", tags[0])
    selected_index = tags.index(default_tag) if default_tag in tags else 0
    tag = st.selectbox("pipeline output", tags, index=selected_index,
                       format_func=lambda t: t or "(default)")

summary = load_json(suffixed("summary", ".json", tag))
obs = load_csv(suffixed("trajectories", ".csv", tag))
events = load_csv(suffixed("events", ".csv", tag))
tracks = load_csv(suffixed("track_summary", ".csv", tag))
boxes = load_csv(suffixed("boxes", ".csv", tag))
queues = load_csv(suffixed("queues", ".csv", tag))
congestion = load_csv(suffixed("congestion", ".csv", tag))
active = load_csv(suffixed("active_tracks", ".csv", tag))
turns = load_csv(suffixed("turning_movements", ".csv", tag))
flow = load_csv(suffixed("directional_flow", ".csv", tag))
inter = load_csv(suffixed("interactions", ".csv", tag))
headways = load_csv(suffixed("headways", ".csv", tag))
kin = load_csv(suffixed("kinematics", ".csv", tag))

calib = summary.get("calibration", {})
calibrated = bool(calib.get("usable"))
if not calibrated:
    # Older runs may have pixel-space pairs graded by the previous pseudo-metric
    # logic. Keep their CSVs intact, but don't present those rows as safety events.
    unsupported_events = {
        "potential_conflict", "close_interaction", "near_miss_proximity"
    }
    if not events.empty and "event_type" in events:
        events = events[~events["event_type"].isin(unsupported_events)].copy()
counts = summary.get("counts", {})
speed = summary.get("speed", {})

with st.sidebar:
    # The API key is intentionally not exposed in the UI. AI interpretation is
    # optional and only available when configured by the operator in the environment.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    llm_model = interp.MODEL
    endpoint = interp.resolve_endpoint()

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Calibration")
    if calibrated:
        st.success("Metric - DJI telemetry")
        st.metric("Altitude", f"{calib.get('altitude_m')} m")
        st.metric("Ground scale", f"{calib.get('metres_per_px_at_centre')} m/px")
        st.caption(
            f"gimbal pitch {calib.get('gimbal_pitch_deg')}deg, "
            f"yaw {calib.get('gimbal_yaw_deg')}deg\n\n"
            f"scene width ~{calib.get('bottom_edge_ground_width_m')} m "
            f"from {calib.get('records')} telemetry records"
        )
        hs = calib.get("hover_stability", {})
        if hs:
            st.caption(
                "**Hover stability** (justifies the single-pose model)\n\n"
                f"altitude spread {hs.get('rel_alt_p5_p95_spread', 0):.3f} m, "
                f"pitch {hs.get('pitch_deg_p5_p95_spread', 0):.2f}deg, "
                f"yaw {hs.get('yaw_deg_p5_p95_spread', 0):.2f}deg"
            )
        if calib.get("gps"):
            st.caption(f"GPS {calib['gps'][0]}, {calib['gps'][1]}")
    else:
        st.warning("No metric calibration")
        st.caption(
            f"{calib.get('reason', 'telemetry unavailable')}\n\n"
            "Analytics fall back to pixel space. **km/h is deliberately not reported** "
            "rather than fabricated."
        )

    st.markdown("### Model")
    det = summary.get("detector", {})
    st.caption(
        f"**{os.path.basename(str(det.get('weights', '?')))}** + ByteTrack\n\n"
        f"imgsz {det.get('imgsz')} - conf {det.get('conf')} - device {det.get('device')}"
    )
    v = summary.get("video", {})
    st.caption(
        f"{v.get('width')}x{v.get('height')} @ {round(float(v.get('fps', 0)), 1)} fps\n\n"
        f"{v.get('frames_processed')} frames at stride {v.get('stride')} "
        f"({v.get('processing_fps')} fps processing)"
    )

    seg = summary.get("road_segmentation", {}) or {}
    if seg.get("enabled"):
        st.markdown("### Road segmentation")
        area_frac = seg.get("road_area_frac", seg.get("travelled_area_frac"))
        st.metric("Carriageway", f"{area_frac:.1%} of frame" if area_frac else "-")
        if seg.get("road_area_m2"):
            st.caption(f"{seg['road_area_m2']:,.0f} m2 of road (via the ground projection)")
        st.caption(f"source: `{seg.get('source')}`")
        agree = seg.get("agreement") or {}
        if agree:
            st.caption(
                "**Two independent estimates, cross-checked**\n\n"
                f"MobileSAM vs observed traffic: IoU {agree.get('iou')}, "
                f"SAM covers {agree.get('sam_covers_travelled_frac', 0):.0%} of the "
                "area traffic demonstrably used."
            )
        vet = seg.get("segment_vetting") or {}
        if vet:
            st.caption(
                f"SAM proposed, traffic vouched: **{vet.get('accepted')} accepted, "
                f"{vet.get('rejected')} rejected**. SAM is class-agnostic, so "
                "unvetted it contributed a rooftop and a tree canopy."
            )
        filt = seg.get("filter") or {}
        if filt.get("applied"):
            st.caption(
                f"off-road filter dropped {filt.get('observations_dropped')} of "
                f"{filt.get('observations_before')} observations"
            )
        else:
            st.caption(
                "Off-road detections are dimmed in the video but **kept** in the "
                "analytics - a detection on a verge may be a real road user."
            )

# --------------------------------------------------------------------- header
run_meta = summary.get("run", {})
if run_meta:
    st.caption(f"Run `{run_meta.get('tag', tag or 'default')}` · "
               f"created {run_meta.get('created_utc', 'unknown')} · "
               f"effective config SHA-256 `"
               f"{run_meta.get('effective_config_sha256', run_meta.get('config_sha256', 'unavailable'))[:12]}`")
elif tag:
    st.caption(f"Saved run `{tag}` · legacy run metadata unavailable")
st.caption(
    "Trajectory-first traffic analytics - every insight below is derived from tracked "
    "road-user trajectories, not from isolated frame detections."
)

k = st.columns(6)
k[0].metric("Total road users", counts.get("total", 0))
k[1].metric("Motor vehicles", counts.get("motor_vehicles", 0))
k[2].metric("Vulnerable users", counts.get("vulnerable", 0),
            help="Pedestrians, cyclists and motorcyclists")
peak = int(active["active_tracks"].max()) if not active.empty else 0
k[3].metric("Peak simultaneous", peak)
if calibrated and speed.get("mean_moving_speed") is not None:
    k[4].metric("Mean moving speed", f"{speed['mean_moving_speed']} km/h",
                help="Estimate. " + str(speed.get("basis", "")))
else:
    k[4].metric("Mean speed", f"{speed.get('mean_moving_speed', '-')} px/s",
                help="No metric calibration, so km/h is not reported.")
if calibrated:
    n_conf = int(summary.get("interactions", {}).get("potential_conflicts") or 0)
    k[5].metric("Potential conflicts", n_conf)
else:
    k[5].metric("Potential conflicts", "Unavailable",
                help="Metric calibration is required for physical conflict grading.")

cong_levels = summary.get("congestion", {}).get("levels", {})
if cong_levels:
    dominant = max(cong_levels, key=cong_levels.get)
    col = {"LOW": "#50c878", "MEDIUM": "#ffa03c", "HIGH": "#ff4d5a"}.get(dominant, "#7d8794")
    st.markdown(
        f"<span class='pill' style='background:{col}22;color:{col};border:1px solid {col}55'>"
        f"CONGESTION {dominant}</span>"
        f"<span class='caveat' style='display:inline;border:none;margin-left:.8rem'>"
        f"peak score {summary.get('congestion', {}).get('peak_score')} - analytical estimate for "
        f"operator triage, not a certified level of service</span>",
        unsafe_allow_html=True,
    )

st.divider()

# ------------------------------------------------------- video + traffic summary
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.subheader("Annotated aerial view")
    legacy_conflicts = summary.get("interactions", {}).get("potential_conflicts")
    if not calibrated and legacy_conflicts:
        st.warning("This is a legacy uncalibrated run. Its saved video may contain pixel-space "
                   "conflict labels that are no longer considered valid; reprocess the clip "
                   "to generate a clean render.")
    playable, newest_raw = playable_video("annotated", tag)
    browser_copy = newest_raw[:-4] + "_web.mp4" if newest_raw else None
    if newest_raw and not os.path.exists(browser_copy):
        st.info("The annotated video is ready. It needs a browser-compatible copy before it can play here.")
        if ffmpeg_executable():
            if st.button("Prepare video for playback", key=f"transcode_{tag}", type="primary"):
                with st.spinner("Converting the rendered video for browser playback…"):
                    converted, error = transcode_for_browser(newest_raw)
                if error:
                    st.error(f"Video conversion failed: {error}")
                else:
                    st.success("Video is ready to play.")
                    st.rerun()
        else:
            st.warning("Video conversion support is missing. Install the project requirements, then refresh this page.")
    elif playable and os.path.exists(playable):
        st.video(playable)
        st.markdown(
            "<div class='caveat'>Bounding box, class, track ID, live speed and trajectory "
            "trail per road user. The trail is drawn from first detection to exit, so ID "
            "continuity is visible end to end. Off-carriageway area is dimmed. Flagged "
            "objects switch to red/orange; potential conflicts are drawn as a line "
            "between the two interacting users.</div>",
            unsafe_allow_html=True,
        )
        mb = os.path.getsize(playable) / 1e6
        st.caption(f"showing `{os.path.basename(playable)}` ({mb:.0f} MB)")
    else:
        st.info("No annotated video for this run.")

    mask_png = suffixed("road_mask", ".png", tag)
    if os.path.exists(mask_png):
        with st.expander("Road segmentation - the drivable-area mask", expanded=False):
            st.image(mask_png, use_container_width=True)
            st.markdown(
                "<div class='caveat'>Computed <b>once</b> per clip, because the aircraft "
                "hovers. Two independent estimates are combined: MobileSAM segments the "
                "frame, and every observed trajectory point is rasterised to show where "
                "traffic demonstrably drove. SAM proposes, the observed traffic vouches - "
                "a segment is only accepted if it overlaps the travelled area, which is "
                "what stops a grey rooftop being called road.</div>",
                unsafe_allow_html=True,
            )

with right:
    st.subheader("Traffic composition")
    by_class = counts.get("by_class", {})
    if by_class:
        cdf = pd.DataFrame({"class": list(by_class), "road users": list(by_class.values())})
        fig = px.bar(
            cdf.sort_values("road users"), x="road users", y="class", orientation="h",
            color="class", color_discrete_map=CLASS_COLOR, text="road users",
        )
        fig.update_layout(
            showlegend=False, height=260, margin=dict(l=0, r=10, t=6, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title=None, yaxis_title=None,
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

        # How the classes were actually decided. Read from the run's own report so
        # this text can never drift from what the pipeline did.
        creport = summary.get("classification", {})
        if creport and calibrated:
            over = creport.get("size_overrode_model", 0)
            pct = creport.get("size_overrode_model_pct", 0)
            flow_txt = "; ".join(creport.get("override_flow", [])[:4])
            st.markdown(
                f"<div class='caveat'><b>How these classes were decided:</b> the aerial detector "
                f"provides the initial class; calibrated size may refine it when the geometry is "
                f"usable. Size overrode the model on <b>{over} of {creport.get('tracks', 0)} "
                f"road users ({pct}%)</b> - {flow_txt}. {creport.get('caveat', '')} Each vehicle's deciding signal is "
                f"recorded per-row in <code>class_source</code>.</div>",
                unsafe_allow_html=True,
            )
        elif creport and not calibrated:
            st.markdown(
                "<div class='caveat'><b>Class labels:</b> shown from the aerial detector. "
                "Without metric calibration, size-based class refinement is not applied. "
                "Review confidence and raw labels in the downloadable boxes CSV.</div>",
                unsafe_allow_html=True,
            )
        elif calibrated and not tracks.empty and "class_source" in tracks:
            n_heur = int((tracks["class_source"] == "size_heuristic_from_calibration").sum())
            if n_heur:
                st.markdown(
                    f"<div class='caveat'><b>LGV / HGV caveat:</b> the detector has no LGV or HGV "
                    f"class. {n_heur} <code>truck</code> detections were split by measured ground "
                    f"footprint (threshold 7 m) using the telemetry calibration. This is a size "
                    f"heuristic, recorded in the CSV as "
                    f"<code>class_source=size_heuristic_from_calibration</code> - not a model output."
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.subheader("Identity stability")
    stab = summary.get("id_stability", {})
    if stab:
        c = st.columns(2)
        c[0].metric("Mean track life", f"{stab.get('mean_track_duration_s', 0)} s")
        c[1].metric("Longest track", f"{stab.get('max_track_duration_s', 0)} s")
        c = st.columns(2)
        c[0].metric("Occlusion recoveries", stab.get("gap_recoveries", 0),
                    help="Times a track was re-associated after missed detections instead "
                         "of being reborn as a new ID.")
        c[1].metric("Longest bridged gap", f"{stab.get('longest_bridged_gap_s', 0)} s")
        st.markdown(
            "<div class='caveat'>Occlusion recoveries are the measurable evidence for stable "
            "identity: ByteTrack's 90-frame buffer re-linked tracks across missed detections "
            "rather than issuing a new ID. Without hand-labelled ground truth this is the "
            "honest metric - we do not quote a MOTA/IDF1 score we cannot compute.</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ------------------------------------------------------------------ trajectories
st.subheader("Trajectory map")
if not obs.empty:
    metric_mode = calibrated and "sx" in obs and obs["sx"].notna().any()
    xcol, ycol = ("sx", "sy") if metric_mode else ("x", "y")
    d = obs.dropna(subset=[xcol, ycol])

    cc = st.columns([1, 1, 2])
    max_tracks = cc[0].slider("tracks to draw", 10, 400,
                              min(120, int(d["track_id"].nunique())), step=10)
    sel_classes = cc[1].multiselect("classes", sorted(d["class"].dropna().unique()),
                                    default=sorted(d["class"].dropna().unique()))
    d = d[d["class"].isin(sel_classes)]
    keep = d["track_id"].drop_duplicates().head(max_tracks)
    d = d[d["track_id"].isin(keep)]

    fig = go.Figure()
    for tid, g in d.groupby("track_id"):
        cls = g["class"].iloc[0]
        fig.add_trace(go.Scattergl(
            x=g[xcol], y=g[ycol], mode="lines",
            line=dict(width=1.4, color=CLASS_COLOR.get(cls, "#9aa4b2")),
            name=str(cls), legendgroup=str(cls), showlegend=False,
            hovertemplate=f"{cls} #{int(tid)}<br>%{{x:.1f}}, %{{y:.1f}}<extra></extra>",
        ))
    unit = "metres east of aircraft" if metric_mode else "pixels (image x)"
    unit_y = "metres north of aircraft" if metric_mode else "pixels (image y)"
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12151a",
        xaxis=dict(title=unit, gridcolor="#1f242c", zeroline=False),
        yaxis=dict(title=unit_y, gridcolor="#1f242c", zeroline=False,
                   scaleanchor="x", scaleratio=1,
                   autorange=True if metric_mode else "reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    frame_note = (
        "common metric ground frame - the single coordinate system a drone provides "
        "and fixed cameras cannot"
        if metric_mode else "image-space frame (no calibration available)"
    )
    st.markdown(
        f"<div class='caveat'>{len(keep)} trajectories in a {frame_note}. "
        f"Axes are equal-scaled, so shapes are geometrically faithful.</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("No trajectories for this run.")

st.divider()

# ------------------------------------------------------------- vehicle register
st.subheader("Vehicle register")
st.caption(
    "Every road user the pipeline tracked, with the measurement that classified it. "
    "One row per identity, not per detection."
)

if tracks.empty:
    st.info("No track summary available for this run.")
else:
    # One row per road user: lifetime/path facts from track_summary, speed and
    # acceleration from the kinematics export. Merged here rather than in the
    # pipeline so each stays a single-purpose artifact.
    reg = tracks.copy()
    if not kin.empty:
        # Take only what track_summary does not already carry, so the merge cannot
        # produce _x/_y suffixed duplicates of mean_speed_kph and friends.
        kin_cols = ["track_id"] + [c for c in kin.columns if c not in tracks.columns]
        reg = reg.merge(kin[kin_cols], on="track_id", how="left")

    f1, f2, f3 = st.columns([2, 2, 1], gap="medium")
    with f1:
        pick_class = st.multiselect(
            "vehicle class", sorted(reg["class"].dropna().unique()), default=[],
            placeholder="all classes",
        )
    with f2:
        search = st.text_input("find vehicle by ID", placeholder="e.g. 42")
    with f3:
        only_moved = st.checkbox("moved only", value=False,
                                 help="hide vehicles that never exceeded the stationary threshold")

    view = reg
    if pick_class:
        view = view[view["class"].isin(pick_class)]
    if search.strip():
        view = view[view["track_id"].astype(str).str.contains(search.strip())]
    if only_moved and "moving_fraction" in view:
        view = view[view["moving_fraction"].fillna(0) > 0.05]

    cols = [c for c in [
        "track_id", "class", "class_source", "length_m", "width_m",
        "first_t", "last_t", "visible_s", "n_obs",
        "mean_speed_kph", "mean_moving_speed_kph", "p85_speed_kph", "max_speed_kph",
        "p95_accel_ms2", "p5_decel_ms2", "path_len_m", "stationary_s", "turn_type",
    ] if c in view.columns]
    st.dataframe(view[cols], use_container_width=True, hide_index=True, height=340)
    st.caption(f"{len(view)} of {len(reg)} road users shown")
    st.download_button("Download vehicle register CSV", view.to_csv(index=False),
                       file_name="vehicle_register.csv", mime="text/csv")
    if not boxes.empty:
        st.download_button("Download detection boxes CSV", boxes.to_csv(index=False),
                           file_name=f"boxes_{tag or 'default'}.csv", mime="text/csv")

    # ------------------------------------------------------ single vehicle drill-down
    st.markdown("#### Individual vehicle")
    ids = view["track_id"].tolist() or reg["track_id"].tolist()
    sel = st.selectbox("vehicle ID", ids, format_func=lambda i: f"#{int(i)}")
    row = reg[reg["track_id"] == sel].iloc[0]
    trace = obs[obs["track_id"] == sel].sort_values("frame") if not obs.empty else pd.DataFrame()

    colr = CLASS_COLOR.get(str(row.get("class")), "#7d8794")
    st.markdown(
        f"<span class='pill' style='background:{colr}22;color:{colr}'>"
        f"#{int(sel)} &middot; {str(row.get('class', '?')).upper()}</span>",
        unsafe_allow_html=True,
    )

    m = st.columns(5, gap="small")
    def _fmt(v, unit="", nd=1):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "n/a"
        return f"{f:.{nd}f}{unit}" if f == f else "n/a"

    m[0].metric("Measured size",
                f"{_fmt(row.get('length_m'), 'm', 2)} x {_fmt(row.get('width_m'), 'm', 2)}")
    m[1].metric("Journey speed", _fmt(row.get("mean_speed_kph"), " km/h"))
    m[2].metric("Cruise speed", _fmt(row.get("mean_moving_speed_kph"), " km/h"))
    m[3].metric("Peak accel / decel",
                f"{_fmt(row.get('p95_accel_ms2'), '', 2)} / {_fmt(row.get('p5_decel_ms2'), '', 2)}")
    m[4].metric("Tracked for", _fmt(row.get("visible_s"), " s"))

    st.markdown(
        f"<div class='caveat'>Classified as <b>{row.get('class', '?')}</b> because "
        f"<code>class_source = {row.get('class_source', 'model')}</code>. "
        "Sizes are consistent relative measurements from the calibrated ground plane, "
        "biased upwards by vehicle height - not catalogue dimensions.</div>",
        unsafe_allow_html=True,
    )

    if not trace.empty:
        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.markdown("**Speed and acceleration over time**")
            speed_col = "speed_kph" if calibrated and trace["speed_kph"].notna().any() else "speed_px_s"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trace["timestamp"], y=trace[speed_col], mode="lines",
                line=dict(color=colr, width=2), name="speed",
            ))
            if calibrated and "accel_kph_s" in trace and trace["accel_kph_s"].notna().any():
                fig.add_trace(go.Scatter(
                    x=trace["timestamp"], y=trace["accel_kph_s"] / 3.6, mode="lines",
                    line=dict(color="#64dcf0", width=1.2), name="accel (m/s2)", yaxis="y2",
                ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12151a",
                xaxis=dict(title="time (s)", gridcolor="#1f242c"),
                yaxis=dict(title="km/h" if speed_col == "speed_kph" else "px/s",
                           gridcolor="#1f242c"),
                yaxis2=dict(title="m/s2", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=1.14, x=0, title=None),
            )
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.markdown("**Path travelled**")
            use_m = calibrated and trace["sx"].notna().any()
            xs, ys = (trace["sx"], trace["sy"]) if use_m else (trace["x"], trace["y"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color=colr, width=2), marker=dict(size=3),
                hovertemplate="t=%{customdata:.1f}s<extra></extra>",
                customdata=trace["timestamp"],
            ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12151a",
                xaxis=dict(title="east (m)" if use_m else "x (px)", gridcolor="#1f242c"),
                yaxis=dict(title="north (m)" if use_m else "y (px)", gridcolor="#1f242c",
                           autorange=None if use_m else "reversed",
                           scaleanchor="x", scaleratio=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    # A vehicle can appear as either party in a conflict, so match both id columns.
    if not events.empty and "track_id" in events:
        mine = events["track_id"] == sel
        if "secondary_track_id" in events:
            mine = mine | (events["secondary_track_id"] == sel)
        own = events[mine]
    else:
        own = pd.DataFrame()
    if not own.empty:
        st.markdown(f"**Events involving #{int(sel)}**")
        ecols = [c for c in ["event_type", "severity", "timestamp", "duration",
                             "value", "unit", "description"] if c in own.columns]
        st.dataframe(own[ecols].sort_values("timestamp"), use_container_width=True,
                     hide_index=True, height=160)
    else:
        st.caption("No events recorded for this vehicle.")

st.divider()

# --------------------------------------------------------- congestion + demand
c1, c2 = st.columns(2, gap="large")

with c1:
    st.subheader("Congestion over time")
    if not congestion.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=congestion["t"], y=congestion["score"],
            marker_color=[{"LOW": "#50c878", "MEDIUM": "#ffa03c", "HIGH": "#ff4d5a"}
                          .get(l, "#7d8794") for l in congestion["level"]],
            hovertemplate="t=%{x}s<br>score %{y:.2f}<extra></extra>", name="score",
        ))
        if not active.empty:
            fig.add_trace(go.Scatter(
                x=active["t"], y=active["active_tracks"], mode="lines",
                line=dict(color="#64dcf0", width=1.5), name="active tracks", yaxis="y2",
            ))
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12151a",
            xaxis=dict(title="time (s)", gridcolor="#1f242c"),
            yaxis=dict(title="congestion score", gridcolor="#1f242c", range=[0, 1]),
            yaxis2=dict(title="active tracks", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<div class='caveat'>{summary.get('congestion', {}).get('method', '')}</div>",
            unsafe_allow_html=True,
        )

with c2:
    st.subheader("Turning movement demand")
    if not turns.empty:
        fig = px.bar(turns, x="turn_type", y="road_users", color="class",
                     color_discrete_map=CLASS_COLOR, barmode="stack")
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#12151a",
            xaxis=dict(title=None, gridcolor="#1f242c"),
            yaxis=dict(title="road users", gridcolor="#1f242c"),
            legend=dict(orientation="h", y=1.14, x=0, title=None),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "<div class='caveat'>Derived from entry vs exit heading of each trajectory. "
            "Tracks too short to judge reliably are excluded rather than forced into a bucket."
            "</div>",
            unsafe_allow_html=True,
        )
    elif not flow.empty:
        st.dataframe(flow, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough sustained trajectories to classify turning movements.")

st.divider()

# ---------------------------------------------------------------- events tabs
st.subheader("Detected events")
t1, t2, t3, t4, t5 = st.tabs(
    ["Event log", "Interactions & conflicts", "Queues", "Directional flow", "All tracks"]
)

with t1:
    if not events.empty:
        sev = st.multiselect("severity", ["high", "medium", "low"],
                             default=["high", "medium", "low"], key="sev")
        types = st.multiselect("event type", sorted(events["event_type"].unique()),
                               default=sorted(events["event_type"].unique()), key="et")
        e = events[events["severity"].isin(sev) & events["event_type"].isin(types)]
        st.caption(f"{len(e)} of {len(events)} events")
        for _, r in e.head(60).iterrows():
            col = SEV_COLOR.get(str(r["severity"]), "#7d8794")
            tid = f"#{int(r['track_id'])}" if pd.notna(r["track_id"]) else ""
            ts = f"{float(r['timestamp']):.1f}s" if pd.notna(r["timestamp"]) else "-"
            st.markdown(
                f"<div style='border-left:3px solid {col};background:#15181d;padding:.55rem .8rem;"
                f"margin-bottom:.4rem;border-radius:0 7px 7px 0'>"
                f"<span class='pill' style='background:{col}22;color:{col}'>"
                f"{str(r['severity']).upper()}</span>&nbsp;"
                f"<b>{str(r['event_type']).replace('_', ' ')}</b> {tid} "
                f"<span style='color:#8b95a5'>at {ts}</span><br>"
                f"<span style='font-size:.85rem;color:#c3cad4'>{r['description']}</span></div>",
                unsafe_allow_html=True,
            )
        with st.expander("raw events table"):
            st.dataframe(e, use_container_width=True, hide_index=True)
    else:
        st.info("No events detected.")

with t2:
    isum = summary.get("interactions", {})
    if not calibrated:
        st.info("Conflict analysis is unavailable without metric calibration. "
                "Pixel-space proximity is not presented as a safety result.")
    elif isum.get("pairs"):
        c = st.columns(4)
        c[0].metric("Interacting pairs", isum.get("pairs", 0))
        c[1].metric("Potential conflicts", isum.get("potential_conflicts", 0))
        c[2].metric("Involving vulnerable users", isum.get("vulnerable_involved", 0))
        c[3].metric("Lowest TTC", f"{isum.get('min_ttc_s', '-')} s")
        st.markdown(
            "<div class='caveat'><b>Method:</b> for every nearby pair we solve for the closest "
            "point of approach under constant velocity. A pair is flagged only if it is genuinely "
            "closing (>=6 km/h along the line of centres), the projected miss distance falls inside "
            "the two users' combined physical envelope, and the trigger <b>persists over multiple "
            "frames</b> - the filter that separates a real conflict from tracker jitter. "
            "A constant-velocity model cannot know a driver was already braking, so these are "
            "conflict <b>indicators</b> for review, not collisions.</div>",
            unsafe_allow_html=True,
        )
        if not inter.empty:
            cols = [c for c in ["track_a", "class_a", "track_b", "class_b", "t", "ttc_s",
                                "min_distance", "d_cpa", "closing_speed", "frames_triggered",
                                "interaction_type", "severity", "unit"] if c in inter.columns]
            st.dataframe(inter[cols], use_container_width=True, hide_index=True)
        if not headways.empty:
            st.markdown("**Car-following headways** (gap / follower speed)")
            st.dataframe(headways, use_container_width=True, hide_index=True)
    else:
        st.info("No qualifying interactions detected.")

with t3:
    if not queues.empty:
        unit = queues["length_unit"].iloc[0]
        c = st.columns(3)
        c[0].metric("Queue detections", len(queues))
        c[1].metric("Largest queue", f"{int(queues['vehicles'].max())} vehicles")
        c[2].metric("Longest queue", f"{queues['length'].max():.0f} {unit}")
        st.dataframe(queues, use_container_width=True, hide_index=True)
        st.markdown(
            f"<div class='caveat'>Single-linkage spatial clustering of slow/stopped vehicles per "
            f"time bin. Length is the cluster's extent along its own principal axis, reported in "
            f"<b>{unit}</b>{' from the telemetry calibration' if unit == 'm' else ' - pixels, because no calibration was available'}. "
            f"Queue membership is also used to downgrade stationary vehicles that are merely "
            f"waiting at a signal.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No queues met the clustering threshold.")

with t4:
    if not flow.empty:
        fig = px.bar_polar(flow, r="road_users", theta="sector",
                           color="road_users", color_continuous_scale="Teal")
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Frame of reference: {flow['frame_of_reference'].iloc[0]}")
        if calibrated:
            st.markdown(
                "<div class='caveat'>These are <b>true compass bearings</b>, recovered from the "
                "gimbal yaw in the telemetry. Without that we would only be able to report "
                "image-space directions - we never invent geographic directions.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No directional flow available.")

with t5:
    if not tracks.empty:
        show = [c for c in ["track_id", "class", "class_source", "first_t", "last_t", "visible_s",
                            "stationary_s", "longest_stop_s", "mean_speed_kph", "max_speed_kph",
                            "path_len_m", "turn_type", "footprint_p90_m", "n_obs"]
                if c in tracks.columns]
        st.dataframe(tracks[show], use_container_width=True, hide_index=True, height=460)
        st.download_button("Download track summary CSV", tracks.to_csv(index=False),
                           file_name="track_summary.csv", mime="text/csv")

# --------------------------------------------------------------- interpretation
st.subheader("Automated interpretation")
st.caption(
    "A language model reads the measurements and writes them up. It performs no "
    "analysis of its own and sees no video - only the evidence brief below, which is "
    "built entirely from the numbers already on this page."
)

ipath = interp.interpretation_path(OUT_DIR, tag)
brief = interp.build_evidence(summary, kin, events, tracks)
saved = load_json(ipath)

ic1, ic2 = st.columns([3, 1], gap="large")
with ic2:
    st.markdown("**Evidence brief**")
    st.caption(
        f"{len(brief)} sections, {len(json.dumps(brief, default=str)) / 1024:.1f} KB, "
        f"{sum(len(v) for v in brief.get('notable_vehicles', {}).values())} notable vehicles. "
        "One API call per run, cached to disk."
    )
    st.download_button("Download evidence brief JSON",
                       json.dumps(brief, indent=1, default=str),
                       file_name=f"evidence_brief{'_' + tag if tag else ''}.json",
                       mime="application/json")
    go_btn = st.button("Generate interpretation" if not saved else "Regenerate",
                       type="primary" if not saved else "secondary")
    if not api_key:
        st.caption("Optional AI write-up is disabled. Video analysis works without it.")

if go_btn:
    if not api_key:
        st.info("Optional AI write-up is not configured. Video analysis and results are unaffected.")
    else:
        with st.spinner("Reading the measurements..."):
            try:
                saved = interp.interpret(brief, api_key, llm_model, base_url=endpoint)
                with open(ipath, "w", encoding="utf-8") as fh:
                    json.dump(saved, fh, indent=1, default=str)
            except Exception as exc:  # surfaced, not swallowed - the key is the usual cause
                st.error(f"Interpretation failed: {type(exc).__name__}: {exc}")
                saved = {}

with ic1:
    if not saved:
        st.info("No interpretation generated for this run yet.")
    else:
        if saved.get("headline"):
            st.markdown(f"#### {saved['headline']}")
        if saved.get("scene"):
            st.write(saved["scene"])

if saved:
    def _ev(items):
        return (" &nbsp;·&nbsp; ".join(f"<code>{e}</code>" for e in items)) if items else ""

    findings = saved.get("findings", [])
    if findings:
        st.markdown("##### Findings")
        for f in findings:
            sev = str(f.get("severity", "low")).lower()
            col = SEV_COLOR.get(sev, "#7d8794")
            st.markdown(
                f"<div style='border-left:3px solid {col};padding:.15rem 0 .35rem .7rem;"
                f"margin:.5rem 0'>"
                f"<span class='pill' style='background:{col}22;color:{col}'>{sev}</span>"
                f"&nbsp;<b>{f.get('title', '')}</b><br>"
                f"<span style='color:#c3cad6'>{f.get('detail', '')}</span><br>"
                f"<span style='font-size:.75rem;color:#8b95a5'>evidence: "
                f"{_ev(f.get('evidence', []))}</span></div>",
                unsafe_allow_html=True,
            )

    b1, b2 = st.columns(2, gap="large")
    with b1:
        beh = saved.get("behaviours", [])
        if beh:
            st.markdown("##### Behaviour patterns")
            for b in beh:
                st.markdown(
                    f"**{b.get('pattern', '')}**  \n"
                    f"<span style='color:#c3cad6'>{b.get('detail', '')}</span>  \n"
                    f"<span style='font-size:.75rem;color:#8b95a5'>evidence: "
                    f"{_ev(b.get('evidence', []))}</span>",
                    unsafe_allow_html=True,
                )
    with b2:
        anom = saved.get("anomalies", [])
        if anom:
            st.markdown("##### Anomalies")
            for a in anom:
                conf = str(a.get("confidence", "low")).lower()
                col = SEV_COLOR.get({"high": "high", "medium": "medium"}.get(conf, "low"),
                                    "#7d8794")
                st.markdown(
                    f"**{a.get('what', '')}** "
                    f"<span class='pill' style='background:{col}22;color:{col}'>"
                    f"{conf} confidence</span>  \n"
                    f"<span style='color:#c3cad6'>{a.get('why_flagged', '')}</span>  \n"
                    f"<span style='font-size:.78rem;color:#8b95a5'>Innocent reading: "
                    f"{a.get('alternative_explanation', 'n/a')}</span>  \n"
                    f"<span style='font-size:.75rem;color:#8b95a5'>evidence: "
                    f"{_ev(a.get('evidence', []))}</span>",
                    unsafe_allow_html=True,
                )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        cav = saved.get("data_caveats", [])
        if cav:
            st.markdown("##### Read this before quoting the numbers")
            for c in cav:
                st.markdown(f"<div class='caveat'>{c}</div>", unsafe_allow_html=True)
    with c2:
        rec = saved.get("recommended_checks", [])
        if rec:
            st.markdown("##### What this clip cannot settle")
            for r in rec:
                st.markdown(f"- {r}")

    meta = saved.get("_meta", {})
    if meta:
        st.caption(
            f"{meta.get('model', 'model')} · {meta.get('input_tokens', '?')} input / "
            f"{meta.get('output_tokens', '?')} output tokens · "
            f"saw {len(meta.get('evidence_keys', []))} evidence sections, no video"
        )
    with st.expander("Exactly what the model was given"):
        st.caption(
            "The full brief, verbatim. Every figure in the write-up above must trace "
            "back to something in here - the system prompt forbids introducing any "
            "number that is not present, and requires an `evidence` list per claim."
        )
        st.json(brief, expanded=False)
    st.download_button("Download interpretation JSON", json.dumps(saved, indent=1, default=str),
                       file_name=f"interpretation{'_' + tag if tag else ''}.json",
                       mime="application/json")

# ------------------------------------------------------------------ limitations
st.divider()
with st.expander("Limitations and what we deliberately do not claim", expanded=False):
    st.markdown(
        """
| Area | What we claim | What we do **not** claim |
|---|---|---|
| **Classes** | Fine-grained classes (car, LGV, HGV, bus, motorcycle, bicycle, pedestrian) decided from the vehicle's **measured** ground length, with the deciding signal recorded per vehicle in `class_source` | Not the detector's own class head - it calls 92% of everything "car" from this altitude. Measured length is biased upwards by vehicle height, so sizes are consistent *relative* measurements, not catalogue dimensions. Bus vs HGV is a tie size cannot break, so the model decides it |
| **Acceleration** | Per-vehicle p95 accel / p5 decel in m/s^2, over a smoothed ground track | Not instantaneous peaks. Acceleration is a second derivative, so raw extremes are dominated by pixel jitter; they are kept in the CSV for audit and flagged when beyond 10 m/s^2 |
| **Speed** | Estimated km/h from a telemetry-derived ground plane | Not survey-grade. Assumes flat ground; uses the box bottom as the road-contact point |
| **Stationary** | "Stationary vehicle **candidate**", cross-referenced against detected queues | Not an incident. One clip cannot separate a breakdown from a red light |
| **Wrong way** | Movement against the **observed dominant flow** of that part of the road, learned from the data | Not a legal violation - we have no HD map of legal carriageway directions |
| **Conflicts** | "**Potential** conflict" from a constant-velocity CPA projection, filtered by closing rate and persistence | Not a collision, and not a certified traffic-conflict technique score |
| **Congestion** | A transparent weighted score for operator triage | Not a certified level of service - that needs lane geometry and capacity |
| **Coordinates** | Metres relative to the aircraft, in a common ground frame | Not absolute survey coordinates, though GPS is recorded so results could be georeferenced |
"""
    )
