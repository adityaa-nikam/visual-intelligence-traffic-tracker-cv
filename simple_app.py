"""Simple upload-first interface for Drone Traffic Intelligence."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")

st.set_page_config(page_title="Traffic Video Analyzer", page_icon="🚦", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1200px; padding-top: 2rem;}
  [data-testid="stMetric"] {background:#f7f9fc; border:1px solid #e7ebf0;
    border-radius:12px; padding:14px;}
  [data-testid="stMetricLabel"] p {color:#5d6878; font-size:.85rem;}
  .stCaption {color:#5d6878;}
</style>
""", unsafe_allow_html=True)


def available_tags() -> list[str]:
    summaries = glob.glob(os.path.join(OUT_DIR, "summary*.json"))
    found = []
    for path in summaries:
        suffix = os.path.basename(path)[len("summary"):-len(".json")]
        found.append((os.path.getmtime(path), suffix.lstrip("_")))
    return [tag for _, tag in sorted(found, reverse=True)]


def artifact(stem: str, ext: str, tag: str) -> str:
    return os.path.join(OUT_DIR, f"{stem}{'_' + tag if tag else ''}{ext}")


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def run_uploaded_video(video_file, telemetry_file, clip_start: float) -> None:
    suffix = Path(video_file.name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
        st.error("Please upload an MP4, MOV, M4V, AVI, or MKV video.")
        return

    tag = "upload_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    upload_dir = os.path.join(OUT_DIR, "uploads")
    config_dir = os.path.join(OUT_DIR, "upload_configs")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, tag + suffix)
    with open(video_path, "wb") as file:
        file.write(video_file.getbuffer())

    config_path = os.path.join(config_dir, tag + ".yaml")
    with open(os.path.join(BASE_DIR, "config", "config.yaml"), "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["video"]["path"] = video_path
    config["video"]["clip_start_s"] = float(clip_start)
    config["video"]["frame_stride"] = 3
    config["telemetry"]["srt_path"] = None
    config["segmentation"]["enabled"] = False
    config["output"]["dir"] = OUT_DIR

    weights = str(config["detector"].get("weights", ""))
    if weights.lower().endswith(".onnx") and not os.path.isfile(os.path.join(BASE_DIR, weights)):
        # Model weights are intentionally not committed. Ultralytics fetches this
        # standard checkpoint on first use if it is not already cached locally.
        config["detector"]["weights"] = "yolov8s.pt"

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    command = [sys.executable, "-u", "run_pipeline.py", "--config", config_path,
               "--video", video_path, "--stride", "3", "--tag", tag]
    if telemetry_file is not None:
        telemetry_path = os.path.join(upload_dir, tag + ".srt")
        with open(telemetry_path, "wb") as file:
            file.write(telemetry_file.getbuffer())
        command.extend(["--srt", telemetry_path])

    status = st.status("Starting video analysis…", expanded=True)
    progress = st.progress(0, text="Preparing video")
    log_view = st.empty()
    recent_logs: list[str] = []
    try:
        process = subprocess.Popen(
            command, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        total_frames = None
        for output in process.stdout:
            line = output.rstrip()
            if not line:
                continue
            recent_logs.append(line)
            recent_logs = recent_logs[-8:]
            video_info = re.search(r"(\d+) frames\s+stride=(\d+)", line)
            if video_info:
                total_frames = int(video_info.group(1))
                status.update(label=f"Analyzing {total_frames:,} frames…")
            frame_info = re.search(r"(\d+) frames\s+[\d.]+ fps", line)
            if frame_info and total_frames:
                done = min(int(frame_info.group(1)), total_frames)
                progress.progress(min(85, int(done / total_frames * 85)),
                                  text=f"Finding and tracking road users: {done:,} / {total_frames:,} frames")
            elif "[4/6]" in line:
                progress.progress(88, text="Summarizing traffic")
            elif "[6/6]" in line:
                progress.progress(94, text="Creating annotated video")
            log_view.code("\n".join(recent_logs), language="text")
        exit_code = process.wait()
    except Exception as exc:
        status.update(label="Could not start analysis", state="error")
        st.error(str(exc))
        return

    if exit_code != 0:
        status.update(label="Analysis stopped with an error", state="error")
        st.error("The pipeline could not finish. See the messages above for details.")
        return

    progress.progress(100, text="Analysis complete")
    status.update(label="Analysis complete", state="complete", expanded=False)

    rendered = os.path.join(OUT_DIR, f"annotated_{tag}.mp4")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and os.path.exists(rendered):
        subprocess.run(
            [ffmpeg, "-y", "-i", rendered, "-vcodec", "libx264", "-pix_fmt",
             "yuv420p", "-crf", "28", rendered[:-4] + "_web.mp4"],
            cwd=BASE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    st.cache_data.clear()
    st.session_state["selected_run"] = tag
    st.rerun()


st.title("Traffic Video Analyzer")
st.write("Upload an aerial traffic video to see detected road users, vehicle paths, and a traffic summary.")

with st.container(border=True):
    st.subheader("1 · Upload a video")
    st.caption("For real-world speeds and distances, include the DJI subtitle file recorded with this video.")
    with st.form("video_upload"):
        video_file = st.file_uploader("Video file", type=["mp4", "mov", "m4v", "avi", "mkv"])
        telemetry_file = st.file_uploader("DJI telemetry file (.srt, optional)", type=["srt"])
        clip_start = st.number_input(
            "Start time in original recording (seconds)", min_value=0.0, value=0.0,
            step=1.0,
        ) if telemetry_file is not None else 0.0
        submitted = st.form_submit_button("Analyze video", type="primary", disabled=video_file is None)
    if submitted and video_file is not None:
        run_uploaded_video(video_file, telemetry_file, clip_start)

tags = available_tags()
if not tags:
    st.info("Your results will appear here after the first video finishes processing.")
    st.stop()

with st.sidebar:
    st.subheader("Saved analyses")
    previous = st.session_state.pop("selected_run", tags[0])
    selected = st.selectbox("Choose a video", tags,
                            index=tags.index(previous) if previous in tags else 0,
                            format_func=lambda value: value.replace("_", " "))
    st.page_link("advanced_app.py", label="More charts and technical details", icon="📊")

summary = read_json(artifact("summary", ".json", selected))
counts = summary.get("counts", {})
speed = summary.get("speed", {})
interactions = summary.get("interactions", {})
video = summary.get("video", {})
run_info = summary.get("run", {})

st.divider()
st.subheader("2 · Review results")
run_caption = f"{video.get('width', '?')} × {video.get('height', '?')} px"
if video.get("fps"):
    run_caption += f" · {float(video['fps']):.1f} frames per second"
if run_info.get("created_utc"):
    run_caption += f" · analyzed {run_info['created_utc']}"
st.caption(run_caption)

metric_cols = st.columns(4)
metric_cols[0].metric("Road users", counts.get("total", 0))
metric_cols[1].metric("Vehicles", counts.get("motor_vehicles", 0))
metric_cols[2].metric("People & two-wheelers", counts.get("vulnerable", 0))
mean_speed = speed.get("mean_moving_speed")
if mean_speed is not None and speed.get("unit") == "km/h":
    metric_cols[3].metric("Estimated moving speed", f"{mean_speed} km/h")
elif mean_speed is not None:
    metric_cols[3].metric("Average image speed", f"{mean_speed} px/s")
else:
    metric_cols[3].metric("Potential conflicts", interactions.get("potential_conflicts", 0))

video_name = artifact("annotated", ".mp4", selected)
web_video = video_name[:-4] + "_web.mp4"
playable = web_video if os.path.exists(web_video) else video_name
left, right = st.columns([1.6, 1], gap="large")
with left:
    st.markdown("#### Annotated video")
    if os.path.exists(playable):
        st.video(playable)
        st.caption("Boxes show detections; labels and trails show each tracked road user.")
    else:
        st.info("No annotated video was created for this analysis.")
with right:
    st.markdown("#### What was detected")
    class_counts = counts.get("by_class", {})
    if class_counts:
        class_data = pd.DataFrame({"Road user type": list(class_counts), "Count": list(class_counts.values())})
        st.bar_chart(class_data.set_index("Road user type"), horizontal=True, height=290)
    else:
        st.info("No road users were detected.")
    if interactions:
        st.metric("Potential conflicts", interactions.get("potential_conflicts", 0))
        st.caption("These are motion-based alerts for review, not confirmed collisions.")

overview_tab, vehicles_tab, events_tab, details_tab = st.tabs(
    ["Overview", "Vehicles", "Events", "Run details"]
)
with overview_tab:
    map_path = artifact("trajectory_map", ".png", selected)
    if os.path.exists(map_path):
        st.markdown("#### Where road users travelled")
        st.image(map_path, width="stretch")
    st.download_button(
        "Download summary", json.dumps(summary, indent=2, default=str),
        file_name=f"traffic_summary_{selected}.json", mime="application/json",
    )

with vehicles_tab:
    vehicle_table = read_csv(artifact("track_summary", ".csv", selected))
    if vehicle_table.empty:
        st.info("No vehicle table is available for this run.")
    else:
        st.caption(f"One row per tracked road user · {len(vehicle_table):,} total")
        st.dataframe(vehicle_table, width="stretch", hide_index=True, height=420)
        st.download_button(
            "Download vehicle list", vehicle_table.to_csv(index=False),
            file_name=f"vehicles_{selected}.csv", mime="text/csv",
        )

with events_tab:
    event_table = read_csv(artifact("events", ".csv", selected))
    if event_table.empty:
        st.info("No events were recorded for this run.")
    else:
        if "event_type" in event_table.columns:
            event_counts = event_table["event_type"].value_counts().rename_axis("Event type").reset_index(name="Count")
            st.bar_chart(event_counts.set_index("Event type"), horizontal=True)
        st.dataframe(event_table, width="stretch", hide_index=True, height=420)
        st.download_button(
            "Download event list", event_table.to_csv(index=False),
            file_name=f"events_{selected}.csv", mime="text/csv",
        )

with details_tab:
    calibration = summary.get("calibration", {})
    if calibration.get("usable"):
        st.success("Real-world scale estimated from DJI flight telemetry.")
        st.write(f"Altitude: {calibration.get('altitude_m')} m · Ground scale at image centre: "
                 f"{calibration.get('metres_per_px_at_centre')} m/px")
    else:
        st.info("No matching telemetry was used. Speeds and distances are in pixels, not metres.")
    detector = summary.get("detector", {})
    st.write(f"**Detector:** {Path(str(detector.get('weights', 'unknown'))).name} · "
             f"**Tracker:** {summary.get('tracker', {}).get('type', 'unknown')}")
    if run_info:
        st.caption(f"Run ID: {run_info.get('tag')} · Configuration fingerprint: "
                   f"{run_info.get('config_sha256', 'unavailable')}")
    st.caption("Speeds, vehicle classes, road masks, and conflict alerts are estimates. "
               "Review the video before treating an alert as confirmed.")
