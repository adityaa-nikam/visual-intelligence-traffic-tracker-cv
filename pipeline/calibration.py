"""
Ground-plane calibration from DJI flight telemetry.

DJI aircraft embed a per-frame telemetry record as a tx3g subtitle track inside
the MP4. Each record carries the relative altitude above the take-off point, the
lens focal length (as a 35 mm equivalent) and the gimbal attitude. That is
exactly enough to project any image pixel onto the road surface and get a
position in METRES, which is what turns pixel trails into real traffic
engineering measurements (km/h, metres of queue, time-to-collision).

Camera / world conventions
--------------------------
World frame is local ENU, origin directly below the aircraft:
    X = east (m), Y = north (m), Z = up (m).
The road is assumed to be the flat plane Z = 0 and the camera sits at Z = h.

Camera frame is the usual pinhole layout: x right, y down, z along the optical
axis. Image pixel (u, v) therefore back-projects to the camera-space ray
    [(u - cx) / f, (v - cy) / f, 1]
which we rotate into the world and intersect with Z = 0.

Documented limitations
----------------------
* The flat-ground assumption ignores road gradient and elevation change. Over an
  intersection-sized footprint the resulting error is small.
* Positions are relative to the aircraft, not absolute survey coordinates. GPS
  latitude/longitude is recorded so results *could* be georeferenced, but we do
  not claim survey-grade absolute accuracy.
* Objects are located by the bottom-centre of their box, i.e. their road-contact
  point. Tall vehicles are therefore placed slightly better than a centroid
  would place them, but a box bottom is still an approximation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

# One regex pass per SRT record; DJI's field set varies slightly by model, so
# every field is looked up independently and missing ones fall back to a default.
_FRAME_RE = re.compile(r"FrameCnt\s*:\s*(\d+)")
_NUM = r"(-?\d+(?:\.\d+)?)"
_FIELD_RES = {
    "focal_len": re.compile(r"focal_len\s*:\s*" + _NUM),
    "dzoom_ratio": re.compile(r"dzoom_ratio\s*:\s*" + _NUM),
    "latitude": re.compile(r"latitude\s*:\s*" + _NUM),
    "longitude": re.compile(r"longitude\s*:\s*" + _NUM),
    "rel_alt": re.compile(r"rel_alt\s*:\s*" + _NUM),
    "abs_alt": re.compile(r"abs_alt\s*:\s*" + _NUM),
    "gb_yaw": re.compile(r"gb_yaw\s*:\s*" + _NUM),
    "gb_pitch": re.compile(r"gb_pitch\s*:\s*" + _NUM),
    "gb_roll": re.compile(r"gb_roll\s*:\s*" + _NUM),
}


@dataclass
class Telemetry:
    """One flight-data record, as read from the SRT."""

    frame: int
    rel_alt: float
    focal_len: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    latitude: float
    longitude: float
    dzoom: float = 1.0


def parse_dji_srt(path: str) -> dict[int, Telemetry]:
    """Parse a DJI telemetry SRT into {original_frame_index: Telemetry}."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    records: dict[int, Telemetry] = {}
    # Records are separated by blank lines; the payload line holds the fields.
    for block in text.split("\n\n"):
        m = _FRAME_RE.search(block)
        if not m:
            continue

        def get(key: str, default: float) -> float:
            fm = _FIELD_RES[key].search(block)
            return float(fm.group(1)) if fm else default

        frame = int(m.group(1))
        records[frame] = Telemetry(
            frame=frame,
            rel_alt=get("rel_alt", 0.0),
            focal_len=get("focal_len", 0.0),
            yaw_deg=get("gb_yaw", 0.0),
            pitch_deg=get("gb_pitch", -90.0),
            roll_deg=get("gb_roll", 0.0),
            latitude=get("latitude", 0.0),
            longitude=get("longitude", 0.0),
            dzoom=get("dzoom_ratio", 1.0),
        )
    return records


def median_telemetry(records: dict[int, Telemetry]) -> Telemetry | None:
    """
    Collapse the flight log to one representative pose.

    Both source clips are shot from a near-stationary hover, so a single pose is
    an accurate and much simpler model than re-deriving a homography every
    frame. Medians are used because they ignore GPS/altimeter spikes.
    """
    if not records:
        return None
    vals = list(records.values())

    def med(attr: str) -> float:
        return float(np.median([getattr(v, attr) for v in vals]))

    return Telemetry(
        frame=-1,
        rel_alt=med("rel_alt"),
        focal_len=med("focal_len"),
        yaw_deg=med("yaw_deg"),
        pitch_deg=med("pitch_deg"),
        roll_deg=med("roll_deg"),
        latitude=med("latitude"),
        longitude=med("longitude"),
        dzoom=med("dzoom"),
    )


def hover_stability(records: dict[int, Telemetry]) -> dict[str, float]:
    """Spread of the flight log, reported so the hover assumption is auditable."""
    if not records:
        return {}
    vals = list(records.values())
    out: dict[str, float] = {}
    for attr in ("rel_alt", "yaw_deg", "pitch_deg", "roll_deg"):
        arr = np.array([getattr(v, attr) for v in vals], dtype=float)
        out[f"{attr}_p5_p95_spread"] = float(np.percentile(arr, 95) - np.percentile(arr, 5))
    return out


class GroundProjector:
    """
    Maps image pixels to metric ground coordinates on the plane Z = 0.

    Built from a single hover pose, so it is a fixed pixel -> ground mapping.
    """

    def __init__(
        self,
        image_w: int,
        image_h: int,
        rel_alt_m: float,
        focal_len_35mm: float,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float = 0.0,
        sensor_width_35mm: float = 36.0,
        dzoom: float = 1.0,
    ):
        self.image_w = image_w
        self.image_h = image_h
        self.height_m = float(rel_alt_m)
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = float(pitch_deg)

        # 35 mm-equivalent focal length -> pixels. This scales with the frame
        # width, so it stays correct after the 4K clip is downscaled to 1920.
        self.f_px = (focal_len_35mm * max(dzoom, 1e-6) / sensor_width_35mm) * image_w
        self.cx = image_w / 2.0
        self.cy = image_h / 2.0

        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)

        # Optical axis: bearing `yaw` clockwise from north, elevated by `pitch`.
        cp = math.cos(pitch)
        fwd = np.array([math.sin(yaw) * cp, math.cos(yaw) * cp, math.sin(pitch)])
        fwd /= np.linalg.norm(fwd)

        # Image +x is to the camera's right, horizontal when roll = 0.
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        right /= np.linalg.norm(right)

        # Image +y points down the sensor.
        down = np.cross(fwd, right)
        down /= np.linalg.norm(down)

        if abs(roll) > 1e-6:  # rotate the sensor axes about the optical axis
            cr, sr = math.cos(roll), math.sin(roll)
            right, down = cr * right + sr * down, -sr * right + cr * down

        # Columns map camera-space (x, y, z) into world directions.
        self.R = np.column_stack([right, down, fwd])
        self.cam_pos = np.array([0.0, 0.0, self.height_m])

        # Valid only if the camera actually looks at the ground.
        self.usable = self.height_m > 1.0 and fwd[2] < -0.05

    def pixels_to_ground(self, uv: np.ndarray) -> np.ndarray:
        """
        Project an (N, 2) array of pixels to (N, 2) ground metres (east, north).

        Rays that point at or above the horizon yield NaN rather than a bogus
        coordinate, so downstream code can drop them explicitly.
        """
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        n = uv.shape[0]
        out = np.full((n, 2), np.nan)
        if n == 0 or not self.usable:
            return out

        rays_cam = np.stack(
            [
                (uv[:, 0] - self.cx) / self.f_px,
                (uv[:, 1] - self.cy) / self.f_px,
                np.ones(n),
            ],
            axis=1,
        )
        rays_world = rays_cam @ self.R.T

        # Intersect with Z = 0: cam_z + t * ray_z = 0.
        rz = rays_world[:, 2]
        valid = rz < -1e-6
        t = np.zeros(n)
        t[valid] = self.height_m / (-rz[valid])

        out[valid, 0] = self.cam_pos[0] + t[valid] * rays_world[valid, 0]
        out[valid, 1] = self.cam_pos[1] + t[valid] * rays_world[valid, 1]
        return out

    def ground_scale_at(self, u: float, v: float) -> float:
        """Local metres-per-pixel at one image location (for reporting)."""
        p = self.pixels_to_ground(np.array([[u, v], [u + 1.0, v]]))
        if np.isnan(p).any():
            return float("nan")
        return float(np.linalg.norm(p[1] - p[0]))

    def describe(self) -> dict:
        """Human-readable calibration summary shown in the dashboard/README."""
        centre_gsd = self.ground_scale_at(self.cx, self.cy)
        corners = self.pixels_to_ground(
            np.array(
                [
                    [0, self.image_h - 1],
                    [self.image_w - 1, self.image_h - 1],
                    [0, self.image_h * 0.55],
                    [self.image_w - 1, self.image_h * 0.55],
                ]
            )
        )
        span = float("nan")
        if not np.isnan(corners[:2]).any():
            span = float(np.linalg.norm(corners[1] - corners[0]))
        return {
            "usable": bool(self.usable),
            "altitude_m": round(self.height_m, 2),
            "focal_px": round(self.f_px, 1),
            "gimbal_pitch_deg": round(self.pitch_deg, 2),
            "gimbal_yaw_deg": round(self.yaw_deg, 2),
            "metres_per_px_at_centre": round(centre_gsd, 4) if centre_gsd == centre_gsd else None,
            "bottom_edge_ground_width_m": round(span, 1) if span == span else None,
        }


def build_projector(cfg: dict, image_w: int, image_h: int):
    """
    Assemble a GroundProjector from config + telemetry.

    Returns (projector_or_None, info_dict). A None projector is not a failure:
    the pipeline falls back to pixel-space analytics and every speed/distance
    figure is then reported as unavailable rather than invented.
    """
    srt_path = cfg.get("telemetry", {}).get("srt_path")
    info: dict = {"source": "none", "reason": ""}
    if not srt_path:
        info["reason"] = "no telemetry path configured"
        return None, info

    try:
        records = parse_dji_srt(srt_path)
    except FileNotFoundError:
        info["reason"] = f"telemetry file not found: {srt_path}"
        return None, info

    pose = median_telemetry(records)
    if pose is None or pose.rel_alt <= 1.0 or pose.focal_len <= 0.0:
        info["reason"] = "telemetry present but altitude/focal length unusable"
        return None, info

    proj = GroundProjector(
        image_w=image_w,
        image_h=image_h,
        rel_alt_m=pose.rel_alt,
        focal_len_35mm=pose.focal_len,
        pitch_deg=pose.pitch_deg,
        yaw_deg=pose.yaw_deg,
        roll_deg=pose.roll_deg,
        sensor_width_35mm=cfg.get("telemetry", {}).get("sensor_width_35mm", 36.0),
        dzoom=pose.dzoom,
    )
    if not proj.usable:
        info["reason"] = "gimbal pose does not intersect the ground plane"
        return None, info

    info.update(
        {
            "source": "dji_srt",
            "records": len(records),
            "gps": [round(pose.latitude, 6), round(pose.longitude, 6)],
            "hover_stability": hover_stability(records),
            **proj.describe(),
        }
    )
    return proj, info
