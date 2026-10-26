"""
LLM interpretation layer - turns the measured artifacts into readable findings.

This module deliberately does NO analysis of its own. It assembles an evidence
brief from what the pipeline already measured, hands that brief to Claude, and
stores the reply. Every number the model can see comes from the pipeline; the
model's job is wording and pattern-spotting, not measurement.

Two properties matter more than anything else here:

1. The model cannot invent numbers, because it is only ever shown the brief and
   is instructed to cite the evidence key behind every claim. If a finding has no
   supporting figure in the brief, that is visible in the output.
2. The brief is a few kilobytes, not the 16 MB trajectory table. One API call per
   run, cached to disk, so re-opening the dashboard costs nothing.

Run standalone:
    python -m pipeline.interpret --tag v4
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

MODEL = "claude-opus-5"
DEFAULT_ENDPOINT = "https://api.anthropic.com"
MAX_NOTABLE = 12

SYSTEM_PROMPT = """You are a traffic engineer writing up one drone survey of a single \
intersection. You are given a JSON evidence brief produced by an automated \
vision pipeline.

HARD RULES - these are not style preferences:
1. Use ONLY figures that appear in the brief. Never estimate, round up, \
extrapolate, or introduce a number that is not there. If you want to say \
something the brief does not support, either drop it or state plainly that the \
data cannot settle it.
2. Cite your evidence. Every finding carries an `evidence` list naming the brief \
keys or vehicle IDs it rests on.
3. Hedge claims the data cannot prove. This is one clip from one camera. Say \
"estimated", "candidate", "potential", "consistent with". Never assert a \
collision, a violation, a driver intention, or a legal fact.
4. Respect the brief's own caveats - they are in `measurement_limits`. Do not \
present a measurement as more precise than that section allows. In particular, \
vehicle lengths are relative measurements biased by vehicle height, and \
acceleration extremes are differentiation noise.
5. Quantity over adjectives. "17 of 492 road users" beats "many road users".
6. If the evidence for something is thin (a handful of tracks), say how thin.

Reply with a single JSON object, no prose around it, no markdown fences:

{
  "headline": "one sentence a traffic engineer would accept as a summary",
  "scene": "2-3 sentences on what this location is doing during the clip",
  "findings": [
    {"title": "short", "severity": "high|medium|low",
     "detail": "what the numbers show and what it means operationally",
     "evidence": ["brief.key", "vehicle #123"]}
  ],
  "behaviours": [
    {"pattern": "short", "detail": "...", "evidence": [...]}
  ],
  "anomalies": [
    {"what": "short", "why_flagged": "the measurement that triggered it",
     "confidence": "high|medium|low",
     "alternative_explanation": "the innocent reading of the same data",
     "evidence": [...]}
  ],
  "data_caveats": ["things a reader must know before quoting these numbers"],
  "recommended_checks": ["what to measure next to settle what this clip cannot"]
}

Aim for 4-6 findings, 3-5 behaviours, 2-4 anomalies. Every anomaly MUST carry an
alternative_explanation - a stationary vehicle is a red light before it is an
incident."""


# --------------------------------------------------------------- evidence brief

def _top(df: pd.DataFrame, col: str, n: int, ascending: bool = False) -> list[dict]:
    """The n most extreme rows on `col`, as compact citable records."""
    if df.empty or col not in df:
        return []
    # dict.fromkeys de-duplicates while preserving order: `col` is often already in
    # the keep list (length_m), and duplicate columns break to_json(orient=records).
    keep = list(dict.fromkeys(
        [c for c in ["track_id", "class", col, "duration_s", "visible_s",
                     "length_m", "width_m", "moving_fraction"] if c in df]))
    sub = df.dropna(subset=[col]).sort_values(col, ascending=ascending).head(n)[keep]
    return json.loads(sub.to_json(orient="records"))


def build_evidence(summary: dict, kin: pd.DataFrame, events: pd.DataFrame,
                   tracks: pd.DataFrame) -> dict:
    """
    Assemble the brief. Aggregates plus a shortlist of individually notable
    vehicles, so the model can cite real IDs instead of speaking in generalities.
    """
    ev: dict = {
        "scene": {
            "source": "single hovering drone camera, one intersection",
            **{k: summary.get("video", {}).get(k)
               for k in ("duration_s", "fps", "frames_processed", "width", "height")
               if summary.get("video", {}).get(k) is not None},
            "calibration": {k: summary.get("calibration", {}).get(k)
                            for k in ("usable", "altitude_m",
                                      "metres_per_px_at_centre",
                                      "bottom_edge_ground_width_m")
                            if k in summary.get("calibration", {})},
        },
        "fleet": summary.get("counts", {}),
        "classification": summary.get("classification", {}),
        "kinematics": summary.get("kinematics", {}),
        "speed": summary.get("speed", {}),
        "congestion": summary.get("congestion", {}),
        "queues": summary.get("queues", {}),
        "turning_movements": summary.get("turning_movements", {}),
        "interactions": summary.get("interactions", {}),
        "stationary_candidates": summary.get("stationary_candidates", {}),
        "events": summary.get("events", {}),
        "tracking_quality": summary.get("id_stability", {}),
        "road_segmentation": summary.get("road_segmentation", {}),
    }

    # Individually notable vehicles - the tails of each distribution.
    notable: dict = {}
    if not kin.empty:
        n = max(3, MAX_NOTABLE // 4)
        notable["fastest"] = _top(kin, "p98_speed_kph", n)
        notable["hardest_braking"] = _top(kin, "p5_decel_ms2", n, ascending=True)
        notable["hardest_accelerating"] = _top(kin, "p95_accel_ms2", n)
        if "accel_exceeds_physical" in kin:
            flagged = kin[kin["accel_exceeds_physical"].fillna(False).astype(bool)]
            notable["flagged_implausible_accel"] = _top(flagged, "max_accel_ms2_raw", 3)
    if not tracks.empty:
        notable["longest_stopped"] = _top(tracks, "longest_stop_s", 4)
        notable["largest_measured"] = _top(tracks, "length_m", 4)
    ev["notable_vehicles"] = {k: v for k, v in notable.items() if v}

    # The most severe individual events, verbatim, so descriptions are not paraphrased.
    if not events.empty:
        cols = [c for c in ["event_type", "track_id", "secondary_track_id", "timestamp",
                            "duration", "severity", "value", "unit", "description"]
                if c in events.columns]
        order = {"high": 0, "medium": 1, "low": 2}
        e = events.copy()
        e["_o"] = e.get("severity", pd.Series(index=e.index)).map(order).fillna(3)
        top_events = e.sort_values(["_o", "duration"], ascending=[True, False]).head(15)
        ev["most_severe_events"] = json.loads(top_events[cols].to_json(orient="records"))

    ev["measurement_limits"] = [
        "One clip, one camera, one intersection. No before/after comparison exists.",
        "Speeds come from a telemetry-derived ground plane assuming flat ground, "
        "using the bounding-box bottom edge as the road-contact point. Estimates, "
        "not survey-grade measurements.",
        "Vehicle length and width are solved from an axis-aligned box and are biased "
        "upwards by vehicle height. They are consistent RELATIVE sizes, not catalogue "
        "dimensions, and the class thresholds are calibrated on this footage.",
        "Class labels above car/two-wheeler come from measured size, not from the "
        "detector's class head, which called 92% of detections 'car' from this "
        "altitude. Bus vs HGV is decided by the model, not by size.",
        "Acceleration is a second derivative, so raw per-vehicle extremes are "
        "dominated by pixel jitter. Use the p95/p5 figures; raw peaks beyond the "
        "physical limit are flagged, not corrected.",
        "'Potential conflict' means a time-to-collision threshold was crossed under a "
        "constant-velocity projection. No collision occurred and none is claimed.",
        "'Stationary candidate' and 'parked candidate' are dwell-time measurements. "
        "This clip cannot distinguish a breakdown from a red light or a legal stop.",
        "Wrong-way flags are deviations from the locally dominant flow direction, not "
        "confirmed violations - the dominant direction is inferred from traffic, not "
        "from signage or lane markings.",
        "Vehicles are only counted while visible to the camera; anything outside the "
        "frame or fully occluded does not exist in this data.",
    ]
    return ev


# --------------------------------------------------------------------- the call

def resolve_endpoint(base_url: str | None = None) -> str:
    """
    Where the request will actually go.

    The Anthropic SDK silently honours ANTHROPIC_BASE_URL, so on a machine with a
    proxy configured (this one has agentrouter.org set) both the API key and the
    evidence brief would leave for a third party with nothing on screen saying so.
    The endpoint is therefore resolved explicitly and shown in the UI.
    """
    return (base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_ENDPOINT).rstrip("/")


def _extract_json(raw: str) -> dict:
    """
    Pull the JSON object out of a model reply.

    Without assistant prefill the reply can arrive wrapped in a markdown fence, or
    with a sentence before or after the object, so the braces are located rather
    than assumed. Cheaper and more portable than a tool-call round trip.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    s = s.strip().removeprefix("json").strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return {"headline": raw[:400], "_parse_failed": True}
    for candidate in (s[start:end + 1], s[start:]):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {"headline": raw[:400], "_parse_failed": True}


def interpret(evidence: dict, api_key: str, model: str = MODEL,
              base_url: str | None = None) -> dict:
    """
    Send the brief to Claude and return the parsed interpretation.

    The JSON shape is requested in the system prompt and located in the reply by
    _extract_json. An earlier version forced it by prefilling an opening brace in an
    assistant turn, which is more reliable on the Anthropic API directly but is
    rejected outright by some gateways ("This model does not support assistant
    message prefill"), so the conversation now ends on the user turn.
    """
    import anthropic

    endpoint = resolve_endpoint(base_url)
    client = anthropic.Anthropic(api_key=api_key, base_url=endpoint)
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user",
             "content": "Evidence brief for this survey:\n\n"
                        + json.dumps(evidence, indent=1, default=str)
                        + "\n\nReply with the JSON object only - no markdown fence, "
                          "no commentary before or after it."},
        ],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    out = _extract_json(raw)
    out["_meta"] = {
        "model": msg.model,
        "endpoint": endpoint,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "evidence_keys": sorted(evidence.keys()),
    }
    return out


# --------------------------------------------------------------------- plumbing

def _p(out_dir: str, name: str, ext: str, tag: str) -> str:
    return os.path.join(out_dir, f"{name}{'_' + tag if tag else ''}{ext}")


def load_run(out_dir: str, tag: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def csv(name: str) -> pd.DataFrame:
        p = _p(out_dir, name, ".csv", tag)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    sp = _p(out_dir, "summary", ".json", tag)
    summary = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    return summary, csv("kinematics"), csv("events"), csv("track_summary")


def interpretation_path(out_dir: str, tag: str) -> str:
    return _p(out_dir, "interpretation", ".json", tag)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM interpretation of a pipeline run")
    ap.add_argument("--tag", default="", help="run tag, e.g. v4")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base-url", default=None,
                    help="API endpoint. Defaults to ANTHROPIC_BASE_URL if set, "
                         "else api.anthropic.com")
    ap.add_argument("--brief-only", action="store_true",
                    help="build and print the evidence brief without calling the API")
    args = ap.parse_args()

    if not args.tag:
        # Default to the newest run rather than the alphabetically first one.
        cands = [(os.path.getmtime(p), os.path.basename(p)[len("summary"):-len(".json")].lstrip("_"))
                 for p in glob.glob(os.path.join(args.out_dir, "summary*.json"))]
        if cands:
            args.tag = sorted(cands, reverse=True)[0][1]
            print(f"[interpret] no --tag given, using newest run: {args.tag or '(default)'}")

    summary, kin, events, tracks = load_run(args.out_dir, args.tag)
    if not summary:
        raise SystemExit(f"no summary found for tag {args.tag!r} in {args.out_dir}")

    ev = build_evidence(summary, kin, events, tracks)
    size = len(json.dumps(ev, default=str))
    print(f"[interpret] evidence brief: {len(ev)} sections, {size / 1024:.1f} KB, "
          f"{sum(len(v) for v in ev.get('notable_vehicles', {}).values())} notable vehicles")

    if args.brief_only:
        bp = _p(args.out_dir, "evidence_brief", ".json", args.tag)
        json.dump(ev, open(bp, "w", encoding="utf-8"), indent=1, default=str)
        print(f"[interpret] wrote {bp}")
        return

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Export it, or generate the "
                         "interpretation from the dashboard sidebar instead.")

    endpoint = resolve_endpoint(args.base_url)
    if endpoint != DEFAULT_ENDPOINT:
        print(f"[interpret] WARNING: sending the brief and your key to {endpoint}, "
              f"not to {DEFAULT_ENDPOINT}. Pass --base-url to change this.")
    print(f"[interpret] calling {args.model} at {endpoint} ...")
    result = interpret(ev, key, args.model, base_url=args.base_url)
    p = interpretation_path(args.out_dir, args.tag)
    json.dump(result, open(p, "w", encoding="utf-8"), indent=1, default=str)
    m = result.get("_meta", {})
    print(f"[interpret] wrote {p}  ({m.get('input_tokens')} in / "
          f"{m.get('output_tokens')} out tokens)")
    print(f"[interpret] headline: {result.get('headline', '')}")


if __name__ == "__main__":
    main()
