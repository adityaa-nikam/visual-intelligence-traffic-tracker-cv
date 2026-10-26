"""Download and verify the VisDrone aerial detector used for video uploads."""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "best.pt"
REPO_ID = "mshamrai/yolov8n-visdrone"
REVISION = "28ca510"
EXPECTED_SHA256 = "e5f2685462a10a24e05bd6e5db33e11eb46e0113c96b3440db61483d70831a60"
EXPECTED_NAMES = {
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
}


def main() -> int:
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(
        repo_id=REPO_ID,
        filename="best.pt",
        revision=REVISION,
        local_dir=str(WEIGHTS.parent),
    ))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checkpoint checksum mismatch: expected {EXPECTED_SHA256}, got {digest}. "
            "Do not use an unverified model file."
        )

    # Verify the checkpoint is loadable and its class order matches the adapter.
    from ultralytics import YOLO

    names = YOLO(str(path)).names
    names = set(names.values()) if isinstance(names, dict) else set(names)
    if names != EXPECTED_NAMES:
        raise RuntimeError(f"Unexpected model classes: {sorted(names)}")
    print(f"Verified aerial checkpoint: {path}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
