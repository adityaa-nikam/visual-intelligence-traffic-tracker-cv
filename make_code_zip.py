"""
Bundles the pipeline source into traffic_ai_code.zip for upload to Colab.

Only source and config - no videos, models or outputs, so the zip stays tiny.
"""

import os
import zipfile

FILES = [
    "run_pipeline.py",
    "app.py",
    "requirements.txt",
    "README.md",
    "ISSUES.md",
    "config/config.yaml",
    "config/bytetrack.yaml",
    "pipeline/__init__.py",
    "pipeline/calibration.py",
    "pipeline/detector.py",
    "pipeline/tracker.py",
    "pipeline/trajectory.py",
    "pipeline/analytics.py",
    "pipeline/interactions.py",
    "pipeline/anomalies.py",
    "pipeline/visualize.py",
]

OUT = "traffic_ai_code.zip"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        if os.path.exists(f):
            z.write(f)
        else:
            print(f"  (skipped, not present yet: {f})")

print(f"wrote {OUT}  {os.path.getsize(OUT) / 1024:.0f} KB")
