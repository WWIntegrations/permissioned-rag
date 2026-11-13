"""Evaluation harness: retrieval quality, ACL leakage, answers, load."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from .. import config


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=20
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            for key in ("Model Name", "Chip", "Total Number of Cores", "Memory"):
                if line.startswith(key + ":"):
                    info[key.lower().replace(" ", "_")] = line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def write_result(name: str, payload: dict) -> Path:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path
