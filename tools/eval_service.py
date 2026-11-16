#!/usr/bin/env python3
"""HTTP wrapper around ``eval_local.py`` for the orchestrator's eval URL.

    CROWDGPT_EVAL_MANIFEST=/data/manifest.json python tools/eval_service.py --port 7400
    # then set CROWDGPT_EVAL_URL=http://127.0.0.1:7400 on the orchestrator

The service is deliberately stateless: one POST evaluates the current
checkpoint and returns its JSON report. It does not cache by round, so a
retried request cannot silently reuse a stale loss.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
APP = FastAPI(title="CrowdGPT eval sidecar")


@APP.get("/health")
async def health():
    return {"ok": True, "manifest": os.getenv("CROWDGPT_EVAL_MANIFEST")}


@APP.post("/eval")
async def evaluate_round(body: dict):
    manifest = os.getenv("CROWDGPT_EVAL_MANIFEST", "")
    if not manifest:
        raise HTTPException(503, {"reason": "eval_manifest_not_configured"})
    checkpoint = body.get("checkpoint") or str(ROOT / "checkpoints" / "model_latest.bin")
    root = os.getenv("CROWDGPT_EVAL_ROOT") or str(Path(manifest).parent)
    cmd = [sys.executable, str(ROOT / "tools" / "eval_local.py"),
           "--manifest", manifest, "--root", root, "--checkpoint", checkpoint,
           "--mode", os.getenv("CROWDGPT_EVAL_MODE", "checkpoint"),
           "--windows", str(int(body.get("windows", 16))),
           "--device", os.getenv("CROWDGPT_EVAL_DEVICE", "cpu")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise HTTPException(500, {"reason": "eval_failed", "stderr": proc.stderr[-500:]})
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise HTTPException(500, {"reason": "eval_output_invalid"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7400)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(APP, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
