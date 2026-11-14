#!/usr/bin/env python3
"""Evaluate a published checkpoint over a hash-pinned manifest.

    python tools/eval_local.py --manifest /path/manifest.json --mode uniform
    python tools/eval_local.py --manifest /path/manifest.json \
        --checkpoint checkpoints/model_latest.bin --mode checkpoint

``uniform`` is the reference baseline (logits = ln(vocab)); it proves the
manifest, windows, and CI path without loading a model. ``checkpoint`` loads
the flat fp32 checkpoint through the client model and is the real evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NG = ROOT.parent / "Crowdgpt-ng"
sys.path.insert(0, str(NG))
sys.path.insert(0, str(ROOT))

from crowdgpt_ng.eval import evaluate, load_manifest  # noqa: E402


def checkpoint_logits(path: Path, manifest, device_name: str):
    import torch
    import client as ng_client

    device = torch.device(device_name)
    model = ng_client.SotaGPT(device.type.upper() if device.type != "cpu" else "CPU",
                              device, gpu_layers=ng_client.N_LAYERS, bs=1,
                              sl=manifest.seq_len)
    weights = np.fromfile(path, dtype=np.float32)
    if weights.size != ng_client._calc_model_size():
        raise SystemExit(f"checkpoint has {weights.size} params, model expects "
                         f"{ng_client._calc_model_size()}")
    model.load_base_weights(torch.from_numpy(weights))
    model.eval()

    def logits_fn(input_ids):
        with torch.no_grad():
            x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            hidden = model.forward_features(x, use_ckpt=False)
            return model.lm_head(hidden)[0].float().cpu().numpy()
    return logits_fn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--mode", choices=("uniform", "checkpoint"), default="uniform")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model_latest.bin"))
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.mode == "uniform":
        vocab = manifest.vocab_size
        baseline = math.log(vocab)

        def logits_fn(_inputs):
            return np.full((manifest.seq_len, vocab), baseline, dtype=np.float64)
        model_id = "uniform-reference"
    else:
        logits_fn = checkpoint_logits(Path(args.checkpoint), manifest, args.device)
        model_id = str(args.checkpoint)

    report = evaluate(manifest, logits_fn, root=args.root, count=args.windows,
                      seed=args.seed, model_id=model_id, code_commit="local")
    print(json.dumps({
        "loss": report.loss, "ciLow": report.ci_low, "ciHigh": report.ci_high,
        "windows": report.windows, "tokens": report.tokens,
        "manifestSha256": report.manifest_sha256, "modelId": report.model_id,
        "mode": args.mode,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
