"""Read-only, token-weighted next-token evaluation on a fixed JSONL dataset."""
import hashlib
import importlib.util
import json
import math
import sys
from functools import lru_cache
from pathlib import Path


def hash_weights(base, engram):
    digest = hashlib.sha256()
    for weights in (base, engram):
        for chunk in weights.detach().split(10_000_000):
            data = chunk.cpu().float().contiguous().numpy().astype("<f4", copy=False)
            digest.update(memoryview(data).cast("B"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_dataset(path, vocab_size, max_seq_len):
    raw = Path(path).read_bytes()
    sequences = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            ids = record["input_ids"]
            if not isinstance(ids, list) or not 2 <= len(ids) <= max_seq_len + 1:
                raise ValueError
            if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid held-out token sequence on line {number}") from exc
        sequences.append(tuple(ids))
    if not sequences:
        raise ValueError("Held-out JSONL contains no sequences")
    return tuple(sequences), hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=1)
def load_client(path):
    spec = importlib.util.spec_from_file_location("crowdgpt_heldout_client", path)
    if spec is None or spec.loader is None:
        raise ValueError("Invalid CrowdGPT client path")
    client = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = client
    try:
        spec.loader.exec_module(client)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return client


def evaluate(path, client_path, device_name, config, weights, engram):
    import torch
    import torch.nn.functional as F

    sequences, digest = load_dataset(path, config["vocabSize"], config["maxSeqLen"])
    client = load_client(client_path)
    if any(client.MODEL_CONFIG.get(key) != config[key] for key in (
        "vocabSize", "dim", "nLayers", "nHeads", "nKvHeads", "headDim",
        "maxSeqLen", "mlpHidden", "weightTying",
    )) or client._calc_model_size() != weights.numel() or not client.ENGRAM_ACTIVE:
        raise ValueError("Held-out evaluator requires a matching client with Engram enabled")

    device = torch.device(device_name)
    model = client.SotaGPT(device.type.upper(), device, gpu_layers=config["nLayers"],
                           sl=max(map(len, sequences)) - 1, u8_override=False)
    model.eval()
    loss_sum = 0.0
    tokens = 0
    with torch.inference_mode():
        model.load_base_weights(weights)
        table = model.engram.table.weight
        table.copy_(engram.view_as(table))
        head = model.head_weight()
        for sequence in sequences:
            ids = torch.tensor(sequence, device=device, dtype=torch.long)
            hidden = model.forward_features(ids[:-1].unsqueeze(0), use_ckpt=False)[0]
            for offset in range(0, hidden.shape[0], 32):
                logits = F.linear(hidden[offset:offset + 32], head).float()
                targets = ids[offset + 1:offset + 1 + logits.shape[0]]
                loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
                tokens += targets.numel()

    loss = loss_sum / tokens
    if not math.isfinite(loss):
        raise ValueError("Held-out loss is non-finite")
    return {"loss": loss, "tokens": tokens, "examples": len(sequences), "datasetSha256": digest}
