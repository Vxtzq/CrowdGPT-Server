#!/usr/bin/env python3
"""
CrowdGPT Server - Continuous FL Coordinator (Matchmaking Pools)

v2.3 — strict pooling, mandatory cross-check, FP32-first, solo-pool re-matching:
  - Pools require EXACT backend match.
  - Optimizer matching normalizes aliases (adamw32/adamw_fp32/adamw → adamw_fp32,
    adamw8/adamw_8bit/8bit → adamw8bit) and allows fp32/8bit AdamW to mix
    in the same pool ('any' stays a wildcard on both sides). Real divergence
    is caught by the mandatory cross-check (5x threshold).
  - Autocast matching is ASYMMETRIC:
      * FP32 pools accept any compatible client (bf16 joiners are forced to
        fp32 training via the task payload's crossCheck.targetAutocast).
      * BF16 pools accept only bf16-native clients.
  - Cross-check is MANDATORY. Pools that can't be verified are discarded.
  - Threshold: 5× core_dist or 2% floor (whichever is larger).
  - Matchmaking priority: FP32 forming pools > BF16 forming pools >
    new pool with client's native autocast.
  - v2.3: solo forming pools are re-matched on every poll (client escapes
    to a better-compatible pool immediately). Training pools and forming
    pools with 2+ clients remain locked.
  - Solo BF16 pools dissolve after BF16_SOLO_DISSOLVE_S (default 30 min).
  - Solo FP32 pools dissolve after FP32_SOLO_DISSOLVE_S (default 60 min).
"""

# ==================================================================
# VOCAB DIAGNOSTIC — read-only. Remove after diagnosis.
# ==================================================================
import os as _dg_os
import json as _dg_json
from pathlib import Path as _dg_Path

_DG_BASE = _dg_Path(__file__).resolve().parent
_DG_CKPT = _DG_BASE / "checkpoints"
_DG_DIM = 1536
_DG_N_LAYERS = 24
_DG_N_HEADS = 16
_DG_N_KV_HEADS = 4
_DG_HEAD_DIM = 96
_DG_MLP_HIDDEN = 2560
_DG_ENG_BUCKETS = 227865

def _dg_rest_params():
    d = _DG_DIM; r = 0
    for _ in range(_DG_N_LAYERS):
        r += d*2 + d*(_DG_N_HEADS*_DG_HEAD_DIM) + d*(_DG_N_KV_HEADS*_DG_HEAD_DIM)*2 + d*d + d*2
        r += d*_DG_MLP_HIDDEN*2 + _DG_MLP_HIDDEN*d
    return r + d*2

_DG_REST = _dg_rest_params()
_DG_ENG_PARAMS = _DG_ENG_BUCKETS * _DG_DIM

def _dg_vocab_from_params(p):
    v = (p - _DG_REST) / _DG_DIM
    return int(v) if v == int(v) and v > 0 else None

def _dg_log(msg): print(f"[VOCAB-DIAG] {msg}", flush=True)

_dg_log("=" * 72)
_dg_log(f"rest_params (non-vocab)     = {_DG_REST:,}")
_dg_log(f"engram_params               = {_DG_ENG_PARAMS:,}")
_dg_log("=" * 72)

def _dg_check(path, is_engram=False):
    if not path.exists():
        _dg_log(f"MISSING: {path.name}"); return
    b = path.stat().st_size
    _dg_log(f"{path.name}: {b:,} bytes")
    if is_engram:
        if b == _DG_ENG_PARAMS * 4: _dg_log("  -> fp32 engram OK")
        elif b == _DG_ENG_PARAMS * 2: _dg_log("  -> bf16 engram OK")
        else: _dg_log(f"  -> UNEXPECTED")
        return
    for db, lbl in ((4, "fp32"), (2, "bf16")):
        if b % db: continue
        v = _dg_vocab_from_params(b // db)
        if v is not None:
            tag = "OK" if v == 151669 else "STALE — needs migration"
            _dg_log(f"  -> {lbl}, inferred vocab = {v}  [{tag}]"); return
    _dg_log("  -> cannot infer vocab from size")

if _DG_CKPT.exists():
    _dg_check(_DG_CKPT / "model_latest.bin")
    _dg_check(_DG_CKPT / "engram_latest.bin", is_engram=True)
_dg_log("=" * 72)
# ==================================================================

import os, sys, io, json, zlib, gzip, time, struct, secrets, threading, requests
import math, random, hashlib, uuid, shutil, tempfile, gc, atexit, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

import numpy as np
import torch
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
import ipaddress
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from huggingface_hub import HfApi, login, hf_hub_download, CommitOperationAdd
    from safetensors.torch import save_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    log.warning("⚠️ huggingface_hub not installed - HF uploads disabled")

def read_exact(file_obj, num_bytes):
    data = bytearray()
    while len(data) < num_bytes:
        chunk = file_obj.read(num_bytes - len(data))
        if not chunk: break
        data.extend(chunk)
    return bytes(data)

from krum_stream import plan_round, stream_add_base_delta, read_engram_tail, open_delta, streaming_pairwise_sq_l2

# ============ CONFIGURATION ============
PORT = 5006
BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"
LATEST_CHECKPOINT_FILE = CHECKPOINT_DIR / "model_latest.bin"
ENGRAM_CHECKPOINT_FILE = CHECKPOINT_DIR / "engram_latest.bin"
GLOBAL_STATE_FILE = CHECKPOINT_DIR / "global_state.json"
DATASET_CACHE_FILE = DATA_DIR / "dataset_info.json"
USERS_DB_FILE = BASE_DIR / "users.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DELTA_TEMP_DIR = CHECKPOINT_DIR / "pending_deltas"
DELTA_TEMP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = BASE_DIR / "pending_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHUNK_BYTES = 80 * 1024 * 1024
MAX_DIRECT_BODY_BYTES = 90 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 3 * 1024 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_CHUNKS = 512
UPLOAD_TTL_SECONDS = 6 * 60 * 60

import socket
hostname = socket.gethostname()
try: local_ip = socket.gethostbyname(hostname)
except Exception: local_ip = "unknown"
print(f"🌐 Container hostname: {hostname}")
print(f"🌐 Container local IP: {local_ip}")
print(f"🌐 Listening on: 0.0.0.0:{PORT}")
print("=" * 60)

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = "Vxtzq/Crowd-v1"
DATASET_REPO_ID = "Vxtzq/CrowdGPT"
HF_UPLOAD_INTERVAL_HOURS = 2
HF_ENABLED = HF_AVAILABLE and bool(HF_TOKEN)
ADMIN_SECRET = "PUT A SECRET HERE"
os.environ.setdefault("CROWDGPT_TRUSTED_PROXIES", "127.0.0.1/32,::1/128")
HF_WEIGHT_FILES = ["model_bf16.safetensors", "engram_bf16.safetensors"]

# ============ MODEL ============
MODEL_CONFIG = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16,
    "nKvHeads": 4, "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560,
    "weightTying": True, "architecture": "SotaGPT",
    "features": {"groupedQueryAttention": True, "rotaryPositionEmbedding": True,
                 "swiGLU": True, "weightTying": True, "chunkedAttention": True},
}
LORA_CONFIG = {
    "FULL_RANK": MODEL_CONFIG["dim"],
    "RANK_TIERS": {"minimal": 32, "low": 64, "medium": 128, "high": 256,
                   "ultra": 512, "extreme": 1024, "max": 2048, "full": MODEL_CONFIG["dim"]},
    "DEFAULT_RANK": MODEL_CONFIG["dim"], "WEIGHT_BY_RANK": True, "LOG_EFFECTIVE_RANK": True,
}
DATASET_CONFIG = {
    "repoId": DATASET_REPO_ID, "totalChunks": 5482, "subChunkMB": 10,
    "tokensPerSample": MODEL_CONFIG["maxSeqLen"] + 1, "tokenBytes": 4,
}

def calculate_model_size(cfg: dict) -> int:
    size = cfg["vocabSize"] * cfg["dim"]
    for _ in range(cfg["nLayers"]):
        d = cfg["dim"]
        size += d * 2 + d * (cfg["nHeads"] * cfg["headDim"]) + d * (cfg["nKvHeads"] * cfg["headDim"]) * 2 + d * d + d * 2
        mlp_hidden = cfg.get("mlpHidden", 4 * d)
        size += d * mlp_hidden * 2 + mlp_hidden * d
    size += cfg["dim"] * 2
    if not cfg.get("weightTying", False): size += cfg["dim"] * cfg["vocabSize"]
    return size

MODEL_SIZE = calculate_model_size(MODEL_CONFIG)
log.info(f"🧮 SotaGPT size: {MODEL_SIZE:,} params (~{MODEL_SIZE*4/1024/1024:.2f} MB FP32)")
ENG_NUM_BUCKETS = 227865
ENG_PARAMS = ENG_NUM_BUCKETS * MODEL_CONFIG["dim"]
log.info(f"🧠 Engram size: {ENG_PARAMS:,} params (~{ENG_PARAMS*4/1024/1024:.2f} MB FP32)")

# ============ MEMORY GUARDS ============
def _avail_mem_gb() -> float:
    if _HAS_PSUTIL: return psutil.virtual_memory().available / (1024 ** 3)
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'): return int(line.split()[1]) / (1024 * 1024)
    except Exception: pass
    return 999.0

def _mem_guard(min_gb: float, context: str) -> bool:
    avail = _avail_mem_gb()
    if avail < min_gb:
        log.warning(f"⚠️ Memory guard: '{context}' needs ~{min_gb:.1f} GB, only {avail:.1f} GB free — SKIPPING")
        return False
    log.info(f"✅ Memory guard: '{context}' has {avail:.1f} GB free (need {min_gb:.1f} GB)")
    return True

# ============ QUANTIZATION ============
def float32_to_bf16_array(a): return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16).view(torch.uint16).numpy()
def float32_to_fp16_array(a): return np.clip(np.ascontiguousarray(a, dtype=np.float32), -65504.0, 65504.0).astype(np.float16).view(np.uint16)
def float32_to_fp8_array(a): return torch.from_numpy(a.astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
def compress_weights_bf16(w): return float32_to_bf16_array(w).tobytes()
def compress_weights_fp16(w): return float32_to_fp16_array(w).tobytes()
def compress_weights_fp8(w): return float32_to_fp8_array(w).tobytes()

def _bytes_per_param(fmt: str) -> int:
    if fmt in ("bf16", "fp16"): return 2
    if fmt == "fp8": return 1
    if fmt == "fp32": return 4
    raise HTTPException(status_code=400, detail=f"Unsupported weight format: {fmt}")

def _decode_delta_chunk(buf: bytearray, fmt: str) -> torch.Tensor:
    if fmt == "fp8": return torch.frombuffer(buf, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    if fmt == "fp16": return torch.frombuffer(buf, dtype=torch.uint16).view(torch.float16).to(torch.float32)
    if fmt == "fp32": return torch.frombuffer(buf, dtype=torch.float32)
    return torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16).to(torch.float32)

def _encode_float32_to_bf16_bytes(t: torch.Tensor) -> bytes: return t.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()

# ============ DISK-BASED DELTA STORAGE ============
def save_delta_to_disk(client_ip, delta, engram_indices, engram_values, metadata):
    ts = int(time.time() * 1000)
    safe_ip = client_ip.replace(".", "_").replace(":", "_")
    fname = DELTA_TEMP_DIR / f"delta_{safe_ip}_{ts}.bin"
    with open(fname, 'wb') as f:
        meta_bytes = json.dumps(metadata).encode('utf-8')
        f.write(struct.pack('<I', len(meta_bytes))); f.write(meta_bytes)
        chunk_size = 10_000_000
        for i in range(0, MODEL_SIZE, chunk_size):
            chunk = delta[i:i + chunk_size].cpu().view(torch.uint16).numpy()
            f.write(chunk.tobytes())
        if engram_indices is not None and len(engram_indices) > 0:
            f.write(struct.pack('<I', len(engram_indices)))
            f.write(engram_indices.tobytes())
            f.write(engram_values.cpu().view(torch.uint16).numpy().tobytes())
        else:
            f.write(struct.pack('<I', 0))
    return str(fname)

def load_delta_from_disk(filepath):
    with open(filepath, 'rb') as f:
        meta_len = struct.unpack('<I', f.read(4))[0]
        metadata = json.loads(f.read(meta_len).decode('utf-8'))
        delta = torch.zeros(MODEL_SIZE, dtype=torch.bfloat16)
        chunk_size = 10_000_000
        offset = 0; bytes_per_param = 2
        while offset < MODEL_SIZE:
            read_params = min(chunk_size, MODEL_SIZE - offset)
            chunk_bytes = f.read(read_params * bytes_per_param)
            if not chunk_bytes: break
            buf = bytearray(chunk_bytes)
            chunk_t = torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16).clone()
            delta[offset:offset + read_params] = chunk_t
            offset += read_params
        eng_count = struct.unpack('<I', f.read(4))[0]
        if eng_count > 0:
            idx_bytes = f.read(eng_count * 4)
            val_bytes = f.read(eng_count * MODEL_CONFIG["dim"] * 2)
            engram_indices = np.frombuffer(bytearray(idx_bytes), dtype=np.uint32)
            val_buf = bytearray(val_bytes)
            engram_values = torch.frombuffer(val_buf, dtype=torch.uint16).view(torch.bfloat16).reshape(-1, MODEL_CONFIG["dim"]).clone()
        else:
            engram_indices = None; engram_values = None
    return metadata, delta, engram_indices, engram_values

# ============ GLOBAL STATE ============
db = {"tasks": {}, "pendingSubmissions": {}, "shardMeta": {}}
active_clients: Dict[str, float] = {}
blacklisted_subnets: Dict[str, float] = {}
byzantine_flag_counts: Dict[str, int] = {}
client_hardware: Dict[str, dict] = {}
banned_users: Dict[str, dict] = {}
banned_client_keys: Dict[str, float] = {}

global_weights: Optional[torch.Tensor] = None
global_engram_weights: Optional[torch.Tensor] = None
global_step = 0
global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}

dataset_info = {
    "repoId": DATASET_CONFIG["repoId"], "totalChunks": 0, "validChunkIndices": [],
    "subChunkBytes": DATASET_CONFIG["subChunkMB"] * 1024 * 1024,
    "initialized": False, "lastChecked": None,
}
data_position = {"currentChunkIdx": 0, "epoch": 0, "totalSamplesSeen": 0,
    "permutation": [], "permutationPos": 0, "consumed_through_pos": -1}
current_data_shard = {"chunkIdx": 0, "assignedClients": [], "shardId": None,
    "createdAt": None, "round": None, "consumed": False}
dataset_stats = {"totalShardsProcessed": 0, "epochsCompleted": 0, "samplesProcessed": 0, "lastUpdate": None}

round_state = {"current_round": 1, "round_start_time": time.time(), "round_duration_hours": 4.0,
    "participating_clients": {}, "aggregations_this_round": 0,
    "last_aggregation_time": 0, "pending_deltas": [], "is_aggregating": False, "force_end_at": None}

FL_CONFIG = {
    "CONSENSUS_REQUIRED": 1, "CHECKPOINT_INTERVAL": 1,
    "MAX_CHECKPOINTS": int(os.getenv("CROWDGPT_MAX_CHECKPOINTS", "5")),
    "DELTA_NORM_BASE": 500.0, "DELTA_NORM_ABSOLUTE_MAX": 5000.0,
    "SERVER_LR": 0.5, "WEIGHT_DECAY": 0.01,
    "MAX_WEIGHT_NORM": 10000000000.0,
    "MAX_DELTA_APPLY_NORM": 1200.0,
    "LR_DECAY_STEPS": 50_000, "MIN_LR_MULT": 0.05,
    "IP_TIMEOUT_S": 900, "SHARD_TIMEOUT_S": 1800, "CLEANUP_INTERVAL_S": 60,
    "SHUFFLE_CHUNKS": True,
}
ROUND_CONFIG = {
    "MAX_ROUND_HOURS": 4.0, "TARGET_SUBMISSION_HOURS": 1.0,
    "MIN_CLIENTS_TO_AGGREGATE": 1, "CLIENT_TIMEOUT_HOURS": 2.5,
    "AGGREGATION_COOLDOWN_SEC": 120, "ULTIMATUM_THRESHOLD": 0.90, "ULTIMATUM_TIMEOUT_MIN": 30,
}

# ============ HARDENING ============
def _env_int(name, default):
    try: return int(os.getenv(name, str(default)))
    except Exception: return default
def _env_float(name, default):
    try: return float(os.getenv(name, str(default)))
    except Exception: return default
def _parse_cidrs(raw):
    nets = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part: continue
        try: nets.append(ipaddress.ip_network(part, strict=False))
        except Exception: pass
    return nets

HARDEN = {
    "MIN_TRAIN_TIME_S": _env_int("CROWDGPT_MIN_TRAIN_TIME_S", 900),
    "MIN_SUBMIT_INTERVAL_S": _env_int("CROWDGPT_MIN_SUBMIT_INTERVAL_S", 0),
    "TRUSTED_PROXY_NETS": _parse_cidrs(os.getenv("CROWDGPT_TRUSTED_PROXIES", "")),
    "MAX_SUBMISSIONS_PER_CLIENT_ROUND": _env_int("CROWDGPT_MAX_SUBMISSIONS_PER_CLIENT_ROUND", 0),
    "MAX_SUBMISSIONS_PER_SUBNET_ROUND": _env_int("CROWDGPT_MAX_SUBMISSIONS_PER_SUBNET_ROUND", 500),
    "MAX_LOCAL_STEPS": _env_int("CROWDGPT_MAX_LOCAL_STEPS", 200_000),
    "MAX_TOKENS_PER_STEP": _env_int("CROWDGPT_MAX_TOKENS_PER_STEP", int(128 * DATASET_CONFIG["tokensPerSample"] * 1.25)),
    "MAX_TOKENS_PER_SEC": _env_float("CROWDGPT_MAX_TOKENS_PER_SEC", 1_000_000.0),
    "REQUIRE_TASK_TICKET": os.getenv("CROWDGPT_REQUIRE_TASK_TICKET", "0") == "1",
    "ENGRAM_ROW_NORM_MAX": _env_float("CROWDGPT_ENGRAM_ROW_NORM_MAX", 8.0),
    "ENGRAM_FULL_TRUST_WEIGHT": _env_float("CROWDGPT_ENGRAM_FULL_TRUST_WEIGHT", 0.2),
    "KRUM_F": _env_int("CROWDGPT_KRUM_F", 1),
    "KRUM_RAM_BYTES": _env_int("CROWDGPT_KRUM_RAM_BYTES", 512 * 1024 * 1024),
    "TOKEN_CAP_FACTOR": _env_float("CROWDGPT_TOKEN_CAP_FACTOR", 3.0),
    "DUP_REL_DIST": _env_float("CROWDGPT_DUP_REL_DIST", 0.02),
    "MIN_FREE_DISK_TASK_BYTES": _env_int("CROWDGPT_MIN_FREE_DISK_TASK_BYTES", 10 * 1024**3),
    "MIN_FREE_DISK_UPLOAD_BYTES": _env_int("CROWDGPT_MIN_FREE_DISK_UPLOAD_BYTES", 2 * 1024**3),
    "MAX_CONCURRENT_UPLOADS": _env_int("CROWDGPT_MAX_CONCURRENT_UPLOADS", 2),
    "UPLOAD_SLOT_WAIT_S": _env_int("CROWDGPT_UPLOAD_SLOT_WAIT_S", 800),
    "MAX_PENDING_DELTAS": _env_int("CROWDGPT_MAX_PENDING_DELTAS", 0),
    "LOSS_CHECK_MODE": os.getenv("CROWDGPT_LOSS_CHECK_MODE", "off").strip().lower(),
    "LOSS_CHECK_URL": os.getenv("CROWDGPT_LOSS_CHECK_URL", "").strip(),
    "LOSS_CHECK_TOKEN": os.getenv("CROWDGPT_LOSS_CHECK_TOKEN", "").strip(),
    "LOSS_TOLERANCE": _env_float("CROWDGPT_LOSS_TOLERANCE", 0.0),
    "LOSS_FAIL_OPEN": os.getenv("CROWDGPT_LOSS_FAIL_OPEN", "1") == "1",
    "LOSS_TIMEOUT_S": _env_int("CROWDGPT_LOSS_TIMEOUT_S", 1800),
    "PUBLIC_URL": os.getenv("CROWDGPT_PUBLIC_URL", "https://server.crowdgpt.net").rstrip("/"),
}
if HARDEN["LOSS_CHECK_MODE"] not in ("off", "log", "enforce"): HARDEN["LOSS_CHECK_MODE"] = "off"
ADMIN_SECRET = os.getenv("CROWDGPT_ADMIN_SECRET") or ADMIN_SECRET

def _admin_ok(secret: str) -> bool:
    return bool(secret) and secrets.compare_digest(str(secret).encode(), str(ADMIN_SECRET).encode())

# ============ CROSS-CHECK CONFIG ============
CROSS_CHECK = {
    "ENABLED": os.getenv("CROWDGPT_CROSS_CHECK_ENABLED", "1") == "1",
    "COHORT_SIZE": _env_int("CROWDGPT_COHORT_SIZE", 3),
    "MAX_OVERFLOW": _env_int("CROWDGPT_COHORT_MAX_OVERFLOW", 1),
    "TARGET_BATCH_SIZE": _env_int("CROWDGPT_TARGET_BATCH_SIZE", 2),
    "TARGET_ACCUM_STEPS": _env_int("CROWDGPT_TARGET_ACCUM_STEPS", 32),
    "TARGET_LOCAL_STEPS": _env_int("CROWDGPT_TARGET_LOCAL_STEPS", 500),
    "TARGET_AUTOCAST": os.getenv("CROWDGPT_TARGET_AUTOCAST", "any").strip().lower(),
    "TARGET_OPTIMIZER": os.getenv("CROWDGPT_TARGET_OPTIMIZER", "any").strip().lower(),
    # v2.0: threshold raised to 5.0 — cross-arch tolerated, same backend/prec/optim
    "DISTANCE_THRESHOLD_RATIO": _env_float("CROWDGPT_CROSS_CHECK_THRESHOLD", 5.0),
    "BAN_THRESHOLD": _env_int("CROWDGPT_BYZANTINE_BAN_THRESHOLD", 3),
}
if CROSS_CHECK["COHORT_SIZE"] < 2: CROSS_CHECK["COHORT_SIZE"] = 3
if CROSS_CHECK["MAX_OVERFLOW"] < 0: CROSS_CHECK["MAX_OVERFLOW"] = 0
if CROSS_CHECK["MAX_OVERFLOW"] > 2: CROSS_CHECK["MAX_OVERFLOW"] = 2
if CROSS_CHECK["TARGET_BATCH_SIZE"] < 0: CROSS_CHECK["TARGET_BATCH_SIZE"] = 0
if CROSS_CHECK["DISTANCE_THRESHOLD_RATIO"] < 1.01: CROSS_CHECK["DISTANCE_THRESHOLD_RATIO"] = 1.01
if CROSS_CHECK["TARGET_AUTOCAST"] not in ("bf16", "fp32", "any"): CROSS_CHECK["TARGET_AUTOCAST"] = "bf16"
if CROSS_CHECK["TARGET_OPTIMIZER"] not in ("adamw_fp32", "adamw8bit", "any"):
    CROSS_CHECK["TARGET_OPTIMIZER"] = "any"

# ============ MATCHMAKING CONFIG ============
MATCHMAKING = {
    "POOL_SIZE":         _env_int("CROWDGPT_POOL_SIZE", 3),
    "POOL_TIMEOUT_S":    _env_int("CROWDGPT_POOL_TIMEOUT_S", 4 * 60 * 60),
    "FORMING_TIMEOUT_S": _env_int("CROWDGPT_FORMING_TIMEOUT_S", 30 * 60),
    "HW_BS_TOLERANCE":   _env_float("CROWDGPT_HW_BS_TOLERANCE", 1.0),
    # v2.2: solo BF16 pools dissolve after BF16_SOLO_DISSOLVE_S.
    "BF16_SOLO_DISSOLVE_S": _env_int("CROWDGPT_BF16_SOLO_DISSOLVE_S", 30 * 60),
    # v2.2: solo FP32 pools dissolve after FP32_SOLO_DISSOLVE_S.
    # Default = 60 min so bf16 clients in solo bf16 pools (dissolving at 30 min)
    # have time to free up and join a still-forming fp32 pool before it dissolves.
    # Prevents the pathological loop where 1 fp32 + 1 bf16 client both dissolve
    # at the same time, both retry, both create solo pools, repeat forever.
    "FP32_SOLO_DISSOLVE_S": _env_int("CROWDGPT_FP32_SOLO_DISSOLVE_S", 60 * 60),
}
if MATCHMAKING["POOL_SIZE"] < 2: MATCHMAKING["POOL_SIZE"] = 2
if MATCHMAKING["BF16_SOLO_DISSOLVE_S"] < 60: MATCHMAKING["BF16_SOLO_DISSOLVE_S"] = 60
if MATCHMAKING["FP32_SOLO_DISSOLVE_S"] < 60: MATCHMAKING["FP32_SOLO_DISSOLVE_S"] = 60
# Sanity: fp32 solo must outlast bf16 solo, otherwise the mixed-client
# infinite-loop case comes back. If someone configured them backwards,
# bump FP32 to BF16 + FORMING_TIMEOUT_S.
if MATCHMAKING["FP32_SOLO_DISSOLVE_S"] <= MATCHMAKING["BF16_SOLO_DISSOLVE_S"]:
    MATCHMAKING["FP32_SOLO_DISSOLVE_S"] = (
        MATCHMAKING["BF16_SOLO_DISSOLVE_S"] + MATCHMAKING["FORMING_TIMEOUT_S"]
    )

def _cohort_seed(shard_id: str) -> int:
    return int(hashlib.sha256(f"crowdgpt_cohort:{shard_id}".encode()).hexdigest()[:8], 16) % (2**31)

task_tickets: Dict[str, dict] = {}
submit_counts_client: Dict[tuple, int] = {}
submit_counts_subnet: Dict[tuple, int] = {}
client_last_submit: Dict[str, float] = {}
upload_slots = threading.BoundedSemaphore(max(1, HARDEN["MAX_CONCURRENT_UPLOADS"]))
_commit_lock = threading.Lock()
shard_lock = threading.RLock()
_cross_check_lock = threading.Lock()

pools: Dict[str, dict] = {}
client_pool_map: Dict[str, str] = {}
ready_pools: List[dict] = []
pool_lock = threading.RLock()

# ============ AUTH ============
users_db = {"users": {}, "tokens": {}}
users_lock = threading.Lock()
stats_lock = threading.Lock()
BAD_WORDS = ["PUT BAD WORDS HERE TO PREVENT BAD USERNAMES"]

def load_users_db():
    global users_db
    try:
        if USERS_DB_FILE.exists():
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f: users_db = json.load(f)
            log.info(f"👥 Loaded {len(users_db['users'])} users from disk")
    except Exception as e:
        log.error(f"❌ load_users_db: {e}"); users_db = {"users": {}, "tokens": {}}

def save_users_db():
    try:
        with open(USERS_DB_FILE, 'w', encoding='utf-8') as f: json.dump(users_db, f, indent=2)
    except Exception as e: log.error(f"❌ save_users_db: {e}")

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
def generate_token() -> str: return secrets.token_urlsafe(32)
def contains_bad_word(text: str) -> bool:
    if not text: return False
    lower = text.lower()
    normalized = lower.translate(str.maketrans("013@$", "oieas"))
    for word in BAD_WORDS:
        if word in lower or word in normalized: return True
    return any(c in text for c in ['\n', '\r', '\t'])

def _is_user_banned(user_uuid: str) -> bool:
    b = banned_users.get(user_uuid)
    if not b: return False
    exp = b.get("until", 0)
    if exp and time.time() > exp:
        banned_users.pop(user_uuid, None); return False
    return True
def _is_client_banned(client_key: str) -> bool:
    if client_key not in banned_client_keys: return False
    exp = banned_client_keys[client_key]
    if exp and time.time() > exp:
        banned_client_keys.pop(client_key, None); return False
    return True

def get_current_user(request: Request) -> Optional[dict]:
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "): return None
    token = auth_header[7:].strip()
    with users_lock:
        user_uuid = users_db["tokens"].get(token)
        if not user_uuid: return None
        if _is_user_banned(user_uuid): return None
        return users_db["users"].get(user_uuid)

def compute_trust_score(stats: dict) -> float:
    contributions = stats.get("contributions", 0)
    flags = stats.get("byzantine_flags", 0)
    base = min(1.0, 0.5 + (contributions / 200.0))
    penalty = min(0.5, flags * 0.1)
    return round(max(0.0, base - penalty), 2)

def compute_global_rank(user_uuid: str) -> int:
    with users_lock:
        sorted_users = sorted(users_db["users"].values(), key=lambda u: u["stats"].get("totalTokens", 0), reverse=True)
        for i, u in enumerate(sorted_users):
            if u["uuid"] == user_uuid: return i + 1
    return 0

# ============ CLIENT IDENTITY ============
def _peer_in_trusted(request: Request) -> bool:
    peer = request.client.host if request.client else ""
    try: a = ipaddress.ip_address(peer)
    except Exception: return False
    return any(a in n for n in HARDEN["TRUSTED_PROXY_NETS"])

def get_client_ip(request: Request) -> str:
    nets = HARDEN["TRUSTED_PROXY_NETS"]
    if not nets:
        cf = request.headers.get("cf-connecting-ip")
        if cf: return cf.strip()
        fwd = request.headers.get("x-forwarded-for")
        if fwd: return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    if _peer_in_trusted(request):
        cf = request.headers.get("cf-connecting-ip")
        if cf: return cf.strip()
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            for hop in reversed([h.strip() for h in fwd.split(",") if h.strip()]):
                try:
                    if not any(ipaddress.ip_address(hop) in n for n in nets): return hop
                except Exception: continue
    return request.client.host if request.client else "unknown"

# ============ v2.4: MULTI-DEVICE ============
# A single account can run training on multiple devices. Each device sends a
# stable X-Client-Device header (persisted client-side as device_id.json). The
# server keys every per-client dict on "user:<uuid>:dev:<device>" so two
# devices are treated as distinct training clients (separate pools, separate
# delta slots, separate quotas, separate hardware sigs). User-level operations
# (bans, flags, trust credit, token accounting) still key on the bare UUID.
_DEVICE_ID_RE = re.compile(r'[^A-Za-z0-9_-]')

def _sanitize_device_id(raw: Optional[str]) -> Optional[str]:
    if not raw: return None
    s = _DEVICE_ID_RE.sub('', str(raw))[:32]
    return s or None

def _user_id_from_key(client_key: str) -> Optional[str]:
    """Extract user UUID from 'user:<uuid>' or 'user:<uuid>:dev:<device>'.
    Returns None for 'ip:<ip>' or anything else."""
    if not client_key or not client_key.startswith("user:"):
        return None
    rest = client_key[5:]
    if ":dev:" in rest:
        return rest.split(":dev:", 1)[0] or None
    return rest or None

def get_client_key(request: Request) -> str:
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        with users_lock:
            user_uuid = users_db["tokens"].get(token)
            if user_uuid:
                dev = _sanitize_device_id(request.headers.get("X-Client-Device"))
                return f"user:{user_uuid}:dev:{dev}" if dev else f"user:{user_uuid}"
    # Anonymous clients stay ip-keyed — no device suffix, keeps NAT ambiguity intact
    return f"ip:{get_client_ip(request)}"

def _banned_reason(request: Request) -> Optional[str]:
    ck = get_client_key(request)
    if _is_client_banned(ck): return f"client banned ({ck})"
    uid = _user_id_from_key(ck)
    if uid and _is_user_banned(uid): return f"account suspended ({uid})"
    if is_blacklisted(get_client_ip(request)): return "subnet temporarily blacklisted"
    return None

# ============ MINIMUM CLIENT VERSION ============
MIN_CLIENT_VERSION_FILE = CHECKPOINT_DIR / "min_client_version.json"
_min_client_version = ""
_block_unversioned = os.getenv("CROWDGPT_BLOCK_UNVERSIONED", "0") == "1"

def _vtuple(v) -> tuple:
    try: return tuple(int(x) for x in str(v).strip().split("-")[0].split(".")[:4])
    except Exception: return ()
def _load_min_client_version() -> str:
    try: return str(json.loads(MIN_CLIENT_VERSION_FILE.read_text()).get("min", "")) if MIN_CLIENT_VERSION_FILE.exists() else ""
    except Exception: return ""
def _set_min_client_version(v: str) -> None:
    global _min_client_version
    _min_client_version = v
    tmp = MIN_CLIENT_VERSION_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"min": v, "set_at": time.time()})); os.replace(tmp, MIN_CLIENT_VERSION_FILE)
_min_client_version = _load_min_client_version()

def _version_gate(request: Request):
    if not _min_client_version: return None
    hdr = request.headers.get("X-Client-Version")
    if hdr is None:
        if not _block_unversioned: return None
        cv = (0,)
    else:
        cv = _vtuple(hdr)
        if not cv: cv = (0,)
    if cv >= _vtuple(_min_client_version): return None
    return Response(content=json.dumps({"status": "update_required", "minVersion": _min_client_version,
                                        "message": f"Client {hdr or 'unknown'} is too old; update to {_min_client_version}+"}).encode(),
                    status_code=426, media_type="application/json", headers={"X-Status": "update"})

# ============ SUBNET HELPERS ============
def get_subnet(ip: str) -> str:
    if not ip or ip == "unknown": return "unknown"
    parts = ip.split(".")
    if len(parts) == 4: return ".".join(parts[:3]) + ".0/24"
    return ip
def is_blacklisted(ip: str) -> bool:
    subnet = get_subnet(ip)
    if subnet in blacklisted_subnets:
        if time.time() < blacklisted_subnets[subnet]: return True
        del blacklisted_subnets[subnet]
    return False
def ban_subnet(ip: str, reason: str, hours: int = 24):
    subnet = get_subnet(ip)
    blacklisted_subnets[subnet] = time.time() + hours * 3600
    log.warning(f"🔨 BANNED {subnet} for {hours}h — {reason}")

def _flag_byzantine(client_key: str, client_ip: str, reason: str):
    log.warning(f"🚩 BYZANTINE FLAG: {client_key} ({client_ip}) — {reason}")
    with users_lock:
        uid = _user_id_from_key(client_key)
        if uid and uid in users_db["users"]:
            u = users_db["users"][uid]
            u["stats"]["byzantine_flags"] = u["stats"].get("byzantine_flags", 0) + 1
            u["stats"]["last_flag"] = datetime.now(timezone.utc).isoformat()
            u["stats"]["last_flag_reason"] = reason
            save_users_db()
    if client_ip and client_ip != "unknown":
        subnet = get_subnet(client_ip)
        byzantine_flag_counts[subnet] = byzantine_flag_counts.get(subnet, 0) + 1
        if byzantine_flag_counts[subnet] >= CROSS_CHECK["BAN_THRESHOLD"]:
            ban_subnet(client_ip, f"subnet has {byzantine_flag_counts[subnet]} byzantine flags")

def _flag_byzantine_from_delta(path: str, reason: str):
    try:
        with open(path, 'rb') as f:
            ml = struct.unpack('<I', f.read(4))[0]
            md = json.loads(f.read(ml).decode('utf-8'))
        _flag_byzantine(md.get("nodeKey", "unknown"), md.get("nodeIp", "unknown"), reason)
    except Exception as e: log.warning(f"Could not flag delta {path}: {e}")

# ============ DATASET STATE ============
def save_dataset_state():
    try:
        with open(DATASET_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({"datasetInfo": dataset_info, "dataPosition": data_position, "datasetStats": dataset_stats}, f, indent=2)
    except Exception as e: log.error(f"❌ save_dataset_state: {e}")

def load_dataset_state():
    global dataset_info, data_position, dataset_stats
    try:
        if DATASET_CACHE_FILE.exists():
            with open(DATASET_CACHE_FILE, 'r', encoding='utf-8') as f: state = json.load(f)
            dataset_info.update(state.get("datasetInfo", {}))
            data_position.update(state.get("dataPosition", {}))
            dataset_stats.update(state.get("datasetStats", {}))
            log.info(f"📂 dataset state: epoch={data_position.get('epoch')} pos={data_position.get('permutationPos')} chunk={data_position.get('currentChunkIdx')}")
    except Exception as e: log.error(f"❌ load_dataset_state: {e}")

# ============ HF CHUNK DISCOVERY ============
def _discover_hf_chunk_indices() -> Optional[List[int]]:
    if not HF_ENABLED: return None
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(DATASET_REPO_ID, repo_type="dataset")
        indices = set()
        for f in files:
            m = re.search(r'(?:^|/)chunk_(\d+)\.bin$', f)
            if m: indices.add(int(m.group(1)))
        if indices: return sorted(indices)
        log.warning(f"⚠️ No chunk_*.bin files in HF dataset {DATASET_REPO_ID}")
        return None
    except Exception as e:
        log.warning(f"⚠️ Could not query HF dataset: {e}"); return None

def _discover_local_chunk_indices() -> Optional[List[int]]:
    chunks_dir = BASE_DIR / "chunks"
    if not chunks_dir.exists(): return None
    indices = []
    for f in chunks_dir.iterdir():
        m = re.match(r'^chunk_(\d+)\.bin$', f.name)
        if m: indices.append(int(m.group(1)))
    return sorted(indices) if indices else None

def _reconcile_chunk_indices():
    hf_indices = _discover_hf_chunk_indices()
    local_indices = _discover_local_chunk_indices()
    old_total = int(dataset_info.get("totalChunks", 0) or 0)
    old_indices = dataset_info.get("validChunkIndices")
    if hf_indices is not None: source, indices = "HF", hf_indices
    elif local_indices is not None: source, indices = "local", local_indices
    else:
        source = "config"; fallback_total = int(DATASET_CONFIG.get("totalChunks", 100))
        indices = list(range(fallback_total))
        log.warning(f"⚠️ Chunk discovery failed; fallback to config range(0..{fallback_total-1})")
    dataset_info["validChunkIndices"] = indices
    dataset_info["totalChunks"] = len(indices)
    changed = old_total != len(indices) or (old_indices is not None and old_indices != indices)
    if changed:
        log.warning(f"🔧 Chunk indices reconciled ({source}): {old_total} → {len(indices)}")
        data_position["permutation"] = []; data_position["permutationPos"] = 0
        data_position["consumed_through_pos"] = -1
        data_position["currentChunkIdx"] = indices[0]
    else: log.info(f"✅ Chunk indices verified ({source}): count={len(indices)}")
    dataset_info["initialized"] = True
    dataset_info["lastChecked"] = datetime.now(timezone.utc).isoformat()
    save_dataset_state()

def initialize_dataset(): _reconcile_chunk_indices()

# ============ CHUNK SCHEDULER ============
def _generate_epoch_permutation(n: int) -> List[int]:
    valid = dataset_info.get("validChunkIndices") or []
    perm = list(valid) if valid else list(range(n))
    if FL_CONFIG.get("SHUFFLE_CHUNKS", True): random.shuffle(perm)
    return perm

def _ensure_permutation() -> bool:
    valid = dataset_info.get("validChunkIndices") or []
    expected_len = len(valid) if valid else dataset_info.get("totalChunks", 0)
    if expected_len <= 0: return False
    perm = data_position.get("permutation") or []
    needs_regen = len(perm) != expected_len
    if not needs_regen and valid:
        valid_set = set(valid)
        needs_regen = any(idx not in valid_set for idx in perm)
    if not needs_regen: return False
    data_position["permutation"] = _generate_epoch_permutation(expected_len)
    data_position["permutationPos"] = 0
    data_position["consumed_through_pos"] = -1
    data_position["currentChunkIdx"] = data_position["permutation"][0]
    log.info(f"🎲 New epoch permutation ({expected_len} chunks) → first chunk {data_position['currentChunkIdx']}")
    return True

def _advance_chunk_pointer():
    global data_position, dataset_stats
    if _ensure_permutation():
        dataset_stats["totalShardsProcessed"] += 1
        dataset_stats["lastUpdate"] = datetime.now(timezone.utc).isoformat()
        return
    total = len(data_position.get("permutation", []))
    if total <= 0: return
    data_position["permutationPos"] += 1
    if data_position["permutationPos"] >= total:
        data_position["epoch"] += 1
        dataset_stats["epochsCompleted"] += 1
        data_position["permutation"] = _generate_epoch_permutation(total)
        data_position["permutationPos"] = 0
        data_position["consumed_through_pos"] = -1
        log.info(f"🔄 epoch {data_position['epoch']} started")
    data_position["currentChunkIdx"] = data_position["permutation"][data_position["permutationPos"]]
    dataset_stats["totalShardsProcessed"] += 1
    dataset_stats["lastUpdate"] = datetime.now(timezone.utc).isoformat()

def _crash_recovery():
    perm = data_position.get("permutation", [])
    if not perm: return
    pos = data_position.get("permutationPos", 0)
    consumed = data_position.get("consumed_through_pos", -1)
    if consumed < pos: return
    old_pos = pos
    data_position["permutationPos"] = pos + 1
    if data_position["permutationPos"] >= len(perm):
        data_position["epoch"] += 1
        data_position["permutation"] = _generate_epoch_permutation(len(perm))
        data_position["permutationPos"] = 0
        data_position["consumed_through_pos"] = -1
    data_position["currentChunkIdx"] = data_position["permutation"][data_position["permutationPos"]]
    log.info(f"🔄 Crash recovery: pos {old_pos} → {data_position['permutationPos']}")
    save_dataset_state()

def _active_client_count() -> int:
    now = time.time()
    return sum(1 for ts in active_clients.values() if now - ts < FL_CONFIG["IP_TIMEOUT_S"])

# ============ GPU CLASSIFICATION ============
def _gpu_arch_class(gpu_name: str) -> str:
    g = (gpu_name or "").lower().strip()
    if not g: return "unknown"
    if any(x in g for x in ("gb10","gb200","b100","b200","dgx spark","grace blackwell")): return "blackwell_dc"
    if any(x in g for x in ("5090","5080","5070","5060","5050","rtx 50")): return "blackwell_rtx"
    if any(x in g for x in ("h100","h200","gh200","hopper")): return "hopper"
    if any(x in g for x in ("4090","4080","4070","4060","4050","rtx 40","l4 ","l40","rtx 6000 ada")): return "ada"
    if any(x in g for x in ("3090","3080","3070","3060","3050","rtx 30","a100","a6000","a5000","a4000","a10g","a10","a30","a40")): return "ampere"
    if any(x in g for x in ("2080","2070","2060","rtx 20","t4","titan rtx","quadro rtx")): return "turing"
    if any(x in g for x in ("1080","1070","1060","1050","titan x","titan v")): return "pascal"
    if "v100" in g: return "volta"
    if any(x in g for x in ("radeon","mi250","mi300","mi210","mi100")): return "rocm"
    return "unknown"

def _gpu_card_class(gpu_name: str) -> str:
    g = (gpu_name or "").lower().strip()
    if not g: return "unknown"
    if any(x in g for x in ("gb10","dgx spark","grace blackwell")): return "gb10"
    for x in ("5090","5080","5070","5060","5050","h100","h200","gh200",
              "4090","4080","4070","4060","4050","3090","3080","3070","3060","3050",
              "2080","2070","2060","1660","1650","1080","1070","1060","1050"):
        if x in g: return x
    if "l40" in g: return "l40"
    if "l4 " in g or g.endswith(" l4"): return "l4"
    for x in ("a100","a6000","a5000","a4000","a10g","a10","a30","a40"):
        if x in g: return x
    if "t4" in g: return "t4"
    if "v100" in g: return "v100"
    return _gpu_arch_class(gpu_name)

def _force_fp32_arch(gpu_name: str) -> bool:
    a = _gpu_arch_class(gpu_name)
    return a in ("pascal", "turing", "volta", "hopper", "blackwell_dc", "unknown")

# ============ MATCHMAKING (STRICT, FP32-FIRST, v2.3) ============
def _hw_signature(client_key: str) -> tuple:
    hw = client_hardware.get(client_key) or {}
    backend  = (hw.get("backend")  or "?").upper()
    bs       = int(hw.get("bs") or 0)
    bucket   = (1 << (bs.bit_length() - 1)) if bs > 0 else 0
    autocast = (hw.get("autocast") or "any").lower()
    optim    = (hw.get("optim")    or "any").lower()
    gpu      = hw.get("gpu") or ""
    return (backend, bucket, autocast, optim, _gpu_arch_class(gpu), _gpu_card_class(gpu))

def _pools_compatible(client_sig: tuple, pool_sig: tuple) -> bool:
    """v2.3 ASYMMETRIC matching — backend exact, optimizer normalize+wildcard,
    autocast asymmetric (FP32 pool accepts anyone; BF16 pool requires bf16).

    Optimizer aliases:
      adamw32 / adamw_fp32 / adamw / fp32  → adamw_fp32
      adamw8 / adamw_8bit / 8bit           → adamw8bit
    fp32-AdamW and 8bit-AdamW are allowed to mix in the same pool; 'any'
    stays a wildcard on either side. Real divergence is caught by the
    mandatory cross-check (5× threshold).

    Autocast rule (asymmetric):
      * FP32 pool: accepts bf16 OR fp32 clients (joiner trains in fp32 via
        task-payload crossCheck.targetAutocast).
      * BF16 pool: accepts only bf16-native clients.
      * Other / any: require exact autocast match.

    Arch is intentionally ignored (cross-arch tolerated at 5× threshold)."""
    if len(client_sig) < 4 or len(pool_sig) < 4: return False
    if client_sig[0] != pool_sig[0]: return False  # backend exact

    def _norm_opt(s):
        s = str(s or "any").lower()
        if s in ("adamw32", "adamw_fp32", "adamw", "fp32"): return "adamw_fp32"
        if s in ("adamw8", "adamw_8bit", "8bit"):           return "adamw8bit"
        return s
    c_opt = _norm_opt(client_sig[3])
    p_opt = _norm_opt(pool_sig[3])
    if c_opt != "any" and p_opt != "any" and c_opt != p_opt:
        if {c_opt, p_opt} != {"adamw_fp32", "adamw8bit"}:
            return False

    pool_ac   = str((pool_sig[2]   if len(pool_sig)   > 2 else "any") or "any").lower()
    client_ac = str((client_sig[2] if len(client_sig) > 2 else "any") or "any").lower()
    if pool_ac == "fp32": return True
    if pool_ac == "bf16": return client_ac == "bf16"
    return pool_ac == client_ac

def _autocast_rank(pool_or_sig) -> int:
    """Rank autocast precision for matchmaking priority.
    Lower = higher priority. FP32 (0) beats BF16 (1) beats any/other (2)."""
    try:
        if isinstance(pool_or_sig, dict):
            sig = pool_or_sig.get("hardware_sig") or ()
        else:
            sig = pool_or_sig
        ac = (sig[2] if len(sig) > 2 else "any") or "any"
        ac = str(ac).lower()
    except Exception:
        ac = "any"
    if ac == "fp32": return 0
    if ac == "bf16": return 1
    return 2

def _pool_target_autocast(pool: dict) -> str:
    """Effective training precision for the WHOLE pool.

    Use the pool's creation-time autocast (hardware_sig[2]). A bf16 client
    who later joins an fp32 pool is forced to fp32 via the task payload,
    and _pool_target_autocast must reflect that."""
    pool_ac = str((pool["hardware_sig"][2] if len(pool["hardware_sig"]) > 2 else "any") or "any").lower()
    if pool_ac in ("fp32", "bf16"): return pool_ac
    # Fallback for "any" / "other": infer from members.
    precisions: set = set()
    for c in pool["clients"]:
        hw = client_hardware.get(c) or {}
        ac = (hw.get("autocast") or "any").lower()
        if ac in ("bf16", "fp32"): precisions.add(ac)
    if len(precisions) == 1: return precisions.pop()
    return "any"

def _activate_pool(pool: dict):
    global current_data_shard
    with shard_lock:
        if not dataset_info["initialized"]: initialize_dataset()
        _ensure_permutation()
        chunk_idx = data_position["currentChunkIdx"]
        _advance_chunk_pointer()
        save_dataset_state()
    pool_id = pool["pool_id"]
    task_id = f"round_{round_state['current_round']}_{pool_id}"
    target_seed = _cohort_seed(pool_id)
    BASE_STEPS = 500
    slowest_client = min(pool["clients"], key=lambda c: client_hardware.get(c, {}).get("vram_gb", 100.0))
    hw = client_hardware.get(slowest_client, {})
    vram_gb = hw.get("vram_gb", 8.0)
    client_bs = max(1, hw.get("bs", 1))
    hw_multiplier = max(0.25, vram_gb / 8.0)
    scaled_steps = int(BASE_STEPS * hw_multiplier)
    target_steps = max(100, min(2500, scaled_steps))
    pool_autocast = _pool_target_autocast(pool)
    pool["target_autocast"] = pool_autocast
    log.info(f"⚙️ Pool {pool_id} pacing: Slowest VRAM={vram_gb:.1f}GB → "
             f"targetLocalSteps={target_steps}; targetAutocast={pool_autocast}")
    pool.update({"status": "training", "chunk_idx": chunk_idx, "task_id": task_id,
        "target_seed": target_seed, "activated_at": time.time(), "target_steps": target_steps})
    db["tasks"][task_id] = {
        "poolId": pool_id, "shardId": pool_id, "chunkIdx": chunk_idx,
        "round": round_state["current_round"], "createdAt": int(time.time() * 1000),
        "poolClients": list(pool["clients"]), "targetSeed": target_seed,
        "targetLocalSteps": target_steps, "targetBatchSize": CROSS_CHECK["TARGET_BATCH_SIZE"],
        "targetAccumSteps": CROSS_CHECK["TARGET_ACCUM_STEPS"],
        "targetAutocast": pool_autocast,
        "targetOptimizer": CROSS_CHECK["TARGET_OPTIMIZER"],
    }
    with shard_lock:
        current_data_shard.update({"chunkIdx": chunk_idx, "shardId": pool_id,
            "assignedClients": list(pool["clients"]), "createdAt": int(time.time() * 1000),
            "round": round_state["current_round"], "consumed": False})
    log.info(f"🎯 Pool {pool_id} activated → chunk {chunk_idx}, {len(pool['clients'])} clients, "
             f"seed {target_seed}, steps={target_steps}, autocast={pool_autocast}")

# ============ POWER USERS ============
# Admin-designated accounts (see /api/admin/power_user).  They behave exactly like everybody else when a pool is
# available, but never have to wait for peers: with no suitable pool they train SOLO immediately and can submit
# back-to-back.  Solo deltas skip the cohort cross-check (there is nobody to compare with), so they are fenced:
#   * only logged-in accounts can be power users (never anonymous ip: clients),
#   * the usual submit-time checks still apply (finite, norm clamp, token clamp, ban/quota),
#   * the aggregation re-checks norms, clips, and the loss ratchet can still roll everything back,
#   * ALL solo power deltas of a round are merged into ONE pseudo-pool, so however often a power user submits,
#     the group counts once, scaled by POWER_SOLO_WEIGHT (default 0.5 = half a cross-checked pool).
POWER = {
    "SOLO_WEIGHT": min(1.0, max(0.05, _env_float("CROWDGPT_POWER_SOLO_WEIGHT", 0.5))),
    "SOLO_ENABLED": os.getenv("CROWDGPT_POWER_SOLO_ENABLED", "1") == "1",
    # 1 (default): a power user joins ANY compatible forming pool that already has someone in it.
    # 0: only join a pool that their arrival completes; otherwise train solo right away (never waits).
    "JOIN_ANY_POOL": os.getenv("CROWDGPT_POWER_JOIN_ANY_POOL", "1") == "1",
}
POWER_POOL_ID = "power_solo"

def _is_power_client(client_key: str) -> bool:
    uid = _user_id_from_key(client_key)
    if not uid: return False
    with users_lock:
        u = users_db["users"].get(uid)
        return bool(u and u.get("power"))

def _power_solo_start(client_key: str, pool: dict) -> dict:
    """Called under pool_lock. A power user alone in a forming pool starts training right now."""
    if (POWER["SOLO_ENABLED"] and pool["status"] == "forming"
            and pool["clients"] == [client_key] and pools.get(pool["pool_id"]) is pool):
        pool["power"] = True
        log.info(f"⚡ {client_key}: power user, no pool available → training solo in {pool['pool_id']}")
        _activate_pool(pool)
    return pool

def _run_power_solo_check(pool: dict) -> Optional[list]:
    """Solo pool of a power user: no cohort to compare with, so only sanity checks."""
    pool_id = pool["pool_id"]
    items = list(pool["deltas"].items())
    survivors = []
    for ck, path in items:
        if _compute_norm_server(path, MODEL_SIZE) <= 0:      # 0.0 == unreadable / NaN / Inf
            log.warning(f"🚫 Power solo {ck}: delta unreadable or non-finite — discarded")
            try: os.remove(path)
            except Exception: pass
            continue
        survivors.append((ck, path))
        uid = _user_id_from_key(ck)
        if uid:
            with users_lock:
                u = users_db["users"].get(uid)
                if u:
                    u["stats"]["contributions"] = u["stats"].get("contributions", 0) + 1
                    save_users_db()
    if not survivors: return None
    log.info(f"⚡ Pool {pool_id}: power solo delta accepted (no cohort cross-check)")
    return [(p, 1.0) for _, p in survivors]

def _merge_power_solo(pool: dict, survivors: list):
    """Fold solo power deltas into the single shared 'power_solo' entry of ready_pools (call under pool_lock)."""
    entry = next((e for e in ready_pools if e.get("pool_id") == POWER_POOL_ID), None)
    if entry is None:
        entry = {"pool_id": POWER_POOL_ID, "chunk_idx": pool.get("chunk_idx"), "deltas": [], "_tokens": {}}
        ready_pools.append(entry)
    for path, _w in survivors:
        try: tok = max(1, int(open_delta(path, MODEL_SIZE).tokens))
        except Exception: tok = 1
        entry["_tokens"][path] = tok
    total = sum(entry["_tokens"].values()) or 1
    entry["deltas"] = [(p, POWER["SOLO_WEIGHT"] * t / total) for p, t in entry["_tokens"].items()]

def _matchmake(client_key: str, power: bool = False) -> dict:
    """Strict matching, FP32-first, with solo-pool re-matching (v2.3).

    Locked states:
      * pool is 'training'                           → return pool, never touched
      * pool is 'forming' and has 2+ clients         → return pool, never touched
    Solo forming pools:
      * kept as a fallback (no churn, no log)
      * but the client also scans every other forming pool on each poll;
        if a better-compatible pool is found, the solo pool is released
        (or popped if empty) and the client migrates.

    If no candidates and no solo pool → create a new pool from hw_sig."""
    hw_sig = _hw_signature(client_key)
    with pool_lock:
        pid = client_pool_map.get(client_key)
        existing_solo = None
        if pid:
            pool = pools.get(pid)
            if pool is None:
                # Pool was popped (cleanup, admin dissolve, etc.) — stale mapping
                client_pool_map.pop(client_key, None)
            else:
                same_round = pool.get("round", round_state["current_round"]) == round_state["current_round"]
                if not same_round:
                    client_pool_map.pop(client_key, None)
                elif pool["status"] == "training":
                    return pool
                elif pool["status"] == "forming" and len(pool["clients"]) > 1:
                    return pool
                elif pool["status"] == "forming" and len(pool["clients"]) == 1:
                    existing_solo = pool
                else:
                    # Unexpected status; treat as stale
                    client_pool_map.pop(client_key, None)

        # Scan candidates, EXCLUDING our own solo pool
        candidates = []
        for pool in pools.values():
            if pool is existing_solo: continue
            if pool["status"] != "forming": continue
            if len(pool["clients"]) >= MATCHMAKING["POOL_SIZE"]: continue
            if not _pools_compatible(hw_sig, pool["hardware_sig"]): continue
            # optional strict mode: power users only join a pool that their arrival completes
            if power and not POWER["JOIN_ANY_POOL"] and len(pool["clients"]) + 1 < MATCHMAKING["POOL_SIZE"]: continue
            candidates.append(pool)

        best = None
        if candidates:
            def _priority(p):
                return (_autocast_rank(p), -len(p["clients"]))
            best = min(candidates, key=_priority)

        if best is not None:
            if existing_solo is not None:
                try: existing_solo["clients"].remove(client_key)
                except ValueError: pass
                if not existing_solo["clients"]:
                    pools.pop(existing_solo["pool_id"], None)
                log.info(f"🧹 {client_key}: releasing solo pool {existing_solo['pool_id']} → joining {best['pool_id']}")
            best["clients"].append(client_key)
            best["last_activity"] = time.time()
            client_pool_map[client_key] = best["pool_id"]
            n = len(best["clients"])
            pool_ac = (best["hardware_sig"][2] if len(best["hardware_sig"]) > 2 else "any")
            log.info(f"👥 {client_key} → pool {best['pool_id']} ({n}/{MATCHMAKING['POOL_SIZE']}, "
                     f"backend={hw_sig[0]}, client_ac={hw_sig[2]}, optim={hw_sig[3]}, "
                     f"pool_ac={pool_ac}, priority=fp32-first)")
            if n >= MATCHMAKING["POOL_SIZE"]: _activate_pool(best)
            return best

        if existing_solo is not None:
            existing_solo["last_activity"] = time.time()
            return existing_solo

        pool_id = f"p{int(time.time()*1000)}_{secrets.token_hex(3)}"
        new_pool = {"pool_id": pool_id, "clients": [client_key], "hardware_sig": hw_sig,
            "status": "forming", "created_at": time.time(), "last_activity": time.time(),
            "chunk_idx": None, "task_id": None, "target_seed": None, "target_autocast": None,
            "round": round_state["current_round"], "deltas": {}}
        pools[pool_id] = new_pool
        client_pool_map[client_key] = pool_id
        log.info(f"🆕 {client_key} → NEW pool {pool_id} "
                 f"(backend={hw_sig[0]}, autocast={hw_sig[2]}, optim={hw_sig[3]})")
        return new_pool

def _build_task_payload(pool: dict, client_key: str) -> dict:
    metadata = {
        "taskId": pool["task_id"], "globalStep": global_step,
        "currentRound": round_state["current_round"],
        "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS, "modelConfig": MODEL_CONFIG,
        "weightFormat": "bf16",
        "datasetConfig": {"repoId": DATASET_CONFIG["repoId"], "chunkIdx": pool["chunk_idx"],
            "totalChunks": dataset_info["totalChunks"], "subChunkSize": DATASET_CONFIG["subChunkMB"] * 1024 * 1024,
            "epoch": data_position["epoch"], "tokensPerSample": DATASET_CONFIG["tokensPerSample"],
            "tokenBytes": DATASET_CONFIG["tokenBytes"]},
        "shardConfig": {"shardId": pool["pool_id"], "slot": pool["clients"].index(client_key),
            "total": len(pool["clients"])},
        "roundConfig": {"maxHours": round_state["round_duration_hours"],
            "targetDurationHours": round_state["round_duration_hours"]},
        "crossCheck": {"enabled": True, "cohortSize": len(pool["clients"]),
            "cohortPosition": pool["clients"].index(client_key),
            "targetSeed": pool["target_seed"],
            "targetLocalSteps": pool.get("target_steps", CROSS_CHECK["TARGET_LOCAL_STEPS"]),
            "targetBatchSize": CROSS_CHECK["TARGET_BATCH_SIZE"],
            "targetAccumSteps": CROSS_CHECK["TARGET_ACCUM_STEPS"],
            "targetAutocast": pool.get("target_autocast") or CROSS_CHECK["TARGET_AUTOCAST"],
            "targetOptimizer": CROSS_CHECK["TARGET_OPTIMIZER"],
            "harmonizeMode": "strict" if CROSS_CHECK["TARGET_BATCH_SIZE"] > 0 else "per_client_bs",
            "instruction": ("Train on the assigned shard with EXACTLY: "
                "seed=targetSeed, micro_steps=targetLocalSteps, "
                "batch_size=min(your_max_bs, targetBatchSize), "
                "accum_steps=targetAccumSteps, autocast=targetAutocast. "
                "If targetOptimizer is set (adamw_fp32/adamw8bit), use that optimizer. "
                "Report actualOptimizer in submit body.")},
    }
    hf = _build_hf_weights_metadata("bf16")
    metadata["weights"] = hf or {"source": "coordinator", "reason": "hf_unavailable_or_step_mismatch"}
    return metadata

def _stream_task_weights(metadata: dict, fmt: str = "bf16"):
    metadata_json = json.dumps(metadata).encode('utf-8')
    metadata_length = struct.pack('<I', len(metadata_json))
    def gen():
        yield metadata_length + metadata_json
        cs = 10_000_000
        for i in range(0, MODEL_SIZE, cs):
            chunk = global_weights[i:i + cs].cpu()
            if fmt == "bf16": yield chunk.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
            else: yield chunk.float().numpy().astype(np.float16).tobytes()
            del chunk
        for i in range(0, ENG_PARAMS, cs):
            chunk = global_engram_weights[i:i + cs].cpu()
            if fmt == "bf16": yield chunk.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
            else: yield chunk.float().numpy().astype(np.float16).tobytes()
            del chunk
    return gen()

def _maybe_complete_pool(pool: dict) -> bool:
    if pool["status"] != "training": return False
    n = len(pool["deltas"])
    age = time.time() - pool.get("activated_at", time.time())
    timed_out = age > MATCHMAKING["POOL_TIMEOUT_S"]
    if n >= len(pool["clients"]):
        pool["status"] = "completing"; return True
    if timed_out and n >= 1:
        pool["status"] = "completing"; return True
    return False

def _run_pool_cross_check(pool: dict) -> Optional[list]:
    """MANDATORY cross-check. Pools that can't be verified are DISCARDED."""
    pool_id = pool["pool_id"]
    client_paths = list(pool["deltas"].items())

    if pool.get("power") and len(pool["clients"]) == 1:
        return _run_power_solo_check(pool)
    # ── Mandatory: need 2+ deltas ──
    if len(client_paths) < 2:
        log.warning(f"🚫 Pool {pool_id}: {len(client_paths)} delta(s) — cross-check MANDATORY, DISCARDING all")
        for _, p in client_paths:
            try: os.remove(p)
            except Exception: pass
        return None

    paths = [p for _, p in client_paths]

    with _cross_check_lock:
        if not _mem_guard(0.8, f"pool {pool_id} cross-check"):
            log.error(f"❌ Pool {pool_id}: memory pressure — cross-check MANDATORY, DISCARDING all")
            for p in paths:
                try: os.remove(p)
                except Exception: pass
            return None
        kept, weights, flagged, discarded = _cross_check_cohort(paths, MODEL_SIZE)

    for fp in flagged:
        _flag_byzantine_from_delta(fp, f"cross-check outlier in pool {pool_id}")
        try: os.remove(fp)
        except Exception: pass
    for dp in discarded:
        try: os.remove(dp)
        except Exception: pass

    # ── Mandatory: need 2+ survivors ──
    if len(kept) < 2:
        log.error(f"❌ Pool {pool_id}: {len(paths)}→{len(kept)} survivors — cross-check FAILED, DISCARDING")
        for kp in kept:
            try: os.remove(kp)
            except Exception: pass
        return None

    for kp in kept:
        try:
            with open(kp, 'rb') as f:
                ml = struct.unpack('<I', f.read(4))[0]
                md = json.loads(f.read(ml).decode('utf-8'))
            nk = md.get("nodeKey", "")
            uid = _user_id_from_key(nk)
            if uid:
                with users_lock:
                    u = users_db["users"].get(uid)
                    if u:
                        u["stats"]["contributions"] = u["stats"].get("contributions", 0) + 1
                        save_users_db()
                log.info(f"✅ trust credited to {nk} (pool {pool_id})")
        except Exception as e: log.error(f"credit failed for {kp}: {e}")

    log.info(f"✅ Pool {pool_id}: {len(paths)}→{len(kept)} survivor(s), cross-check PASSED")
    return list(zip(kept, weights))

def _complete_pool_sync(pool: dict):
    pool_id = pool["pool_id"]
    try:
        survivors = _run_pool_cross_check(pool)
        with pool_lock:
            if survivors and pool.get("power") and len(pool["clients"]) == 1:
                _merge_power_solo(pool, survivors)
            elif survivors:
                ready_pools.append({"pool_id": pool_id, "chunk_idx": pool["chunk_idx"], "deltas": survivors})
            pools.pop(pool_id, None)
    except Exception as e:
        log.error(f"❌ Pool {pool_id} completion failed: {e}")
        with pool_lock: pools.pop(pool_id, None)

def _complete_pool_async(pool: dict): _complete_pool_sync(pool)
def _spawn_pool_completion(pool: dict):
    threading.Thread(target=_complete_pool_async, args=(pool,), daemon=True).start()

def _pool_cleanup_tick():
    """Forming-pool lifecycle.

    Solo-dissolve branches are independent of FORMING_TIMEOUT_S (v2.2 fix).
    With v2.3 solo-pool re-matching, a client that finds no better pool
    stays in its solo pool unchanged (no churn). The pool still dissolves
    at its solo timeout so the client retries with a fresh pool."""
    now = time.time(); to_spawn = []
    with pool_lock:
        for pid, pool in list(pools.items()):
            if pool["status"] == "forming":
                age = now - pool.get("created_at", now)
                alive = [c for c in pool["clients"] if now - active_clients.get(c, 0) < FL_CONFIG["IP_TIMEOUT_S"]]
                if len(alive) != len(pool["clients"]):
                    for c in pool["clients"]:
                        if c not in alive and client_pool_map.get(c) == pid:
                            client_pool_map.pop(c, None)
                    pool["clients"] = alive
                pool_ac = str((pool["hardware_sig"][2]
                               if len(pool["hardware_sig"]) > 2 else "any") or "any").lower()

                if not alive:
                    log.info(f"🧹 Dissolving empty forming pool {pid} (ac={pool_ac})")
                    pools.pop(pid, None)
                    continue

                if len(alive) >= 2:
                    if age > MATCHMAKING["FORMING_TIMEOUT_S"]:
                        log.info(f"⏱️ Activating {len(alive)}-client pool {pid} (ac={pool_ac})")
                        _activate_pool(pool)
                    continue

                # len(alive) == 1 → solo path (independent of FORMING_TIMEOUT_S)
                solo_timeout = (MATCHMAKING["BF16_SOLO_DISSOLVE_S"]
                                if pool_ac == "bf16"
                                else MATCHMAKING["FP32_SOLO_DISSOLVE_S"])
                if age > solo_timeout:
                    log.info(f"🧹 Dissolving solo forming pool {pid} "
                             f"(ac={pool_ac}, age={age:.0f}s > {solo_timeout}s)")
                    for c in pool["clients"]:
                        if client_pool_map.get(c) == pid: client_pool_map.pop(c, None)
                    pools.pop(pid, None)

            elif pool["status"] == "training":
                age = now - pool.get("activated_at", now)
                n_deltas = len(pool["deltas"])
                if age > MATCHMAKING["POOL_TIMEOUT_S"]:
                    if n_deltas >= 1 and _maybe_complete_pool(pool): to_spawn.append(pool)
                    else:
                        log.warning(f"🧹 Timed-out training pool {pid} (age={age:.0f}s, deltas={n_deltas}) — releasing {len(pool['clients'])} client(s)")
                        for c in pool["clients"]:
                            if client_pool_map.get(c) == pid: client_pool_map.pop(c, None)
                        pools.pop(pid, None)
    for p in to_spawn: _spawn_pool_completion(p)

# ============ MODEL WEIGHTS ============
def initialize_model():
    global global_weights, global_engram_weights
    torch.manual_seed(42)
    global_weights = (torch.randn(MODEL_SIZE) * 0.02).float()
    global_engram_weights = (torch.randn(ENG_PARAMS) * 0.02).float()
    log.info(f"🎲 initialized {MODEL_SIZE:,} base + {ENG_PARAMS:,} engram weights (FP32 master)")

def load_global_state():
    global global_step, global_stats, lora_stats
    if not GLOBAL_STATE_FILE.exists():
        log.warning("⚠️ global_state.json not found. Starting fresh."); return
    try:
        with open(GLOBAL_STATE_FILE, 'r', encoding='utf-8') as f: state = json.load(f)
        global_step = state.get("globalStep", 0)
        if "globalStats" in state:
            global_stats.clear(); global_stats.update(state["globalStats"])
            global_stats["lossHistory"] = global_stats.get("lossHistory", [])[-100:]
        if "loraStats" in state:
            lora_stats.clear(); lora_stats.update(state["loraStats"])
        log.info(f"✅ Loaded state → step {global_step}, tokens {global_stats.get('totalTokens', 0):,}")
    except Exception as e: log.error(f"❌ Failed to parse global_state.json: {e}")

def save_global_state():
    try:
        state_data = {"globalStep": global_step, "savedAt": datetime.now(timezone.utc).isoformat(),
            "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS, "modelConfig": MODEL_CONFIG,
            "globalStats": global_stats, "loraStats": lora_stats}
        temp_file = GLOBAL_STATE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f: json.dump(state_data, f, indent=2)
        os.replace(temp_file, GLOBAL_STATE_FILE)
    except Exception as e: log.error(f"❌ Failed to save global_state.json: {e}")

def load_model_weights():
    global global_weights
    if not LATEST_CHECKPOINT_FILE.exists():
        log.warning("⚠️ model_latest.bin not found. Weights will be random."); return
    try:
        buf = np.fromfile(LATEST_CHECKPOINT_FILE, dtype=np.float32)
        if len(buf) == MODEL_SIZE:
            global_weights = torch.from_numpy(buf.copy()); log.info("✅ Loaded base model weights")
        else: log.error(f"❌ model_latest.bin corrupted: expected {MODEL_SIZE}, got {len(buf)}")
    except Exception as e: log.error(f"❌ Failed to load base model weights: {e}")

def load_engram_weights():
    global global_engram_weights
    if not ENGRAM_CHECKPOINT_FILE.exists():
        log.warning("⚠️ engram_latest.bin not found. Engram will be random."); return
    try:
        buf = np.fromfile(ENGRAM_CHECKPOINT_FILE, dtype=np.float32)
        if len(buf) == ENG_PARAMS:
            global_engram_weights = torch.from_numpy(buf.copy()); log.info("✅ Loaded engram weights")
        else: log.error(f"❌ engram_latest.bin corrupted")
    except Exception as e: log.error(f"❌ Failed to load engram weights: {e}")

def save_model_weights():
    try:
        temp_file = LATEST_CHECKPOINT_FILE.with_suffix('.tmp')
        with open(temp_file, 'wb') as f:
            for i in range(0, MODEL_SIZE, 10_000_000):
                global_weights[i:i+10_000_000].cpu().float().numpy().tofile(f)
        gc.collect(); os.replace(temp_file, LATEST_CHECKPOINT_FILE)
    except Exception as e: log.error(f"❌ Failed to save base model weights: {e}")

def save_engram_weights():
    try:
        temp_file = ENGRAM_CHECKPOINT_FILE.with_suffix('.tmp')
        with open(temp_file, 'wb') as f:
            for i in range(0, ENG_PARAMS, 10_000_000):
                global_engram_weights[i:i+10_000_000].cpu().float().numpy().tofile(f)
        gc.collect(); os.replace(temp_file, ENGRAM_CHECKPOINT_FILE)
    except Exception as e: log.error(f"❌ Failed to save engram weights: {e}")

def save_latest_checkpoint(): save_global_state(); save_model_weights(); save_engram_weights()

# ============ HUGGINGFACE UPLOAD ============
last_hf_upload_time = 0
hf_upload_lock = threading.Lock()
hf_info_lock = threading.Lock()
status_upload_lock = threading.Lock()
current_hf_revision: Optional[str] = None
hf_revision_global_step: int = -1
hf_files_info: Dict[str, Dict] = {}

def _hf_resolve_url(revision: str, filename: str) -> str:
    return f"https://huggingface.co/{HF_REPO_ID}/resolve/{revision}/{filename}"
def _extract_commit_sha(commit_info) -> Optional[str]:
    if commit_info is None: return None
    for attr in ("commit_oid", "oid", "commit_id", "sha"):
        v = getattr(commit_info, attr, None)
        if v: return str(v)
    url = getattr(commit_info, "commit_url", None)
    if url: return str(url).rstrip("/").split("/")[-1]
    return None
def _latest_hf_revision() -> Optional[str]:
    if not HF_ENABLED: return None
    try:
        r = requests.head(f"https://huggingface.co/{HF_REPO_ID}/resolve/main/config.json", allow_redirects=False, timeout=15)
        sha = r.headers.get("x-repo-commit")
        if sha and len(sha) >= 12: return sha
        loc = r.headers.get("location", "")
        for part in loc.split("/"):
            if len(part) == 40 and all(c in "0123456789abcdef" for c in part): return part
    except Exception as e: log.warning(f"Could not determine latest HF revision: {e}")
    return None
def _get_hf_config_step(revision: Optional[str]) -> Optional[int]:
    if not revision: return None
    try:
        r = requests.get(_hf_resolve_url(revision, "config.json"), timeout=20, allow_redirects=True)
        if r.ok:
            j = r.json(); gs = j.get("globalStep")
            if gs is not None: return int(gs)
    except Exception as e: log.warning(f"Could not read HF config.json: {e}")
    return None
def _refresh_hf_file_info(revision: Optional[str]) -> Dict[str, Dict]:
    info = {}
    if not revision: return info
    for fn in HF_WEIGHT_FILES:
        try:
            r = requests.head(_hf_resolve_url(revision, fn), timeout=30, allow_redirects=True)
            if r.ok:
                size = int(r.headers.get("content-length", 0) or 0)
                if size > 0: info[fn] = {"size": size, "etag": r.headers.get("etag")}
        except Exception as e: log.warning(f"Could not HEAD HF file {fn}: {e}")
    return info
def _set_hf_revision(revision: Optional[str], step: int) -> bool:
    global current_hf_revision, hf_revision_global_step, hf_files_info
    if not revision: return False
    info = _refresh_hf_file_info(revision)
    if len(info) < len(HF_WEIGHT_FILES):
        log.warning("HF file info incomplete; HF direct downloads disabled"); return False
    with hf_info_lock: hf_files_info = info
    current_hf_revision = revision; hf_revision_global_step = int(step)
    log.info(f"🤗 HF direct weights enabled: revision={revision[:12]} step={hf_revision_global_step}")
    return True
def _build_hf_weights_metadata(fmt: str) -> Optional[dict]:
    if fmt != "bf16" or not HF_ENABLED or not current_hf_revision: return None
    if hf_revision_global_step != global_step: return None
    with hf_info_lock: info = dict(hf_files_info)
    files = {}
    for role, fn in (("model", "model_bf16.safetensors"), ("engram", "engram_bf16.safetensors")):
        fi = info.get(fn)
        if not fi or not fi.get("size"): return None
        files[role] = {"path": fn, "url": _hf_resolve_url(current_hf_revision, fn),
                       "size": int(fi["size"]), "dtype": "bf16", "layout": "flat", "tensorKey": "weights"}
    return {"source": "huggingface", "repo": HF_REPO_ID, "revision": current_hf_revision, "format": "bf16", "files": files}

def upload_to_huggingface(retries: int = 3) -> bool:
    global last_hf_upload_time
    if not HF_ENABLED: return False
    with hf_upload_lock:
        for attempt in range(1, retries + 1):
            temp_dir = None
            try:
                log.info(f"🤗 Uploading to HF (attempt {attempt}/{retries})")
                api = HfApi(token=HF_TOKEN)
                temp_dir = Path(tempfile.mkdtemp(prefix="hf_upload_", dir=str(CHECKPOINT_DIR)))
                _save_fp32 = _avail_mem_gb() > 8.0
                formats = [("model_bf16.safetensors", torch.bfloat16), ("engram_bf16.safetensors", torch.bfloat16)]
                if _save_fp32: formats.extend([("model_fp32.safetensors", torch.float32), ("engram_fp32.safetensors", torch.float32)])
                else: log.info("⚠️ Skipping FP32 HF uploads to preserve RAM")
                for fn, dtype in formats:
                    src = global_weights if fn.startswith("model_") else global_engram_weights
                    w = src.cpu().to(dtype)
                    save_file({"weights": w}, str(temp_dir / fn))
                    del w; gc.collect()
                config_data = {"architecture": MODEL_CONFIG["architecture"], "vocabSize": MODEL_CONFIG["vocabSize"],
                    "dim": MODEL_CONFIG["dim"], "nLayers": MODEL_CONFIG["nLayers"],
                    "nHeads": MODEL_CONFIG["nHeads"], "nKvHeads": MODEL_CONFIG["nKvHeads"],
                    "headDim": MODEL_CONFIG["headDim"], "maxSeqLen": MODEL_CONFIG["maxSeqLen"],
                    "mlpHidden": MODEL_CONFIG["mlpHidden"], "weightTying": MODEL_CONFIG["weightTying"],
                    "globalStep": global_step, "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS,
                    "uploadedAt": datetime.now(timezone.utc).isoformat()}
                with open(temp_dir / "config.json", 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=2)
                commit = api.upload_folder(folder_path=str(temp_dir), repo_id=HF_REPO_ID, repo_type="model",
                    commit_message=f"Update step {global_step} round {round_state['current_round']}")
                rev = _extract_commit_sha(commit) or _latest_hf_revision()
                if rev and _set_hf_revision(rev, global_step):
                    last_hf_upload_time = time.time()
                    log.info(f"✅ Uploaded to HF at step {global_step}")
                    return True
            except Exception as e:
                log.error(f"❌ HF upload failed (attempt {attempt}): {e}")
                if attempt < retries: time.sleep(5 * attempt)
            finally:
                if temp_dir is not None and temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
        return False

def sync_huggingface_state():
    if not HF_ENABLED: return
    try:
        api = HfApi(token=HF_TOKEN)
        files = set(api.list_repo_files(HF_REPO_ID, repo_type="model"))
        required = HF_WEIGHT_FILES + ["config.json"]
        missing = [f for f in required if f not in files]
        if missing:
            log.warning(f"⚠️ Missing HF files: {missing}"); upload_to_huggingface(); return
        rev = _latest_hf_revision()
        if not rev: return
        cfg_step = _get_hf_config_step(rev)
        if cfg_step is None: return
        if cfg_step == global_step: _set_hf_revision(rev, global_step)
        elif cfg_step < global_step and LATEST_CHECKPOINT_FILE.exists(): upload_to_huggingface()
    except Exception as e: log.error(f"❌ HF sync failed: {e}")

# ============ STATUS ============
def update_and_save_status(push_hf: bool = True):
    loss_history = global_stats["lossHistory"]
    training_loss_change = (round(((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100, 1)
        if len(loss_history) >= 2 and loss_history[0] > 0 else 0.0)
    active_gpus = _active_client_count()
    total_tokens = global_stats["totalTokens"]
    with users_lock:
        total_participants = len(users_db["users"])
        users_list = [{"uuid": u["uuid"], "username": u["username"],
            "contributions": u["stats"].get("contributions", 0),
            "totalTokens": u["stats"].get("totalTokens", 0),
            "totalSteps": u["stats"].get("totalSteps", 0),
            "trustScore": compute_trust_score(u["stats"]),
            "byzantineFlags": u["stats"].get("byzantine_flags", 0),
            "power": bool(u.get("power", False))} for u in users_db["users"].values()]
    leaderboard_sorted = sorted(users_list, key=lambda x: x["totalTokens"], reverse=True)
    with pool_lock:
        mm_snapshot = {"poolSize": MATCHMAKING["POOL_SIZE"], "activePools": len(pools),
            "formingPools": sum(1 for p in pools.values() if p["status"] == "forming"),
            "trainingPools": sum(1 for p in pools.values() if p["status"] == "training"),
            "readyPools": len(ready_pools)}
    status_data = {"totalParticipants": total_participants,
        "totalContributions": lora_stats.get("total_submissions", 0),
        "activeGpus": active_gpus, "totalTokens": total_tokens,
        "trainingLossChange": training_loss_change, "globalStep": global_step,
        "lastLoss": global_stats["lastLoss"],
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "currentRound": round_state["current_round"],
        "hfRevision": current_hf_revision, "hfRevisionStep": hf_revision_global_step,
        "hfDirectDownloads": bool(_build_hf_weights_metadata("bf16")),
        "crossCheck": {"enabled": CROSS_CHECK["ENABLED"], "cohortSize": CROSS_CHECK["COHORT_SIZE"],
            "maxOverflow": CROSS_CHECK["MAX_OVERFLOW"], "threshold": CROSS_CHECK["DISTANCE_THRESHOLD_RATIO"],
            "targetBatchSize": CROSS_CHECK["TARGET_BATCH_SIZE"], "targetLocalSteps": CROSS_CHECK["TARGET_LOCAL_STEPS"],
            "targetOptimizer": CROSS_CHECK["TARGET_OPTIMIZER"]},
        "matchmaking": mm_snapshot, "byzantineFlagCounts": dict(byzantine_flag_counts)}
    try:
        with open(BASE_DIR / "status.json", 'w', encoding='utf-8') as f: json.dump(status_data, f, indent=2)
        with open(BASE_DIR / "leaderboard.json", 'w', encoding='utf-8') as f: json.dump(leaderboard_sorted, f, indent=2)
    except Exception as e: log.error(f"❌ Failed to save json files: {e}")
    if not (push_hf and HF_ENABLED): return
    if not status_upload_lock.acquire(blocking=False): return
    temp_dir = None
    try:
        api = HfApi(token=HF_TOKEN)
        temp_dir = Path(tempfile.mkdtemp(prefix="hf_status_", dir=str(BASE_DIR)))
        with open(temp_dir / "status.json", 'w', encoding='utf-8') as f: json.dump(status_data, f, indent=2)
        with open(temp_dir / "leaderboard.json", 'w', encoding='utf-8') as f: json.dump(leaderboard_sorted, f, indent=2)
        def fmt(n):
            if n >= 1e12: return f"{n/1e12:.1f}T"
            if n >= 1e9: return f"{n/1e9:.1f}B"
            if n >= 1e6: return f"{n/1e6:.1f}M"
            if n >= 1e3: return f"{n/1e3:.1f}K"
            return str(n)
        badge_data = {"schemaVersion": 1, "label": "Tokens Processed", "message": fmt(total_tokens), "color": "blue"}
        with open(temp_dir / "badge_tokens.json", 'w', encoding='utf-8') as f: json.dump(badge_data, f, indent=2)
        operations = [
            CommitOperationAdd(path_in_repo="status.json", path_or_fileobj=str(temp_dir / "status.json")),
            CommitOperationAdd(path_in_repo="leaderboard.json", path_or_fileobj=str(temp_dir / "leaderboard.json")),
            CommitOperationAdd(path_in_repo="badge_tokens.json", path_or_fileobj=str(temp_dir / "badge_tokens.json")),
        ]
        api.create_commit(repo_id=HF_REPO_ID, repo_type="model", operations=operations,
            commit_message=f"chore: update stats ({badge_data['message']} tokens)")
        log.info("✅ Uploaded status/leaderboard/badge to HF")
    except Exception as e: log.error(f"❌ HF status upload failed: {e}")
    finally:
        if temp_dir is not None and temp_dir.exists(): shutil.rmtree(temp_dir, ignore_errors=True)
        status_upload_lock.release()

def hf_status_loop():
    if not HF_ENABLED: return
    time.sleep(60)
    while True:
        try: update_and_save_status(push_hf=True)
        except Exception as e: log.error(f"❌ hf_status_loop error: {e}")
        time.sleep(300)

def emergency_save():
    log.info("🛑 Shutting down, saving final state...")
    try: save_global_state(); save_model_weights(); save_engram_weights(); save_users_db(); save_dataset_state()
    except Exception as e: log.error(f"emergency_save: {e}")
    try: update_and_save_status(push_hf=False)
    except Exception: pass
atexit.register(emergency_save)

def _server_lr(step: int) -> float:
    base = FL_CONFIG["SERVER_LR"]
    floor = FL_CONFIG.get("MIN_LR_MULT", 0.05)
    total = FL_CONFIG.get("LR_DECAY_STEPS", 50_000)
    if total <= 0: return base
    progress = min(1.0, max(0.0, step / total))
    mult = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * mult

def compute_max_allowed_norm(local_steps, rank=None):
    steps = max(1, int(local_steps or 1))
    base = FL_CONFIG["DELTA_NORM_BASE"] * math.sqrt(steps / 100.0)
    rank_factor = np.sqrt(rank / LORA_CONFIG["FULL_RANK"]) if rank and rank < LORA_CONFIG["FULL_RANK"] else 1.0
    return base * rank_factor

def handle_checkpoints():
    try:
        fp = CHECKPOINT_DIR / f"model_step_{global_step}.bin"
        with open(fp, 'wb') as f:
            for i in range(0, MODEL_SIZE, 10_000_000):
                global_weights[i:i+10_000_000].cpu().float().numpy().tofile(f)
        fp_eng = CHECKPOINT_DIR / f"engram_step_{global_step}.bin"
        with open(fp_eng, 'wb') as f:
            for i in range(0, ENG_PARAMS, 10_000_000):
                global_engram_weights[i:i+10_000_000].cpu().float().numpy().tofile(f)
        gc.collect()
        save_model_weights(); save_engram_weights()
        with open(CHECKPOINT_DIR / f"model_step_{global_step}.json", 'w', encoding='utf-8') as f:
            json.dump({"globalStep": global_step, "savedAt": datetime.now(timezone.utc).isoformat(),
                       "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS}, f, indent=2)
        files = sorted([f for f in CHECKPOINT_DIR.iterdir() if f.name.startswith("model_step_") and f.suffix == ".bin"],
                       key=lambda f: int(f.stem.split("_")[2]))
        while len(files) > FL_CONFIG["MAX_CHECKPOINTS"]:
            old = files.pop(0); old.unlink()
            old_eng = CHECKPOINT_DIR / old.name.replace("model_", "engram_")
            if old_eng.exists(): old_eng.unlink()
            meta = CHECKPOINT_DIR / f"{old.stem}.json"
            if meta.exists(): meta.unlink()
    except Exception as e: log.error(f"❌ checkpoint: {e}")

# ============ LOSS RATCHET ============
LOSS_STATE_FILE = CHECKPOINT_DIR / "loss_ratchet.json"
CANDIDATE_MODEL_FILE = CHECKPOINT_DIR / "candidate_model_bf16.bin"
CANDIDATE_ENGRAM_FILE = CHECKPOINT_DIR / "candidate_engram_bf16.bin"
loss_ratchet_state = {"lastAcceptedLoss": None, "lastAcceptedStep": None, "history": []}
_candidate_token: Optional[str] = None

def _load_loss_state():
    try:
        if LOSS_STATE_FILE.exists():
            with open(LOSS_STATE_FILE, "r", encoding="utf-8") as f: loss_ratchet_state.update(json.load(f))
    except Exception as e: log.error(f"❌ loss_ratchet.json unreadable: {e}")
def _save_loss_state():
    try:
        loss_ratchet_state["history"] = loss_ratchet_state["history"][-200:]
        tmp = LOSS_STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f: json.dump(loss_ratchet_state, f, indent=2)
        os.replace(tmp, LOSS_STATE_FILE)
    except Exception as e: log.error(f"❌ loss_ratchet save failed: {e}")
def _write_bf16(path, src, n, piece=10_000_000) -> str:
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for i in range(0, n, piece):
            b = src[i:i + piece].cpu().float().to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
            f.write(b); h.update(b)
    return h.hexdigest()
def _stage_candidate() -> dict:
    global _candidate_token
    with shard_lock:
        m_sha = _write_bf16(CANDIDATE_MODEL_FILE, global_weights, MODEL_SIZE)
        e_sha = _write_bf16(CANDIDATE_ENGRAM_FILE, global_engram_weights, ENG_PARAMS)
    _candidate_token = secrets.token_urlsafe(24)
    base = HARDEN["PUBLIC_URL"]
    return {"modelUrl": f"{base}/api/admin/candidate/model?token={_candidate_token}",
            "engramUrl": f"{base}/api/admin/candidate/engram?token={_candidate_token}",
            "format": "bf16", "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS,
            "modelSha256": m_sha, "engramSha256": e_sha}
def _clear_candidate():
    global _candidate_token
    _candidate_token = None
    for p in (CANDIDATE_MODEL_FILE, CANDIDATE_ENGRAM_FILE):
        try: p.unlink(missing_ok=True)
        except Exception: pass
def _query_evaluator(step: int, cand: dict) -> Optional[float]:
    if not HARDEN["LOSS_CHECK_URL"]:
        log.warning("⚠️ LOSS_CHECK_MODE on but URL empty"); return None
    headers = {"Content-Type": "application/json"}
    if HARDEN["LOSS_CHECK_TOKEN"]: headers["Authorization"] = f"Bearer {HARDEN['LOSS_CHECK_TOKEN']}"
    payload = {"step": step, "round": round_state["current_round"],
               "previousAcceptedLoss": loss_ratchet_state.get("lastAcceptedLoss"), "candidate": cand}
    try:
        r = requests.post(HARDEN["LOSS_CHECK_URL"], json=payload, headers=headers, timeout=HARDEN["LOSS_TIMEOUT_S"])
        r.raise_for_status()
        loss = float(r.json()["loss"])
        return loss if math.isfinite(loss) else float("nan")
    except Exception as e: log.error(f"❌ Evaluator failed: {e}"); return None
def _ensure_rollback_point():
    ok = (LATEST_CHECKPOINT_FILE.exists() and ENGRAM_CHECKPOINT_FILE.exists()
          and LATEST_CHECKPOINT_FILE.stat().st_size == MODEL_SIZE * 4
          and ENGRAM_CHECKPOINT_FILE.stat().st_size == ENG_PARAMS * 4)
    if not ok:
        log.info("💾 Writing pre-round rollback point"); save_model_weights(); save_engram_weights()
def _finish_with_loss_check(step_before: int) -> bool:
    global global_step
    mode = HARDEN["LOSS_CHECK_MODE"]; step = global_step
    verdict, reason, loss = "accept", "", None
    try:
        cand = _stage_candidate()
        loss = _query_evaluator(step, cand)
    except Exception as e: log.error(f"❌ Staging failed: {e}"); loss = None
    finally: _clear_candidate()
    last = loss_ratchet_state.get("lastAcceptedLoss")
    if loss is None:
        if HARDEN["LOSS_FAIL_OPEN"]: reason = "evaluator unavailable (fail-open)"
        else: verdict, reason = "reject", "evaluator unavailable (fail-closed)"
    elif not math.isfinite(loss): verdict, reason = "reject", "non-finite held-out loss"
    elif last is None: reason = f"first measurement, loss={loss:.5f}"
    else:
        d = loss - float(last)
        if d <= HARDEN["LOSS_TOLERANCE"]: reason = f"delta={d:+.5f} <= tol={HARDEN['LOSS_TOLERANCE']:.5f}"
        else: verdict, reason = "reject", f"delta={d:+.5f} > tol={HARDEN['LOSS_TOLERANCE']:.5f}"
    loss_ratchet_state["history"].append({"step": step, "loss": loss, "verdict": verdict, "mode": mode,
                                          "reason": reason, "at": datetime.now(timezone.utc).isoformat()})
    if verdict == "reject" and mode == "enforce":
        log.warning(f"🛑 LOSS RATCHET: step {step} REJECTED; rolling back to {step_before}")
        with shard_lock:
            load_model_weights(); load_engram_weights()
            with stats_lock: global_step = step_before
        _save_loss_state(); return False
    if verdict == "reject": log.warning(f"🟡 LOSS RATCHET: would reject ({reason})")
    else: log.info(f"🟢 LOSS RATCHET: accepted ({reason})")
    if loss is not None and math.isfinite(loss) and verdict == "accept":
        loss_ratchet_state["lastAcceptedLoss"] = loss; loss_ratchet_state["lastAcceptedStep"] = step
    _save_loss_state(); handle_checkpoints(); return True

# ============ CHUNKED UPLOAD STATE ============
_active_uploads: Dict[str, dict] = {}
uploads_lock = threading.RLock()

def _valid_upload_id(uid) -> bool:
    if not isinstance(uid, str) or len(uid) != 32: return False
    return all(c in "0123456789abcdef" for c in uid.lower())
def _cleanup_expired_uploads():
    now = time.time()
    with uploads_lock:
        for uid, up in list(_active_uploads.items()):
            if up.get("status") == "finalizing": continue
            if now - up.get("last_activity", up.get("created", 0)) > UPLOAD_TTL_SECONDS:
                try: shutil.rmtree(up["dir"], ignore_errors=True)
                except Exception: pass
                del _active_uploads[uid]
    try:
        for p in UPLOAD_DIR.iterdir():
            if p.is_dir():
                if now - p.stat().st_mtime > UPLOAD_TTL_SECONDS: shutil.rmtree(p, ignore_errors=True)
            else:
                if now - p.stat().st_mtime > UPLOAD_TTL_SECONDS: p.unlink(missing_ok=True)
    except Exception: pass

# ============ BACKGROUND THREADS ============
def cleanup_loop():
    while True:
        time.sleep(FL_CONFIG["CLEANUP_INTERVAL_S"])
        now = time.time(); now_ms = int(now * 1000)
        for key in list(active_clients.keys()):
            if now - active_clients[key] > FL_CONFIG["IP_TIMEOUT_S"]: del active_clients[key]
        for k in list(client_hardware.keys()):
            hw_ls = client_hardware[k].get("last_seen", 0) if isinstance(client_hardware[k], dict) else 0
            if now - hw_ls > 24 * 3600: del client_hardware[k]
        for k in list(client_last_submit.keys()):
            if now - client_last_submit[k] > 24 * 3600: del client_last_submit[k]
        for sn in list(blacklisted_subnets.keys()):
            if now > blacklisted_subnets[sn]: del blacklisted_subnets[sn]
        for k in list(banned_client_keys.keys()):
            if banned_client_keys[k] and banned_client_keys[k] < now: del banned_client_keys[k]
        for k in list(banned_users.keys()):
            if banned_users[k].get("until") and banned_users[k]["until"] < now: del banned_users[k]
        task_timeout_ms = 72 * 60 * 60 * 1000
        expired_tasks = [tid for tid, meta in db["tasks"].items() if now_ms - meta.get("createdAt", now_ms) > task_timeout_ms]
        for tid in expired_tasks: del db["tasks"][tid]
        stale_shards = [sid for sid, meta in db["shardMeta"].items() if now_ms - meta.get("createdAt", 0) > FL_CONFIG["SHARD_TIMEOUT_S"] * 1000]
        for sid in stale_shards:
            db["shardMeta"].pop(sid, None); db["pendingSubmissions"].pop(sid, None)
        for f in DELTA_TEMP_DIR.glob("delta_*.bin"):
            if time.time() - f.stat().st_mtime > 18000:
                try: f.unlink()
                except Exception: pass
        _cleanup_expired_uploads()
        try: _pool_cleanup_tick()
        except Exception as e: log.error(f"pool cleanup error: {e}")
        if expired_tasks or stale_shards:
            log.info(f"🧹 GC: cleaned {len(expired_tasks)} tasks + {len(stale_shards)} shards")
        gc.collect()

def _bg_hf_upload_and_status():
    if not _mem_guard(8.0, "HF weights upload"):
        log.warning("⚠️ Skipping HF weights upload — memory pressure.")
        return
    try:
        if upload_to_huggingface(retries=3): log.info("✅ HF weights synced (bg)")
        else: log.warning("❌ HF sync failed (bg)")
    except Exception as e: log.error(f"❌ BG HF upload error: {e}")
    try: update_and_save_status(push_hf=True)
    except Exception as e: log.error(f"❌ BG status update error: {e}")

def round_watchdog():
    while True:
        try:
            time.sleep(30)
            should_end_round = False
            current_round_ending = round_state["current_round"]
            with shard_lock:
                round_elapsed_hours = (time.time() - round_state["round_start_time"]) / 3600
                force_end_at = round_state.get("force_end_at")
                force_due = force_end_at is not None and time.time() >= force_end_at
                if round_elapsed_hours >= round_state["round_duration_hours"] * ROUND_CONFIG["ULTIMATUM_THRESHOLD"]:
                    for client_key, client_data in list(round_state["participating_clients"].items()):
                        if not client_data["submitted"] and not client_data.get("ultimatum_sent"):
                            last_seen = active_clients.get(client_key, 0)
                            if time.time() - last_seen < FL_CONFIG["IP_TIMEOUT_S"]: continue
                            client_data["ultimatum_sent"] = True
                            client_data["ultimatum_time"] = time.time()
                for client_key in list(round_state["participating_clients"].keys()):
                    client_data = round_state["participating_clients"][client_key]
                    if client_data["submitted"] and client_data.get("ultimatum_sent"):
                        client_data["ultimatum_sent"] = False; continue
                    if client_data.get("ultimatum_sent"):
                        if (time.time() - client_data.get("ultimatum_time", 0)) / 60 > ROUND_CONFIG["ULTIMATUM_TIMEOUT_MIN"]:
                            del round_state["participating_clients"][client_key]
                if not round_state["is_aggregating"]:
                    if (round_elapsed_hours >= round_state["round_duration_hours"] or force_due):
                        should_end_round = True
                        round_state["is_aggregating"] = True
                        if force_due:
                            round_state["force_end_at"] = None
                            log.warning(f"⏱️ Admin-forced end of round {current_round_ending}")
                        log.info(f"⏰ Round {current_round_ending} ended ({round_elapsed_hours:.2f}h)")
            if should_end_round:
                try:
                    to_force = []
                    with pool_lock:
                        for pool in pools.values():
                            if pool["status"] == "training" and len(pool["deltas"]) >= 1:
                                pool["status"] = "completing"; to_force.append(pool)
                        for pid in list(pools.keys()):
                            pool = pools[pid]
                            should_drop = (pool["status"] == "forming" or (pool["status"] == "training" and len(pool["deltas"]) == 0))
                            if should_drop:
                                for p in pool.get("deltas", {}).values():
                                    try: os.remove(p)
                                    except Exception: pass
                                for c in pool["clients"]:
                                    if client_pool_map.get(c) == pid: client_pool_map.pop(c, None)
                                pools.pop(pid, None)
                    for p in to_force:
                        log.info(f"⏱️ Force-completing pool {p['pool_id']} ({len(p['deltas'])}/{len(p['clients'])} deltas)")
                        _complete_pool_sync(p)
                    with pool_lock:
                        ready = list(ready_pools); ready_pools.clear()
                    success = False; loss_rejected = False
                    if ready:
                        step_before = global_step
                        if HARDEN["LOSS_CHECK_MODE"] == "enforce": _ensure_rollback_point()
                        apply_round_aggregation(ready)
                        success = global_step > step_before
                        if not success: log.warning("⚠️ Aggregation was a no-op; chunk will NOT advance")
                        elif HARDEN["LOSS_CHECK_MODE"] != "off":
                            if not _finish_with_loss_check(step_before):
                                loss_rejected = True; success = False
                    else: log.warning("⚠️ Round ended with no completed pools")
                    with shard_lock:
                        if success or loss_rejected:
                            current_data_shard["consumed"] = True
                            if data_position.get("permutation"):
                                data_position["consumed_through_pos"] = data_position["permutationPos"] - 1
                                log.info(f"✅ Pools consumed through pos {data_position['consumed_through_pos']}")
                        round_state["current_round"] += 1
                        round_state["round_start_time"] = time.time()
                        round_state["participating_clients"] = {}
                        round_state["aggregations_this_round"] = 0
                        round_state["last_aggregation_time"] = time.time()
                        round_state["is_aggregating"] = False
                        round_state["force_end_at"] = None
                    if success or loss_rejected: save_dataset_state()
                    log.info(f"🚀 Round {round_state['current_round']} started (step {global_step})")
                    try: update_and_save_status(push_hf=False)
                    except Exception as e: log.error(f"❌ Status write error: {e}")
                    if HF_ENABLED and not loss_rejected:
                        threading.Thread(target=_bg_hf_upload_and_status, daemon=True).start()
                except Exception as agg_err:
                    log.error(f"❌ Aggregation Failed: {agg_err}")
                    import traceback; traceback.print_exc()
                    with shard_lock:
                        round_state["is_aggregating"] = False
                        round_state["force_end_at"] = None
                        round_state["current_round"] += 1
                        round_state["round_start_time"] = time.time()
                        round_state["participating_clients"] = {}
        except Exception as e:
            log.error(f"❌ round_watchdog error: {e}")
            import traceback; traceback.print_exc()
            time.sleep(10)

threading.Thread(target=cleanup_loop, daemon=True).start()
threading.Thread(target=round_watchdog, daemon=True).start()
threading.Thread(target=hf_status_loop, daemon=True).start()

# ============ SUBMISSION PROCESSING ============
def process_decompressed_payload(dec_path: str, client_ip: str, client_key: str, request: Request, init_meta: Optional[dict] = None):
    temp_delta_path = None; final_delta_path = None
    try:
        with open(dec_path, 'rb') as dec_f:
            payload_len_bytes = dec_f.read(4)
            if len(payload_len_bytes) < 4: raise HTTPException(status_code=400, detail="Invalid payload")
            payload_len = struct.unpack('<I', payload_len_bytes)[0]
            payload_bytes = dec_f.read(payload_len)
            if len(payload_bytes) < payload_len: raise HTTPException(status_code=400, detail="Incomplete JSON")
            body = json.loads(payload_bytes.decode('utf-8'))
            if init_meta:
                for k in ("taskId", "weightFormat", "hasEngram", "engramSparseCount"):
                    if k in init_meta and body.get(k) != init_meta.get(k):
                        raise HTTPException(status_code=400, detail=f"Init mismatch: {k}")
            _q = _check_submission_quota(client_key, client_ip)
            if _q:
                log.warning(f"🚫 {client_key} over quota ({_q}); delta discarded")
                return _duplicate_response(_q)
            fmt = body.get("weightFormat", "bf16")
            if fmt not in ("bf16", "fp16", "fp8", "fp32"):
                raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")
            has_engram = bool(body.get("hasEngram", False))
            try: engram_sparse_count = int(body.get("engramSparseCount", 0) or 0)
            except Exception: engram_sparse_count = 0
            if engram_sparse_count < 0 or engram_sparse_count > ENG_NUM_BUCKETS:
                raise HTTPException(status_code=400, detail="Invalid engramSparseCount")
            bpp = _bytes_per_param(fmt)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".delta") as tdf:
                temp_delta_path = tdf.name
            norm_sq = 0.0; offset = 0; has_nan_inf = False
            with open(temp_delta_path, 'wb') as td:
                while offset < MODEL_SIZE:
                    read_params = min(10_000_000, MODEL_SIZE - offset)
                    chunk_bytes = dec_f.read(read_params * bpp)
                    if not chunk_bytes or len(chunk_bytes) < read_params * bpp:
                        raise HTTPException(status_code=400, detail="Incomplete base delta payload")
                    td.write(chunk_bytes)
                    chunk_t = _decode_delta_chunk(bytearray(chunk_bytes), fmt)
                    if torch.isnan(chunk_t).any() or torch.isinf(chunk_t).any():
                        has_nan_inf = True; break
                    norm_sq += torch.sum(chunk_t * chunk_t).item()
                    offset += read_params; del chunk_t
            if has_nan_inf:
                ban_subnet(client_ip, "submitted NaN/Inf delta")
                raise HTTPException(status_code=400, detail="Invalid delta (NaN/Inf)")
            delta_norm = math.sqrt(norm_sq)
            engram_indices = None; engram_values = None
            if has_engram and engram_sparse_count > 0:
                idx_bytes_len = engram_sparse_count * 4
                val_bytes_len = engram_sparse_count * MODEL_CONFIG["dim"] * bpp
                idx_bytes = read_exact(dec_f, idx_bytes_len)
                val_bytes = read_exact(dec_f, val_bytes_len)
                if len(idx_bytes) < idx_bytes_len or len(val_bytes) < val_bytes_len:
                    raise HTTPException(status_code=400, detail="Incomplete engram payload")
                engram_indices = np.frombuffer(bytearray(idx_bytes), dtype=np.uint32)
                if np.any(engram_indices >= ENG_NUM_BUCKETS):
                    raise HTTPException(status_code=400, detail="Invalid engram index")
                val_buf = bytearray(val_bytes)
                if fmt == "fp16": engram_values = torch.frombuffer(val_buf, dtype=torch.uint16).view(torch.float16).to(torch.bfloat16)
                elif fmt == "fp32": engram_values = torch.frombuffer(val_buf, dtype=torch.float32).to(torch.bfloat16)
                elif fmt == "fp8": engram_values = torch.frombuffer(val_buf, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
                else: engram_values = torch.frombuffer(val_buf, dtype=torch.uint16).view(torch.bfloat16)
                engram_values = engram_values.reshape(-1, MODEL_CONFIG["dim"]).contiguous()
                del val_buf
                if len(np.unique(engram_indices)) != len(engram_indices):
                    raise HTTPException(status_code=400, detail="Duplicate engram indices")
                _ev32 = engram_values.float()
                if not bool(torch.isfinite(_ev32).all()):
                    ban_subnet(client_ip, "NaN/Inf engram delta")
                    raise HTTPException(status_code=400, detail="Invalid engram delta (NaN/Inf)")
                _rn = _ev32.norm(dim=1); _lim = HARDEN["ENGRAM_ROW_NORM_MAX"]
                if bool((_rn > _lim).any()):
                    log.warning(f"⚠️ {client_key}: clipping {int((_rn > _lim).sum())} engram rows")
                    engram_values = (_ev32 * torch.clamp(_lim / _rn.clamp_min(1e-12), max=1.0).unsqueeze(1)).to(torch.bfloat16).contiguous()
                del _ev32, _rn
            task_id = body.get("taskId")
            task_meta = db["tasks"].get(task_id)
            if task_meta is None: raise HTTPException(status_code=404, detail="Task expired")
            if task_meta.get("round") != round_state["current_round"]:
                log.info(f"🚫 {client_key}: stale submission — task round {task_meta.get('round')}, current round {round_state['current_round']}")
                return _duplicate_response("stale_round")
            cc_used = bool(body.get("crossCheckUsed", False))
            cc_actual_seed = body.get("actualSeed")
            cc_actual_steps = body.get("actualSteps")
            cc_actual_bs = body.get("actualBatchSize")
            cc_actual_opt = (body.get("actualOptimizer") or "").lower()
            tgt_autocast = task_meta.get("targetAutocast")
            if tgt_autocast and tgt_autocast != "any" and cc_used:
                hw = client_hardware.get(client_key) or {}
                native_ac = (hw.get("autocast") or "any").lower()
                if native_ac in ("bf16", "fp32") and native_ac != tgt_autocast:
                    log.info(f"ℹ️ {client_key}: pool forced {tgt_autocast}, client native {native_ac}")
            tgt_opt = task_meta.get("targetOptimizer") or "any"
            if cc_actual_opt and tgt_opt != "any" and cc_actual_opt != tgt_opt:
                log.warning(f"⚠️ {client_key}: optimizer mismatch actual={cc_actual_opt} target={tgt_opt}")
            cc_matches_ticket = False
            if cc_used and CROSS_CHECK["ENABLED"]:
                tgt_steps = task_meta.get("targetLocalSteps")
                tgt_bs = task_meta.get("targetBatchSize")
                tgt_seed = task_meta.get("targetSeed")
                try:
                    cc_matches_ticket = (
                        tgt_steps is not None and cc_actual_steps is not None and int(cc_actual_steps) == int(tgt_steps)
                        and tgt_bs is not None and cc_actual_bs is not None and int(cc_actual_bs) == int(tgt_bs)
                        and tgt_seed is not None and cc_actual_seed is not None and int(cc_actual_seed) == int(tgt_seed)
                    )
                except (TypeError, ValueError): cc_matches_ticket = False
            if HARDEN["MIN_TRAIN_TIME_S"] > 0 and not cc_matches_ticket:
                t = task_tickets.get(client_key)
                if t and t.get("taskId") == task_id:
                    elapsed = time.time() - t["createdAt"]
                    if elapsed < HARDEN["MIN_TRAIN_TIME_S"]:
                        retry_after = int(HARDEN["MIN_TRAIN_TIME_S"] - elapsed) + 1
                        log.warning(f"🚫 {client_key}: submit after {elapsed:.0f}s < {HARDEN['MIN_TRAIN_TIME_S']}s")
                        raise HTTPException(status_code=429, detail=f"Minimum training time not met: {elapsed:.0f}s",
                            headers={"Retry-After": str(retry_after), "X-Status": "wait"})
            if HARDEN["MIN_SUBMIT_INTERVAL_S"] > 0 and not cc_matches_ticket:
                last = client_last_submit.get(client_key, 0)
                if last > 0:
                    since = time.time() - last
                    if since < HARDEN["MIN_SUBMIT_INTERVAL_S"]:
                        retry_after = int(HARDEN["MIN_SUBMIT_INTERVAL_S"] - since) + 1
                        raise HTTPException(status_code=429, detail=f"Minimum submit interval not met: {since:.0f}s",
                            headers={"Retry-After": str(retry_after), "X-Status": "wait"})
            shard_id_for_meta = task_meta.get("shardId")
            try: rank = int(body.get("loraRank", LORA_CONFIG["FULL_RANK"]) or LORA_CONFIG["FULL_RANK"])
            except Exception: rank = LORA_CONFIG["FULL_RANK"]
            if rank <= 0: rank = LORA_CONFIG["FULL_RANK"]
            try: local_steps = int(body.get("localSteps", 1) or 1)
            except Exception: local_steps = 1
            if local_steps <= 0: local_steps = 1
            local_steps = min(local_steps, HARDEN["MAX_LOCAL_STEPS"])
            loss_raw = body.get("loss"); loss_float = None
            try:
                if loss_raw is not None:
                    loss_float = float(loss_raw)
                    if math.isnan(loss_float) or math.isinf(loss_float): loss_float = None
            except Exception: loss_float = None
            max_allowed = compute_max_allowed_norm(local_steps, rank)
            abs_max = max(FL_CONFIG["DELTA_NORM_ABSOLUTE_MAX"], max_allowed * 3)
            if delta_norm > abs_max:
                ban_subnet(client_ip, f"delta norm {delta_norm:.0f} > {abs_max:.0f}")
                raise HTTPException(status_code=400, detail=f"Delta too large ({delta_norm:.2f})")
            ts = int(time.time() * 1000)
            safe_key = client_key.replace(":", "_").replace(".", "_")
            final_delta_path = str(DELTA_TEMP_DIR / f"delta_{safe_key}_{ts}.bin")
            delta_metadata = {"nodeKey": client_key, "nodeIp": client_ip,
                "delta_norm": delta_norm, "rank": rank, "loss": loss_float,
                "localSteps": local_steps, "tokensProcessed": _bounded_tokens(body, client_key, local_steps),
                "normalized": False, "submittedAt": int(time.time() * 1000), "weightFormat": "bf16",
                "shardId": shard_id_for_meta, "taskId": task_id, "crossCheckUsed": cc_used,
                "actualSeed": cc_actual_seed, "actualSteps": cc_actual_steps,
                "actualBatchSize": cc_actual_bs, "actualOptimizer": cc_actual_opt}
            if delta_norm > max_allowed:
                scale_factor = max_allowed / delta_norm
                delta_metadata["normalized"] = True
                with open(temp_delta_path, 'rb') as in_f, open(final_delta_path, 'wb') as out_f:
                    mb = json.dumps(delta_metadata).encode('utf-8')
                    out_f.write(struct.pack('<I', len(mb))); out_f.write(mb)
                    offset = 0
                    while offset < MODEL_SIZE:
                        rp = min(10_000_000, MODEL_SIZE - offset)
                        cb = in_f.read(rp * bpp)
                        if not cb or len(cb) < rp * bpp: raise HTTPException(status_code=400, detail="Incomplete delta during normalization")
                        ct = _decode_delta_chunk(bytearray(cb), fmt); ct.mul_(scale_factor)
                        out_f.write(_encode_float32_to_bf16_bytes(ct)); offset += rp; del ct
                    if engram_indices is not None and len(engram_indices) > 0:
                        out_f.write(struct.pack('<I', len(engram_indices)))
                        out_f.write(engram_indices.tobytes())
                        es = (engram_values.float() * scale_factor).to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
                        out_f.write(es)
                    else: out_f.write(struct.pack('<I', 0))
            else:
                with open(temp_delta_path, 'rb') as in_f, open(final_delta_path, 'wb') as out_f:
                    mb = json.dumps(delta_metadata).encode('utf-8')
                    out_f.write(struct.pack('<I', len(mb))); out_f.write(mb)
                    offset = 0
                    while offset < MODEL_SIZE:
                        rp = min(10_000_000, MODEL_SIZE - offset)
                        cb = in_f.read(rp * bpp)
                        if not cb or len(cb) < rp * bpp: raise HTTPException(status_code=400, detail="Incomplete delta during copy")
                        if fmt == "bf16": out_f.write(cb)
                        else:
                            ct = _decode_delta_chunk(bytearray(cb), fmt)
                            out_f.write(_encode_float32_to_bf16_bytes(ct)); del ct
                        offset += rp
                    if engram_indices is not None and len(engram_indices) > 0:
                        out_f.write(struct.pack('<I', len(engram_indices)))
                        out_f.write(engram_indices.tobytes())
                        out_f.write(engram_values.contiguous().view(torch.uint16).numpy().tobytes())
                    else: out_f.write(struct.pack('<I', 0))
            if temp_delta_path and os.path.exists(temp_delta_path): os.unlink(temp_delta_path)
            temp_delta_path = None
            submitted_path = None
            pool_for_completion = None
            with pool_lock:
                if round_state.get("is_aggregating", False):
                    log.warning(f"🚫 {client_key}: round ended during upload; discarding")
                    return _duplicate_response("round_ended_during_upload")
                _q = _check_submission_quota(client_key, client_ip)
                if _q:
                    log.warning(f"🚫 {client_key} over quota ({_q}); discarded")
                    return _duplicate_response(_q)
                pool_id = task_meta.get("poolId")
                pool = pools.get(pool_id) if pool_id else None
                if not pool or pool["status"] != "training":
                    log.warning(f"🚫 {client_key}: pool {pool_id} missing/not training")
                    return _duplicate_response("pool_gone")
                if client_key in pool["deltas"]:
                    log.warning(f"🚫 {client_key}: already submitted to {pool_id}")
                    return _duplicate_response("already_submitted")
                _count_submission(client_key, client_ip, task_id)
                pool["deltas"][client_key] = final_delta_path
                submitted_path = final_delta_path
                final_delta_path = None
                client_pool_map.pop(client_key, None)
                entry = round_state["participating_clients"].get(client_key)
                if entry is None:
                    round_state["participating_clients"][client_key] = {
                        "joined": time.time(), "submitted": True, "ultimatum_sent": False, "poolId": pool_id}
                else:
                    entry["submitted"] = True; entry["ultimatum_sent"] = False; entry["poolId"] = pool_id
                n_sub = len(pool["deltas"]); n_need = len(pool["clients"])
                log.info(f"📥 {client_key} → pool {pool_id} ({n_sub}/{n_need}, chunk={pool['chunk_idx']}, "
                         f"norm={delta_norm:.2f}, cc={cc_used}, opt={cc_actual_opt or '?'}, tokens={delta_metadata['tokensProcessed']:,})")
                if _maybe_complete_pool(pool): pool_for_completion = pool
                client_last_submit[client_key] = time.time()
            if pool_for_completion is not None: _spawn_pool_completion(pool_for_completion)
            with stats_lock:
                global_stats["totalTokens"] += delta_metadata["tokensProcessed"]
                global_stats["totalSteps"] += local_steps
                if loss_float is not None and loss_float > 0:
                    global_stats["lastLoss"] = loss_float
                    global_stats["lossHistory"].append(loss_float)
                    global_stats["lossHistory"] = global_stats["lossHistory"][-100:]
                lora_stats["total_submissions"] += 1
                lora_stats["rank_distribution"][str(rank)] = lora_stats["rank_distribution"].get(str(rank), 0) + 1
                lora_stats["last_update"] = datetime.now(timezone.utc).isoformat()
            auth_user = get_current_user(request)
            if auth_user:
                with users_lock:
                    ur = users_db["users"].get(auth_user["uuid"])
                    if ur:
                        ur["stats"]["totalTokens"] = ur["stats"].get("totalTokens", 0) + delta_metadata["tokensProcessed"]
                        ur["stats"]["totalSteps"] = ur["stats"].get("totalSteps", 0) + local_steps
                        save_users_db()
            with _commit_lock:
                save_global_state(); update_and_save_status(push_hf=False)
            return {"status": "success", "globalStep": global_step, "normalized": delta_metadata["normalized"],
                "round": round_state["current_round"], "loss": loss_float, "savedPath": submitted_path}
    finally:
        for p in (temp_delta_path, final_delta_path):
            if p and os.path.exists(p):
                try: os.unlink(p)
                except Exception: pass

# ============ HARDENING HELPERS ============
def _free_disk_bytes() -> int:
    try: return shutil.disk_usage(str(CHECKPOINT_DIR)).free
    except Exception: return 1 << 60
def _check_upload_capacity():
    if _free_disk_bytes() < HARDEN["MIN_FREE_DISK_UPLOAD_BYTES"]:
        raise HTTPException(status_code=503, detail="Storage full, try later", headers={"Retry-After": "600"})
def _pending_cap_reached() -> bool:
    cap = HARDEN["MAX_PENDING_DELTAS"]
    return cap > 0 and len(round_state["pending_deltas"]) >= cap
def _check_submission_quota(client_key: str, client_ip: str) -> Optional[str]:
    rnd = round_state["current_round"]
    lc, ls = HARDEN["MAX_SUBMISSIONS_PER_CLIENT_ROUND"], HARDEN["MAX_SUBMISSIONS_PER_SUBNET_ROUND"]
    if lc > 0 and submit_counts_client.get((rnd, client_key), 0) >= lc: return "client_round_limit"
    if ls > 0 and submit_counts_subnet.get((rnd, get_subnet(client_ip)), 0) >= ls: return "subnet_round_limit"
    return None
def _count_submission(client_key: str, client_ip: str, task_id: Optional[str]):
    rnd = round_state["current_round"]
    submit_counts_client[(rnd, client_key)] = submit_counts_client.get((rnd, client_key), 0) + 1
    sn = (rnd, get_subnet(client_ip)); submit_counts_subnet[sn] = submit_counts_subnet.get(sn, 0) + 1
    if len(submit_counts_client) > 5000:
        for d in (submit_counts_client, submit_counts_subnet):
            for k in [k for k in d if k[0] < rnd]: del d[k]
    t = task_tickets.get(client_key)
    if t and t.get("taskId") == task_id: task_tickets.pop(client_key, None)
def _duplicate_response(reason: str) -> dict:
    return {"status": "success", "duplicate": True, "reason": reason, "globalStep": global_step,
            "normalized": False, "round": round_state["current_round"], "loss": None, "savedPath": None}
def _issue_ticket(client_key: str, task_id: str):
    with shard_lock:
        t = task_tickets.get(client_key)
        if t and t.get("taskId") == task_id: return
        task_tickets[client_key] = {"taskId": task_id, "createdAt": time.time(), "round": round_state["current_round"]}
        if len(task_tickets) > 20000:
            for k in sorted(task_tickets, key=lambda k: task_tickets[k]["createdAt"])[:5000]:
                task_tickets.pop(k, None)
def _bounded_tokens(body: dict, client_key: str, local_steps: int) -> int:
    try: claimed = max(0, int(body.get("tokensProcessed", 0) or 0))
    except Exception: claimed = 0
    t = task_tickets.get(client_key)
    if t and t.get("taskId") == body.get("taskId"):
        age = max(1.0, time.time() - t["createdAt"])
    else:
        if HARDEN["REQUIRE_TASK_TICKET"]: raise HTTPException(status_code=403, detail="No task issued to this client")
        age = ROUND_CONFIG["MAX_ROUND_HOURS"] * 3600.0
    cap = int(min(local_steps * HARDEN["MAX_TOKENS_PER_STEP"], age * HARDEN["MAX_TOKENS_PER_SEC"]))
    if claimed > cap:
        log.warning(f"⚠️ {client_key}: tokens {claimed:,} clamped to {cap:,}")
        claimed = cap
    return claimed
def _guarded_process(path: str, client_ip: str, client_key: str, request: Request, init_meta: Optional[dict]):
    if not upload_slots.acquire(timeout=HARDEN["UPLOAD_SLOT_WAIT_S"]):
        raise HTTPException(status_code=503, detail="Server busy", headers={"Retry-After": "120"})
    try: return process_compressed_payload(path, client_ip, client_key, request, init_meta)
    finally: upload_slots.release()
def _stream_f32_file(path, n_params: int, fmt: str, piece: int = 10_000_000):
    enc = float32_to_fp8_array if fmt == "fp8" else float32_to_fp16_array if fmt == "fp16" else float32_to_bf16_array
    with open(path, "rb") as f:
        left = n_params
        while left > 0:
            a = np.fromfile(f, dtype=np.float32, count=min(piece, left))
            if a.size == 0: break
            left -= a.size
            yield enc(a).tobytes()
def _log_hardening_summary():
    log.info("🛡️ Hardening: proxies=%s | quota client/subnet=%s/%s | krum f=%s ram=%dMB | "
             "slots=%s | loss=%s | cross-check=%s" % (
        [str(n) for n in HARDEN["TRUSTED_PROXY_NETS"]] or "NONE (spoofable)",
        HARDEN["MAX_SUBMISSIONS_PER_CLIENT_ROUND"], HARDEN["MAX_SUBMISSIONS_PER_SUBNET_ROUND"],
        HARDEN["KRUM_F"], HARDEN["KRUM_RAM_BYTES"] // (1024 * 1024),
        HARDEN["MAX_CONCURRENT_UPLOADS"], HARDEN["LOSS_CHECK_MODE"],
        "ON" if CROSS_CHECK["ENABLED"] else "OFF"))
    if not HARDEN["TRUSTED_PROXY_NETS"]: log.warning("⚠️ CROWDGPT_TRUSTED_PROXIES unset: client IPs from X-Forwarded-For are spoofable.")
def _log_cross_check_summary():
    if not CROSS_CHECK["ENABLED"]: log.info(f"🔍 Cross-check: DISABLED"); return
    log.info(f"🔍 Cross-check: ENABLED — pool={MATCHMAKING['POOL_SIZE']} (MANDATORY)")
    log.info(f"   → targetBatchSize={CROSS_CHECK['TARGET_BATCH_SIZE']} targetLocalSteps={CROSS_CHECK['TARGET_LOCAL_STEPS']}")
    log.info(f"   → targetOptimizer={CROSS_CHECK['TARGET_OPTIMIZER']}")
    log.info(f"   → distance threshold={CROSS_CHECK['DISTANCE_THRESHOLD_RATIO']}x (2% absolute floor)")
    log.info(f"   → ban at {CROSS_CHECK['BAN_THRESHOLD']} subnet strikes")
    log.info(f"   → BACKEND exact; OPTIM normalize(adamw32→adamw_fp32, adamw8bit) + wildcard; OPTIM mix fp32/8bit allowed")
    log.info(f"   → ASYMMETRIC autocast: FP32 pools accept any client (forced fp32), BF16 pools accept bf16 only")
    log.info(f"   → MATCHMAKING PRIORITY: FP32 forming pools > BF16 forming pools > new pool (v2.3)")
    log.info(f"   → SOLO RE-MATCHING: solo forming pools re-scanned on every poll (v2.3)")
    log.info(f"   → solo BF16 pools dissolve after {MATCHMAKING['BF16_SOLO_DISSOLVE_S']}s")
    log.info(f"   → solo FP32 pools dissolve after {MATCHMAKING['FP32_SOLO_DISSOLVE_S']}s")

def process_compressed_payload(compressed_path: str, client_ip: str, client_key: str, request: Request, init_meta: Optional[dict] = None):
    temp_decompressed = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dec") as dec_file:
            temp_decompressed = dec_file.name
        dec_size = 0
        with open(compressed_path, 'rb') as f:
            magic = f.read(2); f.seek(0)
            with open(temp_decompressed, 'wb') as dec:
                if len(magic) >= 2 and magic[0] == 0x1f and magic[1] == 0x8b:
                    with gzip.open(f, 'rb') as gz:
                        while True:
                            chunk = gz.read(1024 * 1024)
                            if not chunk: break
                            dec_size += len(chunk)
                            if dec_size > MAX_DECOMPRESSED_BYTES:
                                raise HTTPException(status_code=413, detail="Decompressed payload too large")
                            dec.write(chunk)
                else:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk: break
                        dec_size += len(chunk)
                        if dec_size > MAX_DECOMPRESSED_BYTES:
                            raise HTTPException(status_code=413, detail="Payload too large")
                        dec.write(chunk)
        return process_decompressed_payload(temp_decompressed, client_ip, client_key, request, init_meta)
    finally:
        if temp_decompressed and os.path.exists(temp_decompressed):
            try: os.unlink(temp_decompressed)
            except Exception: pass

# ============ CROSS-CHECK COHORT ============
def _cross_check_cohort(paths: List[str], model_size: int):
    n = len(paths)
    if n == 0: return [], [], [], []
    if n == 1: return paths, [1.0], [], []
    dfs = []
    unreadable = []
    for p in paths:
        try: dfs.append(open_delta(p, model_size))
        except Exception as e: log.error(f"Cannot open delta {p}: {e}"); unreadable.append(p)
    if len(dfs) < 2:
        if len(dfs) == 1:
            log.warning(f"📥 Cross-check: only 1 readable delta — keeping it, discarding {len(unreadable)}")
            return [dfs[0].path], [1.0], [], unreadable
        return [], [], [], paths
    if len(dfs) == 2:
        toks = [max(1, d.tokens) for d in dfs]
        total = sum(toks)
        weights = [t/total for t in toks]
        log.info(f"📥 Cross-check: 2 deltas (no 3-way possible, no trust)")
        try:
            dist_sq, bad, sq_norms = streaming_pairwise_sq_l2(dfs, max_ram_bytes=HARDEN["KRUM_RAM_BYTES"])
            if bad[0] or bad[1]:
                log.warning(f"⚠️ Cross-check: NaN/Inf in n=2 cohort — discarding both")
                return [], [], [], [d.path for d in dfs] + unreadable
            d = math.sqrt(max(0.0, float(dist_sq[0, 1].item())))
            min_norm = math.sqrt(max(1.0, min(float(sq_norms[0].item()), float(sq_norms[1].item()))))
            if d > 0.10 * min_norm:
                log.warning(f"⚠️ Cross-check: 2-client divergence {d:.2e} > 10% of norm {min_norm:.2e} — DISCARDING BOTH")
                return [], [], [], [d.path for d in dfs] + unreadable
        except Exception as e:
            log.error(f"n=2 distance check failed: {e}")
            return [], [], [], [d.path for d in dfs] + unreadable
        return [d.path for d in dfs], weights, [], unreadable
    try:
        dist_sq, bad, sq_norms = streaming_pairwise_sq_l2(dfs, max_ram_bytes=HARDEN["KRUM_RAM_BYTES"])
    except Exception as e:
        log.error(f"Cross-check distance computation failed: {e}; falling back to token-weight")
        toks = [max(1, d.tokens) for d in dfs]
        total = sum(toks)
        return [d.path for d in dfs], [t/total for t in toks], [], unreadable
    good = [i for i in range(len(dfs)) if not bad[i]]
    bad_idx = [i for i in range(len(dfs)) if bad[i]]
    for i in bad_idx: log.warning(f"🚩 Cross-check: NaN/Inf in delta {dfs[i].path}")
    if len(good) < 2:
        log.warning(f"🚩 Cross-check: only {len(good)}/{len(dfs)} good — flagging bad, discarding good")
        return [], [], [dfs[i].path for i in bad_idx] + unreadable, [dfs[i].path for i in good]
    core_i, core_j, core_dist_sq = None, None, float('inf')
    for ai in range(len(good)):
        for bi in range(ai + 1, len(good)):
            i, j = good[ai], good[bi]
            d = float(dist_sq[i, j].item())
            if d < core_dist_sq: core_dist_sq = d; core_i, core_j = i, j
    core_dist = math.sqrt(max(0.0, core_dist_sq))
    min_norm = math.sqrt(max(1.0, min(float(sq_norms[i].item()) for i in good)))
    abs_floor = 0.02 * min_norm
    ratio = CROSS_CHECK["DISTANCE_THRESHOLD_RATIO"]
    threshold = max(ratio * core_dist, abs_floor)
    kept_idx = [core_i, core_j]
    flagged_idx = list(bad_idx)
    for k in good:
        if k == core_i or k == core_j: continue
        d_ik = math.sqrt(max(0.0, float(dist_sq[core_i, k].item())))
        d_jk = math.sqrt(max(0.0, float(dist_sq[core_j, k].item())))
        min_k = min(d_ik, d_jk)
        if min_k > threshold:
            flagged_idx.append(k)
            log.warning(f"🚩 Cross-check outlier: {dfs[k].path} (min_dist_to_core={min_k:.4e} > threshold={threshold:.4e}, core_dist={core_dist:.4e}, floor={abs_floor:.4e})")
        else: kept_idx.append(k)
    if not any(f not in bad_idx for f in flagged_idx):
        log.info(f"✅ Cross-check: all {len(kept_idx)} kept (core_dist={core_dist:.4e}, threshold={threshold:.4e})")
    kept_paths = [dfs[i].path for i in kept_idx]
    flagged_paths = [dfs[i].path for i in flagged_idx]
    toks = [max(1, dfs[i].tokens) for i in kept_idx]
    total = sum(toks)
    weights = [t/total for t in toks] if total > 0 else [1.0/len(kept_idx)]*len(kept_idx)
    return kept_paths, weights, flagged_paths, unreadable

# ============ AGGREGATION (WITH ANTI-HALVING SAFEGUARDS) ============
_prev_weight_norm = {"v": 0.0}

def _compute_norm_server(path: str, model_size: int) -> float:
    try: df = open_delta(path, model_size)
    except Exception: return 0.0
    try:
        with open(path, 'rb') as f:
            f.seek(df.base_offset)
            total_sq = 0.0; remaining = model_size; chunk = 4_194_304
            while remaining > 0:
                read = min(chunk, remaining)
                raw = f.read(read * 2)
                if len(raw) < read * 2: return 0.0
                arr = torch.from_numpy(np.frombuffer(raw, dtype=np.int16).copy()).view(torch.bfloat16).to(torch.float32)
                if not bool(torch.isfinite(arr).all()): return 0.0
                total_sq += float((arr * arr).sum().item())
                remaining -= read
    except Exception: return 0.0
    return math.sqrt(max(0.0, total_sq))

def apply_round_aggregation(ready_pool_list: List[dict]):
    global global_weights, global_step
    if not ready_pool_list:
        log.warning("⚠️ No pool results to aggregate"); return
    lr = _server_lr(global_step)
    wd = FL_CONFIG["WEIGHT_DECAY"]
    cap = FL_CONFIG["MAX_DELTA_APPLY_NORM"]
    mxn = FL_CONFIG["MAX_WEIGHT_NORM"]
    step_before = global_step
    n_pools = len(ready_pool_list)
    pool_norm = 1.0 / n_pools if n_pools > 0 else 1.0
    log.info(f"🧮 Merging {n_pools} pool result(s) "
             f"(lr={lr:.4f}, pool_norm={pool_norm:.4f}, wd={wd}, delta_cap={cap}, max_norm={mxn})")
    if not _mem_guard(1.5, "pool aggregation"):
        log.error("❌ Skipping aggregation — memory pressure. Round advances WITHOUT model update.")
        for pool_result in ready_pool_list:
            for path, _weight in pool_result.get("deltas", []):
                try: os.remove(path)
                except Exception: pass
        return
    with shard_lock:
        pre_norm_sq = 0.0; cs = 10_000_000
        for i in range(0, MODEL_SIZE, cs):
            chunk = global_weights[i:i + cs].float()
            pre_norm_sq += torch.sum(chunk * chunk).item()
            del chunk
        pre_norm = math.sqrt(pre_norm_sq)
        global_weights.mul_(1.0 - lr * wd)
    engram_accumulator = {}
    row_lim = HARDEN["ENGRAM_ROW_NORM_MAX"]
    total_applied = 0; total_tokens = 0; clipped_count = 0; rejected_count = 0
    for pool_result in ready_pool_list:
        pool_id = pool_result["pool_id"]
        for path, weight in pool_result["deltas"]:
            try:
                md = _delta_metadata(path)
                if md is None:
                    log.error(f"🚨 REJECTING {path}: metadata missing/corrupt")
                    rejected_count += 1
                    try: os.remove(path)
                    except Exception: pass
                    continue
                dnorm = float(md.get("delta_norm", 0.0))
                if not math.isfinite(dnorm) or dnorm <= 0:
                    log.error(f"🚨 REJECTING {path}: delta_norm={dnorm}")
                    rejected_count += 1
                    try: os.remove(path)
                    except Exception: pass
                    continue
                try:
                    df = open_delta(path, MODEL_SIZE)
                    actual_norm = _compute_norm_server(path, MODEL_SIZE)
                    if actual_norm <= 0:
                        log.error(f"🚨 REJECTING {path}: actual norm=0 or NaN/Inf")
                        rejected_count += 1
                        try: os.remove(path)
                        except Exception: pass
                        continue
                    ratio_check = actual_norm / dnorm
                    if ratio_check < 0.8 or ratio_check > 1.25:
                        log.error(f"🚨 REJECTING {path}: metadata={dnorm:.2f} vs actual={actual_norm:.2f}")
                        rejected_count += 1
                        try: os.remove(path)
                        except Exception: pass
                        continue
                except Exception as e:
                    log.error(f"🚨 REJECTING {path}: open_delta failed: {e}")
                    rejected_count += 1
                    try: os.remove(path)
                    except Exception: pass
                    continue
                scale = (cap / dnorm) if (dnorm > cap > 0) else 1.0
                if scale < 1.0: clipped_count += 1
                if weight > 0:
                    # was: stream_add_base_delta(global_weights, df, lr * weight * scale)
                    with shard_lock: stream_add_base_delta(global_weights, df, lr * pool_norm * weight * scale)
                    eng_idx, eng_val = read_engram_tail(df, MODEL_CONFIG["dim"])
                    if eng_idx is not None and len(eng_idx) > 0:
                        evf = eng_val.float()
                        rn = evf.norm(dim=1)
                        ok = torch.isfinite(rn)
                        rscale = torch.where(ok, torch.clamp(row_lim / rn.clamp_min(1e-12), max=1.0), torch.zeros_like(rn))
                        evf = (torch.where(ok.unsqueeze(1), evf, torch.zeros_like(evf)) * rscale.unsqueeze(1))
                        for idx, val, okk in zip(eng_idx.tolist(), evf, ok.tolist()):
                            if not okk or not (0 <= idx < ENG_NUM_BUCKETS): continue
                            if idx not in engram_accumulator:
                                engram_accumulator[idx] = [torch.zeros(MODEL_CONFIG["dim"], dtype=torch.float32), 0.0]
                            engram_accumulator[idx][0] += val * weight * scale
                            engram_accumulator[idx][1] += weight
                total_applied += 1
                total_tokens += df.tokens
            except Exception as e:
                log.error(f"Failed to apply {path} (pool {pool_id}): {e}")
                import traceback; traceback.print_exc()
            finally:
                try: os.remove(path)
                except Exception: pass
                gc.collect()
    with shard_lock:
        norm_sq = 0.0; cs = 10_000_000
        for i in range(0, MODEL_SIZE, cs):
            chunk = global_weights[i:i + cs].float()
            norm_sq += torch.sum(chunk * chunk).item()
            del chunk
        norm = math.sqrt(norm_sq)
        if pre_norm > 0 and norm > 0:
            ratio = norm / pre_norm
            if ratio < 0.70 or ratio > 1.50:
                log.error(f"🚨🚨 AGGREGATION SANITY FAIL: norm {pre_norm:.1f} → {norm:.1f} (ratio={ratio:.3f}). Reverting from disk, NOT advancing step.")
                load_model_weights(); load_engram_weights()
                with stats_lock: global_step = step_before
                for pool_result in ready_pool_list:
                    for path, _w in pool_result.get("deltas", []):
                        try: os.remove(path)
                        except Exception: pass
                return
            else:
                log.info(f"✅ Norm sanity: {pre_norm:.1f} → {norm:.1f} (ratio={ratio:.3f})")
        if norm > mxn:
            global_weights.mul_(mxn / norm)
            log.warning(f"⚠️ Weight norm {norm:.2f} clipped to {mxn}")
        if engram_accumulator:
            log.info(f"🧠 Updating {len(engram_accumulator)} engram buckets")
            ftw = HARDEN["ENGRAM_FULL_TRUST_WEIGHT"]
            for idx, (vs, ws) in engram_accumulator.items():
                if ws > 0:
                    trust = min(1.0, ws / ftw) if ftw > 0 else 1.0
                    avg = vs / ws
                    sp = idx * MODEL_CONFIG["dim"]; ep = sp + MODEL_CONFIG["dim"]
                    global_engram_weights[sp:ep] += avg * (lr * pool_norm * trust)   # ← add pool_norm
            del engram_accumulator
            gc.collect()
        with stats_lock: global_step += 1
    log.info(f"✅ SGD step {global_step}: applied {total_applied}, rejected {rejected_count}, "
             f"tokens={total_tokens:,}, norm={norm:.2f}, lr={lr:.4f}, clipped={clipped_count}/{total_applied}")
    if HARDEN["LOSS_CHECK_MODE"] == "off": handle_checkpoints()
    gc.collect()

# ============ FASTAPI + /fl/* ENDPOINTS ============
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="CrowdGPT Continuous FL Coordinator (v2.3)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
    expose_headers=["X-Task-Id", "X-Shard-Id", "X-Chunk-Idx", "X-Global-Step",
                    "X-Model-Size", "X-Quantized", "X-Round", "X-Status",
                    "X-Pool-Id", "X-Pool-Fill"],
)

def _task_alive_for(task_id: str, client_key: str) -> bool:
    meta = db["tasks"].get(task_id)
    if not meta or meta.get("round") != round_state["current_round"]: return False
    with pool_lock:
        pool = pools.get(meta.get("poolId"))
        return bool(pool) and pool["status"] == "training" and client_key in pool["clients"]

@app.get("/fl/heartbeat")
async def heartbeat(request: Request, taskId: Optional[str] = None):
    if _banned_reason(request):
        return Response(content='{"status":"blocked"}', media_type="application/json")
    client_key = get_client_key(request)
    active_clients[client_key] = time.time()
    if client_key in client_hardware:
        client_hardware[client_key]["last_seen"] = time.time()
    if not taskId:
        return Response(content='{"status":"ok"}', media_type="application/json")
    task = "active" if _task_alive_for(taskId, client_key) else "gone"
    return Response(content=json.dumps({"status": "ok", "task": task}), media_type="application/json")

@app.get("/fl/round_status")
async def get_round_status(request: Request):
    _reason = _banned_reason(request)
    if _reason: raise HTTPException(status_code=403, detail=_reason)
    client_key = get_client_key(request)
    active_clients[client_key] = time.time()
    with shard_lock:
        if client_key not in round_state["participating_clients"]:
            round_state["participating_clients"][client_key] = {"joined": time.time(), "submitted": False, "ultimatum_sent": False}
    in_cooldown = (time.time() - round_state["last_aggregation_time"]) < ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"]
    round_elapsed_hours = (time.time() - round_state["round_start_time"]) / 3600
    time_left = round_state["round_duration_hours"] - round_elapsed_hours
    client_max_round = round_state["round_duration_hours"] if time_left < 0.25 else round_elapsed_hours + ROUND_CONFIG["TARGET_SUBMISSION_HOURS"]
    with pool_lock:
        mm = {"poolSize": MATCHMAKING["POOL_SIZE"], "activePools": len(pools),
            "formingPools": sum(1 for p in pools.values() if p["status"] == "forming"),
            "trainingPools": sum(1 for p in pools.values() if p["status"] == "training"),
            "readyPools": len(ready_pools)}
    return {"current_round": round_state["current_round"], "round_elapsed_hours": round_elapsed_hours,
        "max_round_hours": client_max_round, "round_duration_hours": round_state["round_duration_hours"],
        "clients_in_round": len(round_state["participating_clients"]),
        "submitted_clients": sum(1 for c in round_state["participating_clients"].values() if c["submitted"]),
        "in_cooldown": in_cooldown, "global_step": global_step,
        "next_round_ready": not in_cooldown and round_state["last_aggregation_time"] > 0,
        "is_aggregating": round_state.get("is_aggregating", False),
        "force_end_at": round_state.get("force_end_at"),
        "hfDirectDownloads": bool(_build_hf_weights_metadata("bf16")),
        "cohortSize": MATCHMAKING["POOL_SIZE"], "maxCohortSize": MATCHMAKING["POOL_SIZE"],
        "matchmaking": mm}

@app.get("/fl/task")
async def get_task(request: Request, format: str = "bf16", skip_weights: bool = False):
    if round_state.get("is_aggregating", False):
        return Response(content=json.dumps({"status": "wait", "message": "Server aggregating"}).encode(),
                        media_type="application/json", headers={"X-Status": "wait"})
    _reason = _banned_reason(request)
    if _reason: raise HTTPException(status_code=403, detail=_reason)
    _gate = _version_gate(request)
    if _gate is not None: return _gate
    client_ip = get_client_ip(request)
    client_key = get_client_key(request)
    active_clients[client_key] = time.time()
    client_hardware[client_key] = {
        "backend":  (request.headers.get("X-Client-Backend")    or "").upper(),
        "bs":        int(request.headers.get("X-Client-Batch-Size") or 0),
        "vram_gb":   float(request.headers.get("X-Client-Vram-Gb")   or 0.0),
        "gpu":       request.headers.get("X-Client-Gpu-Name")      or "",
        "autocast": (request.headers.get("X-Client-Autocast")   or "any").lower(),
        "optim":    (request.headers.get("X-Client-Optim")      or "any").lower(),
        "last_seen": time.time(),
    }
    hw = client_hardware[client_key]
    # ── Force fp32 for legacy/divergent archs ──
    if _force_fp32_arch(hw.get("gpu") or ""):
        native_ac = hw["autocast"]
        hw["autocast"] = "fp32"
        log.info(f"⚙️ {client_key}: forcing fp32 (gpu='{hw['gpu'] or '?'}', arch={_gpu_arch_class(hw['gpu'])}, was={native_ac})")
    log.info(f"📡 /fl/task from {client_key} ({client_ip}) "
             f"backend={hw['backend']} bs={hw['bs']} vram={hw['vram_gb']:.1f}GB "
             f"autocast={hw['autocast']} optim={hw['optim']} "
             f"arch={_gpu_arch_class(hw['gpu'])} card={_gpu_card_class(hw['gpu'])}")
    if _free_disk_bytes() < HARDEN["MIN_FREE_DISK_TASK_BYTES"]:
        return Response(content=json.dumps({"status": "wait", "message": "Server storage full"}).encode(),
                        media_type="application/json", headers={"X-Status": "wait"})
    in_cooldown = (time.time() - round_state["last_aggregation_time"]) < ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"]
    if in_cooldown:
        wait_time = int(ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"] - (time.time() - round_state["last_aggregation_time"]))
        return Response(content=json.dumps({"status": "wait", "message": f"Aggregating, wait {wait_time}s"}).encode(),
                        media_type="application/json", headers={"X-Status": "wait"})
    if _is_power_client(client_key):
        with pool_lock:      # atomic: nobody can slip into the pool between creation and solo activation
            pool = _power_solo_start(client_key, _matchmake(client_key, power=True))
    else:
        pool = _matchmake(client_key)
    if pool["status"] == "forming":
        return Response(
            content=json.dumps({"status": "wait",
                "message": f"Matching pool ({len(pool['clients'])}/{MATCHMAKING['POOL_SIZE']})",
                "poolId": pool["pool_id"], "poolFill": len(pool["clients"]),
                "poolSize": MATCHMAKING["POOL_SIZE"]}).encode(),
            media_type="application/json",
            headers={"X-Status": "wait", "X-Pool-Id": pool["pool_id"], "X-Pool-Fill": str(len(pool["clients"]))})
    task_id = pool["task_id"]
    _issue_ticket(client_key, task_id)
    metadata = _build_task_payload(pool, client_key)
    with shard_lock:
        entry = round_state["participating_clients"].get(client_key)
        if entry is None:
            round_state["participating_clients"][client_key] = {"joined": time.time(), "submitted": False,
                "ultimatum_sent": False, "poolId": pool["pool_id"]}
        else:
            entry["ultimatum_sent"] = False
            entry["poolId"] = pool["pool_id"]
    headers = {"X-Task-Id": task_id, "X-Round": str(round_state["current_round"]),
        "X-Global-Step": str(global_step), "X-Pool-Id": pool["pool_id"]}
    if skip_weights: return JSONResponse(metadata, headers=headers)
    return StreamingResponse(_stream_task_weights(metadata, format),
        media_type="application/octet-stream", headers=headers)

@app.post("/fl/submit")
async def submit_delta(request: Request):
    _reason = _banned_reason(request)
    if _reason: raise HTTPException(status_code=403, detail=_reason)
    _gate = _version_gate(request)
    if _gate is not None: return _gate
    if round_state.get("is_aggregating", False):
        raise HTTPException(status_code=503, detail="Server aggregating", headers={"Retry-After": "120"})
    client_ip = get_client_ip(request)
    client_key = get_client_key(request)
    active_clients[client_key] = time.time()
    _check_upload_capacity()
    temp_compressed = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin.gz") as tmp_file:
            temp_compressed = tmp_file.name
        size = 0
        try:
            with open(temp_compressed, 'wb') as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_DIRECT_BODY_BYTES:
                        raise HTTPException(status_code=413, detail="Too large; use /fl/submit/init")
                    f.write(chunk)
        except ClientDisconnect:
            raise HTTPException(status_code=499, detail="Client disconnected during upload")
        if size == 0: raise HTTPException(status_code=400, detail="Empty upload")
        return await run_in_threadpool(_guarded_process, temp_compressed, client_ip, client_key, request, None)
    finally:
        if temp_compressed and os.path.exists(temp_compressed):
            try: os.unlink(temp_compressed)
            except Exception: pass

@app.post("/fl/submit/init")
async def submit_init(request: Request):
    _reason = _banned_reason(request)
    if _reason: raise HTTPException(status_code=403, detail=_reason)
    _gate = _version_gate(request)
    if _gate is not None: return _gate
    if round_state.get("is_aggregating", False):
        raise HTTPException(status_code=503, detail="Server aggregating", headers={"Retry-After": "120"})
    client_ip = get_client_ip(request)
    client_key = get_client_key(request)
    active_clients[client_key] = time.time()
    _check_upload_capacity()
    try: data = await request.json()
    except ClientDisconnect: raise HTTPException(status_code=499, detail="Client disconnected before init body")
    except Exception: raise HTTPException(status_code=400, detail="Invalid JSON")
    required = ["taskId", "totalBytes", "chunkSize", "chunks", "checksum", "weightFormat", "hasEngram", "engramSparseCount"]
    for k in required:
        if k not in data: raise HTTPException(status_code=400, detail=f"Missing field: {k}")
    task_id = str(data["taskId"])
    if task_id not in db["tasks"]: raise HTTPException(status_code=404, detail="Task expired")
    try:
        total_bytes = int(data["totalBytes"]); chunk_size = int(data["chunkSize"])
        chunks = int(data["chunks"]); engram_sparse_count = int(data.get("engramSparseCount", 0) or 0)
    except Exception: raise HTTPException(status_code=400, detail="Invalid numeric init fields")
    if total_bytes <= 0 or total_bytes > MAX_TOTAL_UPLOAD_BYTES: raise HTTPException(400, "Invalid totalBytes")
    if chunk_size <= 0 or chunk_size > MAX_CHUNK_BYTES: raise HTTPException(400, f"Invalid chunkSize")
    if chunks <= 0 or chunks > MAX_CHUNKS: raise HTTPException(400, f"Invalid chunks")
    if (chunks - 1) * chunk_size >= total_bytes or chunks * chunk_size < total_bytes:
        raise HTTPException(400, "Inconsistent chunks/chunkSize/totalBytes")
    checksum = str(data["checksum"]).lower()
    if len(checksum) != 64 or not all(c in "0123456789abcdef" for c in checksum):
        raise HTTPException(400, "Invalid checksum")
    fmt = str(data.get("weightFormat", "bf16"))
    if fmt not in ("bf16", "fp16", "fp8", "fp32"): raise HTTPException(400, f"Unsupported format: {fmt}")
    has_engram = bool(data.get("hasEngram", False))
    if engram_sparse_count < 0 or engram_sparse_count > ENG_NUM_BUCKETS: raise HTTPException(400, "Invalid engramSparseCount")
    auth_user = get_current_user(request)
    upload_id = uuid.uuid4().hex
    up_dir = UPLOAD_DIR / upload_id
    up_dir.mkdir(parents=True, exist_ok=True)
    meta = {"taskId": task_id, "totalBytes": total_bytes, "chunkSize": chunk_size, "chunks": chunks,
        "checksum": checksum, "weightFormat": fmt, "hasEngram": has_engram,
        "engramSparseCount": engram_sparse_count, "loss": data.get("loss"),
        "localSteps": data.get("localSteps"), "tokensProcessed": data.get("tokensProcessed"),
        "loraRank": data.get("loraRank"), "isDelta": data.get("isDelta", True),
        "crossCheckUsed": data.get("crossCheckUsed", False),
        "actualSeed": data.get("actualSeed"), "actualSteps": data.get("actualSteps"),
        "actualBatchSize": data.get("actualBatchSize"),
        "actualOptimizer": data.get("actualOptimizer")}
    with uploads_lock:
        _active_uploads[upload_id] = {"dir": up_dir, "created": time.time(), "last_activity": time.time(),
            "status": "open", "client_key": client_key, "client_ip": client_ip,
            "user_uuid": auth_user["uuid"] if auth_user else None,
            "chunks": chunks, "chunk_size": chunk_size, "total_bytes": total_bytes,
            "checksum": checksum, "meta": meta, "received": set(),
            "chunk_sizes": {}, "bytes_received": 0}
    return {"uploadId": upload_id, "chunkSize": chunk_size, "chunks": chunks, "totalBytes": total_bytes}

@app.post("/fl/submit/chunk")
async def submit_chunk(request: Request, uploadId: str, index: int):
    _reason = _banned_reason(request)
    if _reason: raise HTTPException(status_code=403, detail=_reason)
    if not _valid_upload_id(uploadId): raise HTTPException(400, "Invalid uploadId")
    with uploads_lock:
        up = _active_uploads.get(uploadId)
        if not up: raise HTTPException(404, "Unknown uploadId")
        if up.get("status") != "open": raise HTTPException(409, "Upload not open")
        if index < 0 or index >= up["chunks"]: raise HTTPException(400, "Chunk out of range")
        cur_user = get_current_user(request)
        if up.get("user_uuid") is not None:
            if not cur_user or cur_user["uuid"] != up["user_uuid"]: raise HTTPException(403, "Owner mismatch")
        else:
            if up.get("client_key") and get_client_key(request) != up["client_key"]:
                raise HTTPException(403, "Owner mismatch")
        up_dir = up["dir"]; up["last_activity"] = time.time()
    part_tmp = up_dir / f"part_{index:06d}.tmp"
    part_final = up_dir / f"part_{index:06d}.bin"
    h = hashlib.sha256(); size = 0
    try:
        with open(part_tmp, 'wb') as f:
            try:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_CHUNK_BYTES: raise HTTPException(413, "Chunk too large")
                    f.write(chunk); h.update(chunk)
            except ClientDisconnect:
                raise HTTPException(status_code=499, detail="Client disconnected during chunk upload")
        if size == 0: raise HTTPException(400, "Empty chunk")
        expected_sha = request.headers.get("X-Chunk-Sha256")
        if expected_sha and expected_sha.lower() != h.hexdigest():
            raise HTTPException(400, "Checksum mismatch")
        os.replace(part_tmp, part_final)
        with uploads_lock:
            up = _active_uploads.get(uploadId)
            if not up: raise HTTPException(404, "Unknown uploadId")
            old = up["chunk_sizes"].get(index, 0)
            up["chunk_sizes"][index] = size
            up["bytes_received"] = up["bytes_received"] - old + size
            up["received"].add(index); up["last_activity"] = time.time()
        return {"ok": True, "index": index, "size": size,
                "receivedChunks": len(up["received"]), "totalChunks": up["chunks"]}
    finally:
        try:
            if part_tmp.exists(): part_tmp.unlink()
        except Exception: pass

@app.post("/fl/submit/finalize")
async def submit_finalize(request: Request):
    _reason = _banned_reason(request)
    if _reason: raise HTTPException(status_code=403, detail=_reason)
    if round_state.get("is_aggregating", False):
        raise HTTPException(status_code=503, detail="Round ended; retry next round", headers={"Retry-After": "120"})
    try: data = await request.json()
    except ClientDisconnect: raise HTTPException(status_code=499, detail="Client disconnected before finalize body")
    except Exception: raise HTTPException(400, "Invalid JSON")
    upload_id = data.get("uploadId")
    if not _valid_upload_id(upload_id): raise HTTPException(400, "Invalid uploadId")
    with uploads_lock:
        up = _active_uploads.get(upload_id)
        if not up: raise HTTPException(404, "Unknown uploadId")
        if up.get("status") != "open": raise HTTPException(409, "Upload not open")
        cur_user = get_current_user(request)
        if up.get("user_uuid") is not None:
            if not cur_user or cur_user["uuid"] != up["user_uuid"]: raise HTTPException(403, "Owner mismatch")
        else:
            if up.get("client_key") and get_client_key(request) != up["client_key"]:
                raise HTTPException(403, "Owner mismatch")
        if len(up["received"]) != up["chunks"]: raise HTTPException(400, "Missing chunks")
        up["status"] = "finalizing"; up["last_activity"] = time.time()
    client_ip = get_client_ip(request)
    client_key = up.get("client_key") or get_client_key(request)
    compressed_path = UPLOAD_DIR / f"{upload_id}_compressed.bin.gz"
    completed = False
    try:
        h = hashlib.sha256(); total = 0
        with open(compressed_path, 'wb') as out:
            for i in range(up["chunks"]):
                part = up["dir"] / f"part_{i:06d}.bin"
                if not part.exists(): raise HTTPException(400, f"Missing chunk {i}")
                with open(part, 'rb') as inp:
                    while True:
                        b = inp.read(4 * 1024 * 1024)
                        if not b: break
                        total += len(b)
                        if total > up["total_bytes"]: raise HTTPException(400, "Too large")
                        out.write(b); h.update(b)
        if total != up["total_bytes"]: raise HTTPException(400, "Size mismatch")
        if h.hexdigest() != up["checksum"]: raise HTTPException(400, "Checksum mismatch")
        result = await run_in_threadpool(_guarded_process, str(compressed_path), client_ip, client_key, request, up["meta"])
        completed = True
        return result
    except HTTPException: raise
    except Exception as e:
        log.error(f"❌ Finalize failed: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(400, f"Finalize failed: {e}")
    finally:
        try:
            if os.path.exists(compressed_path): os.unlink(compressed_path)
        except Exception: pass
        with uploads_lock:
            up2 = _active_uploads.get(upload_id)
            if up2:
                if completed:
                    try: shutil.rmtree(up2["dir"], ignore_errors=True)
                    except Exception: pass
                    del _active_uploads[upload_id]
                else:
                    up2["status"] = "open"; up2["last_activity"] = time.time()

# ============ AUTH ROUTES ============
@app.post("/auth/register")
async def register(request: Request):
    try: data = await request.json()
    except ClientDisconnect: raise HTTPException(499, "Client disconnected")
    except Exception: raise HTTPException(400, "Invalid JSON")
    username = data.get("username", "").strip(); email = data.get("email", "").strip()
    password = data.get("password", "")
    if not username or not (3 <= len(username) <= 20): raise HTTPException(400, "Username 3-20 chars")
    if not username.replace("_", "").replace("-", "").isalnum(): raise HTTPException(400, "Invalid chars")
    if contains_bad_word(username): raise HTTPException(400, "Prohibited words")
    if not email or "@" not in email: raise HTTPException(400, "Invalid email")
    if not password or len(password) < 6: raise HTTPException(400, "Password too short")
    with users_lock:
        for u in users_db["users"].values():
            if u["username_lower"] == username.lower():
                if _is_user_banned(u["uuid"]): raise HTTPException(403, "Username belongs to a suspended account")
                raise HTTPException(400, "Username taken")
            if u.get("email", "").lower() == email.lower(): raise HTTPException(400, "Email registered")
        user_uuid = str(uuid.uuid4()); salt = secrets.token_hex(16); token = generate_token()
        users_db["users"][user_uuid] = {
            "uuid": user_uuid, "username": username, "username_lower": username.lower(),
            "email": email, "password_hash": hash_password(password, salt), "salt": salt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stats": {"contributions": 0, "totalTokens": 0, "totalSteps": 0, "byzantine_flags": 0}}
        users_db["tokens"][token] = user_uuid
        save_users_db()
        log.info(f"✅ Registered {username}")
    return {"token": token, "uuid": user_uuid, "username": username}

@app.post("/auth/login")
async def auth_login(request: Request):
    try: data = await request.json()
    except ClientDisconnect: raise HTTPException(499, "Client disconnected")
    except Exception: raise HTTPException(400, "Invalid JSON")
    username = data.get("username", "").strip(); password = data.get("password", "")
    with users_lock:
        user = next((u for u in users_db["users"].values() if u["username_lower"] == username.lower()), None)
        if not user: raise HTTPException(401, "Invalid credentials")
        if _is_user_banned(user["uuid"]): raise HTTPException(403, "Account suspended")
        if hash_password(password, user["salt"]) != user["password_hash"]: raise HTTPException(401, "Invalid credentials")
        token = generate_token(); users_db["tokens"][token] = user["uuid"]; save_users_db()
        log.info(f"✅ Login {username}")
    return {"token": token, "uuid": user["uuid"], "username": user["username"]}

@app.get("/auth/me")
async def auth_me(request: Request):
    user = get_current_user(request)
    if not user: raise HTTPException(401, "Invalid or expired token")
    return {"uuid": user["uuid"], "username": user["username"], "email": user.get("email"),
            "stats": user["stats"], "trustScore": compute_trust_score(user["stats"]),
            "globalRank": compute_global_rank(user["uuid"])}

# ============ ADMIN ROUTES ============
@app.get("/api/checkpoints")
async def list_checkpoints(secret: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    checkpoints = []
    for f in CHECKPOINT_DIR.iterdir():
        if f.name.startswith("model_step_") and f.suffix == ".bin":
            try: checkpoints.append({"step": int(f.stem.split("_")[2]), "size": f.stat().st_size, "mtime": f.stat().st_mtime})
            except Exception: continue
    return {"current_step": global_step, "checkpoints": sorted(checkpoints, key=lambda x: x["step"], reverse=True)}

@app.get("/api/admin/clients")
async def admin_list_clients(secret: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    now = time.time()
    with pool_lock:
        client_to_pool = {}
        for pool in pools.values():
            for c in pool["clients"]:
                client_to_pool[c] = {"poolId": pool["pool_id"], "status": pool["status"],
                    "poolSize": len(pool["clients"]), "chunk": pool["chunk_idx"],
                    "ageS": int(now - pool["created_at"])}
        clients = []
        for ck, last_seen in sorted(active_clients.items(), key=lambda x: -x[1]):
            age = now - last_seen
            if age > FL_CONFIG["IP_TIMEOUT_S"]: continue
            hw = client_hardware.get(ck, {})
            clients.append({"clientKey": ck, "lastSeenS": int(age),
                "hardware": {"backend": hw.get("backend", "?"), "bs": hw.get("bs", 0),
                    "vram_gb": hw.get("vram_gb", 0.0), "gpu": hw.get("gpu", ""),
                    "autocast": hw.get("autocast", "any"), "optim": hw.get("optim", "any"),
                    "arch": _gpu_arch_class(hw.get("gpu") or ""),
                    "card": _gpu_card_class(hw.get("gpu") or ""),
                    "force_fp32": _force_fp32_arch(hw.get("gpu") or "")},
                "power": _is_power_client(ck),
                "pool": client_to_pool.get(ck)})
    return {"now": datetime.now(timezone.utc).isoformat(), "activeClients": len(clients),
        "formingPools": sum(1 for p in pools.values() if p["status"] == "forming"),
        "trainingPools": sum(1 for p in pools.values() if p["status"] == "training"),
        "clients": clients}

@app.get("/api/chunk_state")
async def chunk_state(secret: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    with pool_lock:
        pool_snapshot = [{"poolId": p["pool_id"], "status": p["status"], "clients": list(p["clients"]),
            "chunkIdx": p["chunk_idx"], "deltas": len(p["deltas"]),
            "ageS": int(time.time() - p["created_at"]), "targetAutocast": p.get("target_autocast")}
            for p in pools.values()]
        ready_count = len(ready_pools)
    return {"epoch": data_position["epoch"], "permutationPos": data_position["permutationPos"],
        "permutationLength": len(data_position["permutation"]),
        "currentChunkIdx": data_position["currentChunkIdx"],
        "consumedThroughPos": data_position.get("consumed_through_pos", -1),
        "currentShard": {"chunkIdx": current_data_shard["chunkIdx"], "round": current_data_shard.get("round"),
            "consumed": current_data_shard.get("consumed", False), "shardId": current_data_shard["shardId"],
            "assignedClients": list(current_data_shard.get("assignedClients", []))},
        "currentRound": round_state["current_round"], "globalStep": global_step,
        "totalChunks": dataset_info["totalChunks"],
        "validChunkRange": [min(dataset_info["validChunkIndices"]), max(dataset_info["validChunkIndices"])]
                           if dataset_info["validChunkIndices"] else None,
        "force_end_at": round_state.get("force_end_at"),
        "is_aggregating": round_state.get("is_aggregating", False),
        "crossCheck": {"enabled": CROSS_CHECK["ENABLED"], "cohortSize": CROSS_CHECK["COHORT_SIZE"],
            "maxOverflow": CROSS_CHECK["MAX_OVERFLOW"], "threshold": CROSS_CHECK["DISTANCE_THRESHOLD_RATIO"],
            "targetOptimizer": CROSS_CHECK["TARGET_OPTIMIZER"]},
        "matchmaking": {"poolSize": MATCHMAKING["POOL_SIZE"], "poolTimeoutS": MATCHMAKING["POOL_TIMEOUT_S"],
            "bf16SoloDissolveS": MATCHMAKING["BF16_SOLO_DISSOLVE_S"],
            "fp32SoloDissolveS": MATCHMAKING["FP32_SOLO_DISSOLVE_S"],
            "activePools": pool_snapshot, "readyPoolCount": ready_count},
        "byzantineFlagCounts": dict(byzantine_flag_counts),
        "clientHardware": {k: v for k, v in client_hardware.items()
                           if k in current_data_shard.get("assignedClients", [])}}

@app.post("/api/admin/dissolve_pool")
async def admin_dissolve_pool(secret: str = "", pool_id: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if not pool_id: raise HTTPException(400, "pool_id required")
    with pool_lock:
        pool = pools.pop(pool_id, None)
        if not pool: raise HTTPException(404, "Pool not found")
        freed = []
        for c in pool.get("clients", []):
            if client_pool_map.get(c) == pool_id:
                client_pool_map.pop(c, None)
                freed.append(c)
        for p in pool.get("deltas", {}).values():
            try: os.remove(p)
            except Exception: pass
    log.warning(f"🔧 Admin dissolved pool {pool_id} ({len(freed)} clients freed)")
    return {"status": "dissolved", "pool_id": pool_id, "clients_freed": freed}

@app.post("/api/admin/reload")
async def reload_checkpoint(step: int = -1, secret: str = ""):
    global global_weights, global_step
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if step == -1: load_global_state(); load_model_weights(); load_engram_weights()
    else:
        cp_file = CHECKPOINT_DIR / f"model_step_{step}.bin"
        if not cp_file.exists(): raise HTTPException(404, "Not found")
        buf = np.fromfile(cp_file, dtype=np.float32)
        if len(buf) != MODEL_SIZE: raise HTTPException(400, "Size mismatch")
        global_weights = torch.from_numpy(buf.copy())
        cp_eng_file = CHECKPOINT_DIR / f"engram_step_{step}.bin"
        if cp_eng_file.exists():
            buf_eng = np.fromfile(cp_eng_file, dtype=np.float32)
            if len(buf_eng) == ENG_PARAMS: global_engram_weights = torch.from_numpy(buf_eng.copy())
        global_step = step
    loss_ratchet_state.update({"lastAcceptedLoss": None, "lastAcceptedStep": None}); _save_loss_state()
    save_latest_checkpoint(); sync_huggingface_state(); update_and_save_status(push_hf=True)
    return {"status": "reloaded", "step": global_step}

@app.post("/api/admin/push_hf")
async def force_hf_push(secret: str = ""):
    global last_hf_upload_time
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    last_hf_upload_time = 0
    ok = upload_to_huggingface()
    update_and_save_status(push_hf=True)
    return {"status": "uploaded" if ok else "failed"}

@app.get("/api/admin/candidate/{part}")
async def get_candidate(part: str, token: str = ""):
    if not _candidate_token or not token or not secrets.compare_digest(token, _candidate_token):
        raise HTTPException(403, "Invalid or expired token")
    p = CANDIDATE_MODEL_FILE if part == "model" else CANDIDATE_ENGRAM_FILE if part == "engram" else None
    if p is None or not p.exists(): raise HTTPException(404, "No candidate")
    return FileResponse(str(p), media_type="application/octet-stream", headers={"X-Global-Step": str(global_step)})

@app.api_route("/api/admin/loss_ratchet", methods=["GET", "POST"])
async def admin_loss_ratchet(secret: str = "", mode: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if mode:
        if mode not in ("off", "log", "enforce"): raise HTTPException(400, "mode must be off|log|enforce")
        HARDEN["LOSS_CHECK_MODE"] = mode
        log.warning(f"🔧 Loss ratchet mode → {mode}")
    return {"mode": HARDEN["LOSS_CHECK_MODE"], "failOpen": HARDEN["LOSS_FAIL_OPEN"],
            "tolerance": HARDEN["LOSS_TOLERANCE"], "evaluatorConfigured": bool(HARDEN["LOSS_CHECK_URL"]),
            "lastAcceptedLoss": loss_ratchet_state.get("lastAcceptedLoss"),
            "lastAcceptedStep": loss_ratchet_state.get("lastAcceptedStep"),
            "recent": loss_ratchet_state["history"][-10:]}

@app.post("/api/admin/clear_byzantine_flags")
async def admin_clear_byzantine(secret: str = "", client_key: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if client_key:
        uid = _user_id_from_key(client_key)
        if uid:
            with users_lock:
                if uid in users_db["users"]:
                    users_db["users"][uid]["stats"]["byzantine_flags"] = 0
                    save_users_db()
            try: update_and_save_status(push_hf=False)
            except Exception as e: log.error(f"status refresh after clear: {e}")
            return {"status": "cleared", "client": client_key}
    else:
        with users_lock:
            for u in users_db["users"].values(): u["stats"]["byzantine_flags"] = 0
            save_users_db()
        byzantine_flag_counts.clear()
        try: update_and_save_status(push_hf=False)
        except Exception as e: log.error(f"status refresh after clear: {e}")
        return {"status": "all_cleared"}
    raise HTTPException(404, "Not found")

@app.post("/api/admin/set_user_stats")
async def admin_set_user_stats(secret: str = "", user_id: str = "",
    contributions: Optional[int] = None, byzantine_flags: Optional[int] = None,
    total_tokens: Optional[int] = None, total_steps: Optional[int] = None):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if not user_id: raise HTTPException(400, "user_id required")
    with users_lock:
        u = users_db["users"].get(user_id)
        if not u: raise HTTPException(404, "User not found")
        s = u.setdefault("stats", {})
        if contributions is not None: s["contributions"] = max(0, int(contributions))
        if byzantine_flags is not None: s["byzantine_flags"] = max(0, int(byzantine_flags))
        if total_tokens is not None: s["totalTokens"] = max(0, int(total_tokens))
        if total_steps is not None: s["totalSteps"] = max(0, int(total_steps))
        save_users_db()
    try: update_and_save_status(push_hf=False)
    except Exception as e: log.error(f"status refresh after set_user_stats: {e}")
    return {"user_id": user_id, "username": u["username"],
        "stats": {k: s.get(k, 0) for k in ("contributions","byzantine_flags","totalTokens","totalSteps")},
        "trustScore": compute_trust_score(s)}

@app.post("/api/admin/reset")
async def admin_reset(secret: str = ""):
    global global_weights, global_engram_weights, global_step, global_stats, lora_stats
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    torch.manual_seed(42)
    global_weights = (torch.randn(MODEL_SIZE) * 0.02).float()
    global_engram_weights = (torch.randn(ENG_PARAMS) * 0.02).float()
    global_step = 0
    loss_ratchet_state.update({"lastAcceptedLoss": None, "lastAcceptedStep": None, "history": []}); _save_loss_state()
    global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
    lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}
    save_latest_checkpoint(); sync_huggingface_state(); update_and_save_status(push_hf=True)
    return {"status": "reset"}

@app.post("/api/admin/force_round_end")
async def admin_force_round_end(secret: str = "", minutes: float = 0):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if minutes < 0 or minutes > 24 * 60: raise HTTPException(400, "minutes must be 0..1440")
    with shard_lock:
        if round_state.get("is_aggregating"): raise HTTPException(409, "Round is already ending — wait for the next round")
        round_state["force_end_at"] = time.time() + minutes * 60
        ends_at, current = round_state["force_end_at"], round_state["current_round"]
    log.warning(f"⏱️ Admin scheduled end of round {current} in {minutes:.1f} min")
    return {"status": "scheduled", "round": current, "minutes": minutes,
        "endsAt": datetime.fromtimestamp(ends_at, tz=timezone.utc).isoformat()}

@app.delete("/api/admin/force_round_end")
async def admin_cancel_force_round_end(secret: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    with shard_lock: old = round_state.pop("force_end_at", None)
    return {"status": "cancelled" if old is not None else "no_pending_force_end",
        "cancelledAt": datetime.fromtimestamp(old, tz=timezone.utc).isoformat() if old else None}

@app.post("/api/admin/force_update")
async def admin_force_update(secret: str = "", min_version: str = "", end_round: bool = True):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if min_version and not _vtuple(min_version): raise HTTPException(400, "min_version must look like 1.9")
    _set_min_client_version(min_version)
    ended = "skipped"
    if end_round:
        with shard_lock:
            if round_state.get("is_aggregating"): ended = "already_ending"
            else:
                round_state["force_end_at"] = time.time()
                ended = "scheduled_now"
    log.warning(f"⬆️ Admin force_update: min_version={min_version or '(none)'} round_end={ended}")
    return {"status": "ok", "minClientVersion": min_version or None, "roundEnd": ended,
            "blockUnversioned": _block_unversioned}

@app.get("/api/admin/force_update")
async def admin_force_update_status(secret: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    return {"minClientVersion": _min_client_version or None, "blockUnversioned": _block_unversioned}

def _delta_metadata(path: str) -> Optional[dict]:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<I", f.read(4))[0]
            return json.loads(f.read(n))
    except Exception: return None

def _find_user_uuid(user_id: str = "", username: str = "") -> Optional[str]:
    if user_id and user_id in users_db["users"]: return user_id
    if username:
        for u in users_db["users"].values():
            if u.get("username_lower") == username.lower(): return u["uuid"]
    return None

@app.post("/api/admin/power_user")
async def admin_power_user(secret: str = "", user_id: str = "", username: str = "",
                           enabled: bool = True, note: str = ""):
    """Make an account a power user (enabled=true, default) or remove it (enabled=false).
    Identify the account by user_id (uuid) or username."""
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if not user_id and not username: raise HTTPException(400, "user_id or username required")
    with users_lock:
        uid = _find_user_uuid(user_id, username)
        if not uid: raise HTTPException(404, "User not found")
        u = users_db["users"][uid]
        if enabled:
            u["power"] = True; u["power_since"] = datetime.now(timezone.utc).isoformat()
            if note: u["power_note"] = note[:200]
        else:
            u.pop("power", None); u.pop("power_since", None); u.pop("power_note", None)
        save_users_db()
        name = u["username"]
    log.warning(f"⚡ Admin {'granted' if enabled else 'removed'} power user: {name} ({uid})")
    try: update_and_save_status(push_hf=False)       # refresh leaderboard.json so the shield shows up
    except Exception as e: log.error(f"status refresh after power_user: {e}")
    return {"status": "power_enabled" if enabled else "power_removed", "user_id": uid, "username": name}

@app.get("/api/admin/power_users")
async def admin_list_power_users(secret: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    with users_lock:
        out = [{"user_id": u["uuid"], "username": u["username"], "since": u.get("power_since"),
                "note": u.get("power_note", "")} for u in users_db["users"].values() if u.get("power")]
    return {"count": len(out), "soloWeight": POWER["SOLO_WEIGHT"], "soloEnabled": POWER["SOLO_ENABLED"], "joinAnyPool": POWER["JOIN_ANY_POOL"], "powerUsers": out}

@app.post("/api/admin/ban_user")
async def admin_ban_user(secret: str = "", user_id: str = "", reason: str = "",
    hours: int = 0, ban_ip: bool = False, purge_pending: bool = True):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if not user_id: raise HTTPException(400, "user_id required")
    until = time.time() + hours * 3600 if hours > 0 else 0.0
    purged = 0
    with users_lock:
        user = users_db["users"].get(user_id)
        if not user: raise HTTPException(404, "User not found")
        ips = {t.get("requestedByIp") for t in db["tasks"].values()
               if t.get("requestedByKey") == f"user:{user_id}" and t.get("requestedByIp")}
        banned_users[user_id] = {"username": user["username"], "reason": reason,
            "bannedAt": time.time(), "until": until, "ips": list(ips)}
        banned_client_keys[f"user:{user_id}"] = until
        if ban_ip:
            for ip in ips: ban_subnet(ip, f"user {user['username']} banned", hours or 24)
        save_users_db()
    if purge_pending:
        with shard_lock:
            kept = []
            for p in round_state["pending_deltas"]:
                m = _delta_metadata(p)
                if m and _user_id_from_key(m.get("nodeKey", "")) == user_id:
                    try: os.remove(p)
                    except Exception: pass
                    purged += 1
                else: kept.append(p)
            round_state["pending_deltas"] = kept
        with pool_lock:
            for pool in pools.values():
                for ck in list(pool["deltas"].keys()):
                    if _user_id_from_key(ck) == user_id:
                        try: os.remove(pool["deltas"][ck])
                        except Exception: pass
                        del pool["deltas"][ck]
                        purged += 1
    log.warning(f"🔨 Admin banned {user['username']} ({user_id}) for {hours or 'permanent'}h — {reason} (ips={list(ips)}, purged={purged})")
    return {"status": "banned", "user_id": user_id, "username": user["username"],
        "until": datetime.fromtimestamp(until, tz=timezone.utc).isoformat() if until else "permanent",
        "ips_banned": list(ips) if ban_ip else [], "pending_deltas_purged": purged}

@app.post("/api/admin/unban_user")
async def admin_unban_user(secret: str = "", user_id: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if not user_id: raise HTTPException(400, "user_id required")
    with users_lock:
        had = banned_users.pop(user_id, None)
        banned_client_keys.pop(f"user:{user_id}", None)
    return {"status": "unbanned" if had else "not_banned", "user_id": user_id}

@app.post("/api/admin/ban_ip")
async def admin_ban_ip(secret: str = "", ip: str = "", reason: str = "", hours: int = 24):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if not ip: raise HTTPException(400, "ip required")
    ban_subnet(ip, reason or "admin", hours)
    return {"status": "banned", "ip": ip, "subnet": get_subnet(ip), "hours": hours}

@app.post("/api/admin/clear_subnet_bans")
async def admin_clear_subnet_bans(secret: str = "", subnet: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    with pool_lock:
        if subnet:
            n = 1 if blacklisted_subnets.pop(subnet, None) else 0
            byzantine_flag_counts.pop(subnet, None)
            return {"status": "cleared_one" if n else "not_found", "subnet": subnet}
        count = len(blacklisted_subnets)
        blacklisted_subnets.clear()
        byzantine_flag_counts.clear()
    log.warning(f"🔧 Admin cleared {count} subnet ban(s) + all flag counters")
    try: update_and_save_status(push_hf=False)
    except Exception: pass
    return {"status": "cleared", "subnets_cleared": count}

@app.get("/api/admin/bans")
async def admin_list_bans(secret: str = ""):
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    now = time.time()
    return {"users": [{"user_id": uid, **b} for uid, b in banned_users.items() if not b.get("until") or b["until"] > now],
        "client_keys": [{"client_key": k, "expires_in_s": int(v - now) if v else None}
                        for k, v in banned_client_keys.items() if not v or v > now],
        "subnets": [{"subnet": k, "expires_in_s": int(v - now)} for k, v in blacklisted_subnets.items() if v > now],
        "force_end_at": round_state.get("force_end_at"), "current_round": round_state["current_round"],
        "is_aggregating": round_state.get("is_aggregating", False)}

# ============ PUBLIC STATS ============
@app.get("/api/stats")
async def get_stats(request: Request):
    recent = global_stats["lossHistory"][-10:]
    avg_loss = sum(recent) / len(recent) if recent else 0.0
    loss_history = global_stats["lossHistory"]
    training_loss_change = (round(((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100, 1)
        if len(loss_history) >= 2 and loss_history[0] > 0 else 0.0)
    with users_lock: total_users = len(users_db["users"])
    active_gpus = _active_client_count()
    auth_user = get_current_user(request)
    your_contribution = 0.0
    if auth_user:
        total_tokens = global_stats["totalTokens"]
        if total_tokens > 0: your_contribution = round((auth_user["stats"].get("totalTokens", 0) / total_tokens) * 100, 4)
    with pool_lock:
        mm = {"poolSize": MATCHMAKING["POOL_SIZE"], "activePools": len(pools),
            "formingPools": sum(1 for p in pools.values() if p["status"] == "forming"),
            "trainingPools": sum(1 for p in pools.values() if p["status"] == "training"),
            "readyPools": len(ready_pools)}
    return {"totalParticipants": total_users, "totalContributions": lora_stats.get("total_submissions", 0),
        "activeGpus": active_gpus, "totalTokens": global_stats["totalTokens"],
        "trainingLossChange": training_loss_change, "yourContribution": your_contribution,
        "activeNow": active_gpus, "totalSteps": global_stats["totalSteps"], "globalStep": global_step,
        "modelSize": MODEL_SIZE, "lastLoss": global_stats["lastLoss"], "avgLoss": avg_loss,
        "weightNorm": float(torch.norm(global_weights.float()).item()) if global_weights is not None else 0.0,
        "loraStats": lora_stats, "totalUsers": total_users, "activeContributors": active_gpus,
        "lossHistory": global_stats["lossHistory"][-20:],
        "datasetStats": {"totalChunks": dataset_info["totalChunks"], "currentChunk": data_position["currentChunkIdx"],
            "currentEpoch": data_position["epoch"], "totalShardsProcessed": dataset_stats["totalShardsProcessed"],
            "epochsCompleted": dataset_stats["epochsCompleted"],
            "shuffleEnabled": FL_CONFIG.get("SHUFFLE_CHUNKS", True),
            "permutationProgress": data_position.get("permutationPos", 0),
            "permutationSize": len(data_position.get("permutation", [])),
            "consumedThroughPos": data_position.get("consumed_through_pos", -1)},
        "currentRound": round_state["current_round"], "forceEndAt": round_state.get("force_end_at"),
        "crossCheck": {"enabled": CROSS_CHECK["ENABLED"], "cohortSize": CROSS_CHECK["COHORT_SIZE"],
            "maxOverflow": CROSS_CHECK["MAX_OVERFLOW"], "threshold": CROSS_CHECK["DISTANCE_THRESHOLD_RATIO"],
            "targetBatchSize": CROSS_CHECK["TARGET_BATCH_SIZE"], "targetLocalSteps": CROSS_CHECK["TARGET_LOCAL_STEPS"],
            "targetOptimizer": CROSS_CHECK["TARGET_OPTIMIZER"],
            "banThreshold": CROSS_CHECK["BAN_THRESHOLD"]},
        "matchmaking": mm, "byzantineFlagCounts": dict(byzantine_flag_counts),
        "huggingface": {"repoId": HF_REPO_ID, "enabled": HF_ENABLED,
            "uploadIntervalHours": HF_UPLOAD_INTERVAL_HOURS,
            "revision": current_hf_revision, "revisionStep": hf_revision_global_step,
            "directDownloads": bool(_build_hf_weights_metadata("bf16"))} if HF_ENABLED else {}}

@app.get("/api/leaderboard")
async def get_leaderboard():
    lb_file = BASE_DIR / "leaderboard.json"
    if lb_file.exists():
        try:
            with open(lb_file, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception: pass
    with users_lock:
        users_list = [{"uuid": u["uuid"], "username": u["username"],
            "contributions": u["stats"].get("contributions", 0),
            "totalTokens": u["stats"].get("totalTokens", 0),
            "totalSteps": u["stats"].get("totalSteps", 0),
            "trustScore": compute_trust_score(u["stats"]),
            "byzantineFlags": u["stats"].get("byzantine_flags", 0),
            "power": bool(u.get("power", False))} for u in users_db["users"].values()]
    return sorted(users_list, key=lambda x: x["totalTokens"], reverse=True)

@app.get("/api/checkpoint")
async def get_checkpoint(request: Request, format: str = "bf16"):
    if not LATEST_CHECKPOINT_FILE.exists(): raise HTTPException(404, "No checkpoint")
    return StreamingResponse(_stream_f32_file(LATEST_CHECKPOINT_FILE, MODEL_SIZE, format),
        media_type="application/octet-stream",
        headers={"X-Global-Step": str(global_step), "X-Model-Size": str(MODEL_SIZE)})

@app.get("/api/engram_checkpoint")
async def get_engram_checkpoint(request: Request, format: str = "bf16"):
    if not ENGRAM_CHECKPOINT_FILE.exists(): raise HTTPException(404, "No engram checkpoint")
    return StreamingResponse(_stream_f32_file(ENGRAM_CHECKPOINT_FILE, ENG_PARAMS, format),
        media_type="application/octet-stream",
        headers={"X-Global-Step": str(global_step), "X-Engram-Size": str(ENG_PARAMS)})

@app.get("/api/model_info")
async def get_model_info():
    return {"architecture": MODEL_CONFIG["architecture"], "parameters": MODEL_SIZE, "engramParams": ENG_PARAMS,
        "config": MODEL_CONFIG, "globalStep": global_step, "currentRound": round_state["current_round"],
        "crossCheck": {"enabled": CROSS_CHECK["ENABLED"], "cohortSize": CROSS_CHECK["COHORT_SIZE"],
            "maxOverflow": CROSS_CHECK["MAX_OVERFLOW"], "threshold": CROSS_CHECK["DISTANCE_THRESHOLD_RATIO"],
            "targetBatchSize": CROSS_CHECK["TARGET_BATCH_SIZE"], "targetLocalSteps": CROSS_CHECK["TARGET_LOCAL_STEPS"],
            "targetOptimizer": CROSS_CHECK["TARGET_OPTIMIZER"]},
        "matchmaking": {"poolSize": MATCHMAKING["POOL_SIZE"], "poolTimeoutS": MATCHMAKING["POOL_TIMEOUT_S"],
            "formingTimeoutS": MATCHMAKING["FORMING_TIMEOUT_S"],
            "bf16SoloDissolveS": MATCHMAKING["BF16_SOLO_DISSOLVE_S"],
            "fp32SoloDissolveS": MATCHMAKING["FP32_SOLO_DISSOLVE_S"],
            "hwBsTolerance": MATCHMAKING["HW_BS_TOLERANCE"],
            "matchingRule": "v2.3: backend exact; optim normalize+wildcard+fp32/8bit mix; autocast asymmetric (fp32 pool accepts any, bf16 pool bf16 only); solo forming pools re-matched on every poll"},
        "huggingface": {"repoId": HF_REPO_ID, "enabled": HF_ENABLED, "revision": current_hf_revision,
            "revisionStep": hf_revision_global_step,
            "directDownloads": bool(_build_hf_weights_metadata("bf16"))} if HF_ENABLED else {}}

@app.post("/api/admin/restore_from_hf")
async def admin_restore_from_hf(secret: str = "", revision: str = "", step: int = -1):
    global global_weights, global_engram_weights, global_step
    if not _admin_ok(secret): raise HTTPException(403, "Invalid secret")
    if not HF_ENABLED: raise HTTPException(503, "HF disabled")
    if not revision: raise HTTPException(400, "revision required (commit SHA or tag)")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    hf_cache = CHECKPOINT_DIR / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    log.warning(f"🔧 Restore-from-HF: revision={revision[:12]} requested_step={step}")
    try:
        cfg_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json",
            revision=revision, token=HF_TOKEN, cache_dir=str(hf_cache))
        with open(cfg_path, 'r', encoding='utf-8') as f: hf_cfg = json.load(f)
        hf_step = int(hf_cfg.get("globalStep", -1))
        if hf_step < 0: raise ValueError("no globalStep in config.json")
    except Exception as e: raise HTTPException(500, f"config.json at {revision}: {e}")
    if step < 0: step = hf_step
    elif step != hf_step:
        log.warning(f"⚠️ Requested step={step} but config.json says {hf_step}; using requested")
    try:
        st_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model_bf16.safetensors",
            revision=revision, token=HF_TOKEN, cache_dir=str(hf_cache))
        tensors = load_file(st_path)
        if "weights" not in tensors: raise ValueError(f"no 'weights' key; keys={list(tensors.keys())[:5]}")
        w = tensors["weights"]
        if w.numel() != MODEL_SIZE: raise ValueError(f"numel mismatch: {w.numel():,} vs {MODEL_SIZE:,}")
        arr = w.float().numpy().astype(np.float32)
        del w, tensors; gc.collect()
    except Exception as e: raise HTTPException(500, f"weights at {revision}: {e}")
    step_bin = CHECKPOINT_DIR / f"model_step_{step}.bin"
    tmp = step_bin.with_suffix('.bin.tmp')
    try:
        with open(tmp, 'wb') as f:
            for i in range(0, len(arr), 10_000_000): arr[i:i + 10_000_000].tofile(f)
        os.replace(tmp, step_bin)
        log.info(f"💾 Wrote {step_bin} ({step_bin.stat().st_size / 1024**2:.1f} MB)")
    except Exception as e:
        try: tmp.unlink(missing_ok=True)
        except: pass
        raise HTTPException(500, f"write .bin: {e}")
    try:
        new_w = np.fromfile(step_bin, dtype=np.float32)
        if len(new_w) != MODEL_SIZE: raise ValueError(f"size {len(new_w):,} != {MODEL_SIZE:,}")
        with shard_lock:
            global_weights = torch.from_numpy(new_w.copy())
            with stats_lock: global_step = step
        log.warning(f"🔧 global_weights ← step {step} (revision {revision[:12]})")
    finally:
        del arr; gc.collect()
    try:
        eng_path = hf_hub_download(repo_id=HF_REPO_ID, filename="engram_bf16.safetensors",
            revision=revision, token=HF_TOKEN, cache_dir=str(hf_cache))
        et = load_file(eng_path)
        if "weights" in et:
            ew = et["weights"]
            if ew.numel() == ENG_PARAMS:
                with shard_lock: global_engram_weights = ew.float().cpu()
                step_eng = CHECKPOINT_DIR / f"engram_step_{step}.bin"
                tmp_e = step_eng.with_suffix('.bin.tmp')
                with open(tmp_e, 'wb') as f:
                    for i in range(0, ENG_PARAMS, 10_000_000):
                        global_engram_weights[i:i+10_000_000].float().numpy().astype(np.float32).tofile(f)
                os.replace(tmp_e, step_eng)
                log.info(f"🧠 Restored engram step {step}")
    except Exception as e: log.warning(f"⚠️ Engram restore skipped: {e}")
    save_latest_checkpoint()
    pushed = upload_to_huggingface()
    try: update_and_save_status(push_hf=True)
    except Exception as e: log.error(f"status: {e}")
    wte_norm = float(global_weights[:MODEL_CONFIG["vocabSize"] * MODEL_CONFIG["dim"]].float().norm().item())
    return {"status": "restored", "revision": revision, "step": step, "hf_step": hf_step,
        "model_size": MODEL_SIZE, "wte_norm": wte_norm, "pushed_to_hf": pushed}

# ============ STARTUP ============
@app.on_event("startup")
async def startup():
    load_dataset_state(); load_users_db(); load_global_state(); _load_loss_state()
    initialize_model(); load_model_weights(); load_engram_weights()
    initialize_dataset(); _ensure_permutation(); _crash_recovery()
    with pool_lock:
        pools.clear(); client_pool_map.clear(); ready_pools.clear()
    round_state["round_start_time"] = time.time()
    round_state["current_round"] = global_step + 1
    try:
        for f in DELTA_TEMP_DIR.glob("*.bin"): f.unlink()
        log.info("🧹 Cleared old pending delta files")
    except Exception: pass
    try:
        for p in UPLOAD_DIR.iterdir():
            if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
            else: p.unlink(missing_ok=True)
        log.info("🧹 Cleared old pending upload dirs")
    except Exception: pass
    sync_huggingface_state(); update_and_save_status(push_hf=True)
    log.info("=" * 60)
    log.info(f"🚀 CrowdGPT Coordinator on port {PORT} (v2.3 solo-rematch + optim-normalize)")
    log.info(f"⏱️  Round: {round_state['round_duration_hours']}h")
    log.info(f"🎲 Shuffle: {FL_CONFIG['SHUFFLE_CHUNKS']} | epoch={data_position['epoch']} | "
             f"pos={data_position['permutationPos']}/{len(data_position['permutation'])}")
    log.info(f"📦 Dataset: {dataset_info['totalChunks']} chunks")
    log.info(f"👥 Matchmaking: pool_size={MATCHMAKING['POOL_SIZE']} pool_timeout={MATCHMAKING['POOL_TIMEOUT_S']}s")
    log.info(f"   → bf16 solo dissolve={MATCHMAKING['BF16_SOLO_DISSOLVE_S']}s, "
             f"fp32 solo dissolve={MATCHMAKING['FP32_SOLO_DISSOLVE_S']}s")
    log.info(f"🎯 Priority: FP32 pools > BF16 pools; FP32 accepts any, BF16 bf16-only")
    log.info(f"🎯 Optim: normalize aliases + allow fp32/8bit AdamW to mix; 'any' = wildcard")
    log.info(f"🔄 Solo forming pools re-matched on every poll (v2.3)")
    log.info(f"🔒 Exactly-once: chunk advances per-pool at activation; merge only at round-end.")
    log.info(f"🔄 Initialized round {round_state['current_round']}")
    _log_hardening_summary()
    _log_cross_check_summary()
    log.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    from logging.handlers import RotatingFileHandler
    log_file = BASE_DIR / "server.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", access_log=False)
