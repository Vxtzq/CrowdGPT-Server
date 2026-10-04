#!/usr/bin/env python3
"""
CrowdGPT Server - Continuous FL Coordinator
- HF-direct weight downloads
- Chunked secure delta uploads
- Disk-based sequential aggregation
- Cloudflare-friendly control plane
- Dual-stack safe client identity (user UUID primary, IP fallback)
"""

import os
import sys
import io
import json
import zlib
import gzip
import time
import struct
import secrets
import threading
import requests
import math
import random
import hashlib
import uuid
import shutil
import tempfile
import gc
import atexit
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

# ============ LOGGING ============
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ============ IMPORTS ============
import numpy as np
import torch
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


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

# Upload limits
MAX_CHUNK_BYTES = 80 * 1024 * 1024
MAX_DIRECT_BODY_BYTES = 90 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 3 * 1024 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_CHUNKS = 512
UPLOAD_TTL_SECONDS = 6 * 60 * 60

import socket
hostname = socket.gethostname()
try:
    local_ip = socket.gethostbyname(hostname)
except Exception:
    local_ip = "unknown"

print(f"🌐 Container hostname: {hostname}")
print(f"🌐 Container local IP: {local_ip}")
print(f"🌐 Listening on: 0.0.0.0:{PORT}")
print("=" * 60)

HF_TOKEN = os.getenv("HF_TOKEN", "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")
HF_REPO_ID = "Vxtzq/Crowd-v1"
DATASET_REPO_ID = "Vxtzq/CrowdGPT"
HF_UPLOAD_INTERVAL_HOURS = 2
HF_ENABLED = HF_AVAILABLE and bool(HF_TOKEN)
ADMIN_SECRET = "crowdgpt-admin-9f3k2m8x"

HF_WEIGHT_FILES = [
    "model_bf16.safetensors",
    "engram_bf16.safetensors",
]

# ============ MODEL ============
MODEL_CONFIG = {
    "vocabSize": 151669,
    "dim": 1536,
    "nLayers": 24,
    "nHeads": 16,
    "nKvHeads": 4,
    "headDim": 96,
    "maxSeqLen": 2048,
    "mlpHidden": 2560,
    "weightTying": True,
    "architecture": "SotaGPT",
    "features": {
        "groupedQueryAttention": True,
        "rotaryPositionEmbedding": True,
        "swiGLU": True,
        "weightTying": True,
        "chunkedAttention": True,
    },
}

LORA_CONFIG = {
    "FULL_RANK": MODEL_CONFIG["dim"],
    "RANK_TIERS": {
        "minimal": 32,
        "low": 64,
        "medium": 128,
        "high": 256,
        "ultra": 512,
        "extreme": 1024,
        "max": 2048,
        "full": MODEL_CONFIG["dim"],
    },
    "DEFAULT_RANK": MODEL_CONFIG["dim"],
    "WEIGHT_BY_RANK": True,
    "LOG_EFFECTIVE_RANK": True,
}

DATASET_CONFIG = {
    "repoId": DATASET_REPO_ID,
    "totalChunks": 260,
    "subChunkMB": 10,
    "tokensPerSample": MODEL_CONFIG["maxSeqLen"] + 1,
    "tokenBytes": 4,
}


def calculate_model_size(cfg: dict) -> int:
    size = cfg["vocabSize"] * cfg["dim"]
    for _ in range(cfg["nLayers"]):
        d = cfg["dim"]
        size += d * 2 + d * (cfg["nHeads"] * cfg["headDim"]) + d * (cfg["nKvHeads"] * cfg["headDim"]) * 2 + d * d + d * 2
        mlp_hidden = cfg.get("mlpHidden", 4 * d)
        size += d * mlp_hidden * 2 + mlp_hidden * d
    size += cfg["dim"] * 2
    if not cfg.get("weightTying", False):
        size += cfg["dim"] * cfg["vocabSize"]
    return size


MODEL_SIZE = calculate_model_size(MODEL_CONFIG)
log.info(f"🧮 SotaGPT size: {MODEL_SIZE:,} params (~{MODEL_SIZE*4/1024/1024:.2f} MB in FP32, ~{MODEL_SIZE*2/1024/1024:.2f} MB in BF16)")

ENG_NUM_BUCKETS = 227865
ENG_PARAMS = ENG_NUM_BUCKETS * MODEL_CONFIG["dim"]
log.info(f"🧠 Engram size: {ENG_PARAMS:,} params (~{ENG_PARAMS*4/1024/1024:.2f} MB in FP32, ~{ENG_PARAMS*2/1024/1024:.2f} MB in BF16)")


# ============ QUANTIZATION ============
def float32_to_bf16_array(a):
    return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16).view(torch.uint16).numpy()

def bf16_array_to_float32(a):
    return torch.from_numpy(a.copy()).view(torch.uint16).view(torch.bfloat16).to(torch.float32).numpy()

def float32_to_fp16_array(a):
    return np.clip(np.ascontiguousarray(a, dtype=np.float32), -65504.0, 65504.0).astype(np.float16).view(np.uint16)

def fp16_array_to_float32(a):
    return a.view(np.float16).astype(np.float32)

def float32_to_fp8_array(a):
    return torch.from_numpy(a.astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()

def fp8_array_to_float32(a):
    return torch.from_numpy(a.copy()).view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).numpy()

def compress_weights_bf16(w):
    return float32_to_bf16_array(w).tobytes()

def compress_weights_fp16(w):
    return float32_to_fp16_array(w).tobytes()

def compress_weights_fp8(w):
    return float32_to_fp8_array(w).tobytes()


def _bytes_per_param(fmt: str) -> int:
    if fmt in ("bf16", "fp16"):
        return 2
    if fmt == "fp8":
        return 1
    if fmt == "fp32":
        return 4
    raise HTTPException(status_code=400, detail=f"Unsupported weight format: {fmt}")


def _decode_delta_chunk(buf: bytearray, fmt: str) -> torch.Tensor:
    if fmt == "fp8":
        return torch.frombuffer(buf, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    if fmt == "fp16":
        return torch.frombuffer(buf, dtype=torch.uint16).view(torch.float16).to(torch.float32)
    if fmt == "fp32":
        return torch.frombuffer(buf, dtype=torch.float32)
    return torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16).to(torch.float32)


def _encode_float32_to_bf16_bytes(t: torch.Tensor) -> bytes:
    return t.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()


# ============ DISK-BASED DELTA STORAGE ============
def save_delta_to_disk(client_ip, delta, engram_indices, engram_values, metadata):
    ts = int(time.time() * 1000)
    safe_ip = client_ip.replace(".", "_").replace(":", "_")
    fname = DELTA_TEMP_DIR / f"delta_{safe_ip}_{ts}.bin"

    with open(fname, 'wb') as f:
        meta_bytes = json.dumps(metadata).encode('utf-8')
        f.write(struct.pack('<I', len(meta_bytes)))
        f.write(meta_bytes)

        chunk_size = 10_000_000
        for i in range(0, MODEL_SIZE, chunk_size):
            chunk = delta[i:i + chunk_size].cpu().view(torch.uint16).numpy()
            f.write(chunk.tobytes())

        if engram_indices is not None and len(engram_indices) > 0:
            f.write(struct.pack('<I', len(engram_indices)))
            f.write(engram_indices.tobytes())
            eng_vals_bytes = engram_values.cpu().view(torch.uint16).numpy().tobytes()
            f.write(eng_vals_bytes)
        else:
            f.write(struct.pack('<I', 0))

    return str(fname)


def load_delta_from_disk(filepath):
    with open(filepath, 'rb') as f:
        meta_len = struct.unpack('<I', f.read(4))[0]
        metadata = json.loads(f.read(meta_len).decode('utf-8'))

        delta = torch.zeros(MODEL_SIZE, dtype=torch.bfloat16)
        chunk_size = 10_000_000
        offset = 0
        bytes_per_param = 2
        while offset < MODEL_SIZE:
            read_params = min(chunk_size, MODEL_SIZE - offset)
            chunk_bytes = f.read(read_params * bytes_per_param)
            if not chunk_bytes:
                break
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
            engram_indices = None
            engram_values = None

    return metadata, delta, engram_indices, engram_values


# ============ GLOBAL STATE ============
db = {"tasks": {}, "pendingSubmissions": {}, "shardMeta": {}}
active_clients: Dict[str, float] = {}          # was active_ips — now keyed by client_key
blacklisted_subnets: Dict[str, float] = {}

global_weights: Optional[torch.Tensor] = None
global_engram_weights: Optional[torch.Tensor] = None

global_step = 0
global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}

dataset_info = {
    "repoId": DATASET_CONFIG["repoId"],
    "totalChunks": 0,
    "subChunkBytes": DATASET_CONFIG["subChunkMB"] * 1024 * 1024,
    "initialized": False,
    "lastChecked": None,
}
data_position = {"currentChunkIdx": 0, "epoch": 0, "totalSamplesSeen": 0}
current_data_shard = {"chunkIdx": 0, "assignedClients": [], "shardId": None, "createdAt": None}
dataset_stats = {"totalShardsProcessed": 0, "epochsCompleted": 0, "samplesProcessed": 0, "lastUpdate": None}

round_state = {
    "current_round": 1,
    "round_start_time": time.time(),
    "round_duration_hours": 4.0,
    "participating_clients": {},               # keyed by client_key now
    "aggregations_this_round": 0,
    "last_aggregation_time": 0,
    "pending_deltas": [],
    "is_aggregating": False,
}

FL_CONFIG = {
    "CONSENSUS_REQUIRED": 1,
    "CHECKPOINT_INTERVAL": 1,
    "MAX_CHECKPOINTS": 5,
    "DELTA_NORM_BASE": 500.0,
    "DELTA_NORM_ABSOLUTE_MAX": 5000.0,
    "MAX_WEIGHT_NORM": 1000.0,
    "WEIGHT_DECAY": 0.0001,
    "SERVER_LR": 1.0,
    "IP_TIMEOUT_S": 900,
    "SHARD_TIMEOUT_S": 1800,
    "CLEANUP_INTERVAL_S": 60,
}

ROUND_CONFIG = {
    "MAX_ROUND_HOURS": 4.0,
    "TARGET_SUBMISSION_HOURS": 1.0,
    "MIN_CLIENTS_TO_AGGREGATE": 9999,
    "CLIENT_TIMEOUT_HOURS": 2.5,
    "AGGREGATION_COOLDOWN_SEC": 120,
    "ULTIMATUM_THRESHOLD": 0.90,
    "ULTIMATUM_TIMEOUT_MIN": 30,
}

shard_lock = threading.RLock()


# ============ AUTH & LEADERBOARD ============
users_db = {"users": {}, "tokens": {}}
users_lock = threading.Lock()
stats_lock = threading.Lock()

BAD_WORDS = [
    "fuck", "shit", "ass", "bitch", "dick", "cock", "pussy", "cunt", "bastard", "damn", "piss", "crap", "twat",
    "nigger", "nigga", "faggot", "fag", "retard", "tranny", "whore", "slut", "rape", "pedo", "nazi", "hitler",
    "porn", "xxx", "hentai", "admin", "administrator", "root", "system", "moderator", "mod", "staff", "official"
]


def load_users_db():
    global users_db
    try:
        if USERS_DB_FILE.exists():
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                users_db = json.load(f)
            log.info(f"👥 Loaded {len(users_db['users'])} users from disk")
    except Exception as e:
        log.error(f"❌ load_users_db: {e}")
        users_db = {"users": {}, "tokens": {}}


def save_users_db():
    try:
        with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_db, f, indent=2)
    except Exception as e:
        log.error(f"❌ save_users_db: {e}")


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def contains_bad_word(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    normalized = lower.translate(str.maketrans("013@$", "oieas"))
    for word in BAD_WORDS:
        if word in lower or word in normalized:
            return True
    return any(c in text for c in ['\n', '\r', '\t'])


def get_current_user(request: Request) -> Optional[dict]:
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:].strip()
    with users_lock:
        user_uuid = users_db["tokens"].get(token)
        if not user_uuid:
            return None
        return users_db["users"].get(user_uuid)


def compute_trust_score(stats: dict) -> float:
    return round(min(1.0, 0.5 + (stats.get("contributions", 0) / 200.0)), 2)


def compute_global_rank(user_uuid: str) -> int:
    with users_lock:
        sorted_users = sorted(users_db["users"].values(), key=lambda u: u["stats"].get("totalTokens", 0), reverse=True)
        for i, u in enumerate(sorted_users):
            if u["uuid"] == user_uuid:
                return i + 1
    return 0


# ============ CLIENT IDENTITY (DUAL-STACK SAFE) ============
def get_client_ip(request: Request) -> str:
    """
    Real network IP for bans / logging.
    Prefers Cloudflare's cf-connecting-ip so IPv4/IPv6 hits resolve to the true client.
    """
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_client_key(request: Request) -> str:
    """
    Stable client identity for round tracking, shard assignment, and active counts.
    - Authenticated → "user:<uuid>" (survives IPv4/IPv6 flip, re-login, IP change)
    - Anonymous    → "ip:<real-ip>" (best effort)
    This prevents one physical machine from being counted twice when it has both
    an IPv4 and an IPv6 connection (very common on home ISPs in 2024+).
    """
    user = get_current_user(request)
    if user:
        return f"user:{user['uuid']}"
    return f"ip:{get_client_ip(request)}"


# ============ SUBNET HELPERS (for bans) ============
def get_subnet(ip: str) -> str:
    if not ip or ip == "unknown":
        return "unknown"
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3]) + ".0/24"
    return ip


def is_blacklisted(ip: str) -> bool:
    subnet = get_subnet(ip)
    if subnet in blacklisted_subnets:
        if time.time() < blacklisted_subnets[subnet]:
            return True
        del blacklisted_subnets[subnet]
    return False


def ban_subnet(ip: str, reason: str, hours: int = 24):
    subnet = get_subnet(ip)
    blacklisted_subnets[subnet] = time.time() + hours * 3600
    log.warning(f"🔨 BANNED {subnet} for {hours}h — {reason}")


# ============ DATASET STATE ============
def save_dataset_state():
    try:
        with open(DATASET_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "datasetInfo": dataset_info,
                "dataPosition": data_position,
                "datasetStats": dataset_stats
            }, f, indent=2)
    except Exception as e:
        log.error(f"❌ save_dataset_state: {e}")


def load_dataset_state():
    global dataset_info, data_position, dataset_stats
    try:
        if DATASET_CACHE_FILE.exists():
            with open(DATASET_CACHE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            dataset_info.update(state.get("datasetInfo", {}))
            data_position.update(state.get("dataPosition", {}))
            dataset_stats.update(state.get("datasetStats", {}))
            log.info(f"📂 dataset state: epoch={data_position['epoch']} chunk={data_position['currentChunkIdx']}")
    except Exception as e:
        log.error(f"❌ load_dataset_state: {e}")


def initialize_dataset():
    if dataset_info["initialized"]:
        return
    dataset_info["totalChunks"] = DATASET_CONFIG.get("totalChunks", 100)
    chunks_dir = BASE_DIR / "chunks"
    if chunks_dir.exists():
        local_chunks = sorted(list(chunks_dir.glob("chunk_*.bin")))
        if local_chunks:
            dataset_info["totalChunks"] = len(local_chunks)
            log.info(f"✅ Auto-detected {dataset_info['totalChunks']} local chunks")
    dataset_info["initialized"] = True
    dataset_info["lastChecked"] = datetime.now(timezone.utc).isoformat()
    save_dataset_state()


# ============ SHARD MANAGEMENT ============
def _active_client_count() -> int:
    now = time.time()
    return sum(1 for ts in active_clients.values() if now - ts < FL_CONFIG["IP_TIMEOUT_S"])


def _advance_chunk_pointer():
    global data_position, dataset_stats
    data_position["currentChunkIdx"] += 1
    if data_position["currentChunkIdx"] >= dataset_info["totalChunks"]:
        data_position["currentChunkIdx"] = 0
        data_position["epoch"] += 1
        dataset_stats["epochsCompleted"] += 1
        log.info(f"🔄 epoch {data_position['epoch']} completed!")
    dataset_stats["totalShardsProcessed"] += 1
    dataset_stats["lastUpdate"] = datetime.now(timezone.utc).isoformat()


def get_or_advance_shard(client_key: str, force_new: bool = False) -> dict:
    global current_data_shard
    if not dataset_info["initialized"]:
        initialize_dataset()

    with shard_lock:
        shard_from_prev_round = (
            current_data_shard["shardId"] is not None
            and current_data_shard.get("round") != round_state["current_round"]
        )
        is_new_client = (current_data_shard["shardId"] is None) or (client_key not in current_data_shard["assignedClients"])
        enough_clients = len(current_data_shard["assignedClients"]) >= FL_CONFIG["CONSENSUS_REQUIRED"]
        need_new_shard = (
            current_data_shard["shardId"] is None
            or force_new
            or shard_from_prev_round
            or (is_new_client and enough_clients)
        )

        if need_new_shard:
            if current_data_shard["shardId"] is not None:
                _advance_chunk_pointer()

            shard_id = f"round_{round_state['current_round']}_{data_position['currentChunkIdx']}_{int(time.time()*1000)}"
            current_data_shard = {
                "chunkIdx": data_position["currentChunkIdx"],
                "assignedClients": [],
                "shardId": shard_id,
                "createdAt": int(time.time() * 1000),
                "round": round_state["current_round"],
            }
            db["shardMeta"][shard_id] = {
                "createdAt": int(time.time() * 1000),
                "required": FL_CONFIG["CONSENSUS_REQUIRED"],
                "chunkIdx": current_data_shard["chunkIdx"],
            }
            db["pendingSubmissions"][shard_id] = []
            log.info(
                f"📦 New shard {shard_id} chunk={current_data_shard['chunkIdx']} "
                f"(force={force_new}, prev_round={shard_from_prev_round}, new_client={is_new_client})"
            )

        if client_key not in current_data_shard["assignedClients"]:
            current_data_shard["assignedClients"].append(client_key)

        save_dataset_state()
        return current_data_shard


# ============ MODEL WEIGHTS & STATE ============
def initialize_model():
    global global_weights, global_engram_weights
    torch.manual_seed(42)
    global_weights = (torch.randn(MODEL_SIZE) * 0.02).float()
    global_engram_weights = (torch.randn(ENG_PARAMS) * 0.02).float()
    log.info(f"🎲 initialized {MODEL_SIZE:,} base weights + {ENG_PARAMS:,} engram weights (FP32 master)")


def load_global_state():
    global global_step, global_stats, lora_stats
    if not GLOBAL_STATE_FILE.exists():
        log.warning("⚠️ global_state.json not found. Starting fresh.")
        return
    try:
        with open(GLOBAL_STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
        global_step = state.get("globalStep", 0)
        if "globalStats" in state:
            global_stats.clear()
            global_stats.update(state["globalStats"])
            global_stats["lossHistory"] = global_stats.get("lossHistory", [])[-100:]
        if "loraStats" in state:
            lora_stats.clear()
            lora_stats.update(state["loraStats"])
        log.info(f"✅ SUCCESS: Loaded global_state.json -> Step: {global_step}, Tokens: {global_stats.get('totalTokens', 0):,}")
    except Exception as e:
        log.error(f"❌ CRITICAL: Failed to parse global_state.json: {e}")


def save_global_state():
    try:
        state_data = {
            "globalStep": global_step,
            "savedAt": datetime.now(timezone.utc).isoformat(),
            "modelSize": MODEL_SIZE,
            "engramParams": ENG_PARAMS,
            "modelConfig": MODEL_CONFIG,
            "globalStats": global_stats,
            "loraStats": lora_stats,
        }
        temp_file = GLOBAL_STATE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2)
        os.replace(temp_file, GLOBAL_STATE_FILE)
    except Exception as e:
        log.error(f"❌ Failed to save global_state.json: {e}")


def load_model_weights():
    global global_weights
    if not LATEST_CHECKPOINT_FILE.exists():
        log.warning("⚠️ model_latest.bin not found. Weights will be random.")
        return
    try:
        buf = np.fromfile(LATEST_CHECKPOINT_FILE, dtype=np.float32)
        if len(buf) == MODEL_SIZE:
            global_weights = torch.from_numpy(buf.copy())
            log.info("✅ SUCCESS: Loaded base model weights (fp32 master).")
        else:
            log.error(f"❌ CRITICAL: model_latest.bin is corrupted! Expected {MODEL_SIZE} params, got {len(buf)}.")
    except Exception as e:
        log.error(f"❌ Failed to load base model weights: {e}")


def load_engram_weights():
    global global_engram_weights
    if not ENGRAM_CHECKPOINT_FILE.exists():
        log.warning("⚠️ engram_latest.bin not found. Engram will be random.")
        return
    try:
        buf = np.fromfile(ENGRAM_CHECKPOINT_FILE, dtype=np.float32)
        if len(buf) == ENG_PARAMS:
            global_engram_weights = torch.from_numpy(buf.copy())
            log.info("✅ SUCCESS: Loaded engram weights (fp32 master).")
        else:
            log.error(f"❌ CRITICAL: engram_latest.bin is corrupted! Expected {ENG_PARAMS} params, got {len(buf)}.")
    except Exception as e:
        log.error(f"❌ Failed to load engram weights: {e}")


def save_model_weights():
    try:
        temp_file = LATEST_CHECKPOINT_FILE.with_suffix('.tmp')
        with open(temp_file, 'wb') as f:
            chunk_size = 10_000_000
            for i in range(0, MODEL_SIZE, chunk_size):
                chunk = global_weights[i:i + chunk_size].cpu().float().numpy()
                chunk.tofile(f)
                del chunk
        gc.collect()
        os.replace(temp_file, LATEST_CHECKPOINT_FILE)
    except Exception as e:
        log.error(f"❌ Failed to save base model weights: {e}")


def save_engram_weights():
    try:
        temp_file = ENGRAM_CHECKPOINT_FILE.with_suffix('.tmp')
        with open(temp_file, 'wb') as f:
            chunk_size = 10_000_000
            for i in range(0, ENG_PARAMS, chunk_size):
                chunk = global_engram_weights[i:i + chunk_size].cpu().float().numpy()
                chunk.tofile(f)
                del chunk
        gc.collect()
        os.replace(temp_file, ENGRAM_CHECKPOINT_FILE)
    except Exception as e:
        log.error(f"❌ Failed to save engram weights: {e}")


def save_latest_checkpoint():
    save_global_state()
    save_model_weights()
    save_engram_weights()


# ============ HUGGINGFACE DIRECT DOWNLOAD SUPPORT ============
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
    if commit_info is None:
        return None
    for attr in ("commit_oid", "oid", "commit_id", "sha"):
        v = getattr(commit_info, attr, None)
        if v:
            return str(v)
    url = getattr(commit_info, "commit_url", None)
    if url:
        return str(url).rstrip("/").split("/")[-1]
    return None


def _latest_hf_revision() -> Optional[str]:
    if not HF_ENABLED:
        return None
    try:
        # HEAD on resolve/main returns the commit SHA as a header (no pagination)
        r = requests.head(
            f"https://huggingface.co/{HF_REPO_ID}/resolve/main/config.json",
            allow_redirects=False,
            timeout=15,
        )
        # HF puts the resolved revision in `x-repo-commit`
        sha = r.headers.get("x-repo-commit")
        if sha and len(sha) >= 12:
            return sha

        # Fallback: read it from the redirect URL (resolve/<sha>/...)
        loc = r.headers.get("location", "")
        # e.g. /api/resolve-cache/.../Vxtzq/Crowd-v1/<sha>/config.json
        for part in loc.split("/"):
            if len(part) == 40 and all(c in "0123456789abcdef" for c in part):
                return part
    except Exception as e:
        log.warning(f"Could not determine latest HF revision: {e}")
    return None


def _get_hf_config_step(revision: Optional[str]) -> Optional[int]:
    if not revision:
        return None
    try:
        r = requests.get(_hf_resolve_url(revision, "config.json"), timeout=20, allow_redirects=True)
        if r.ok:
            j = r.json()
            gs = j.get("globalStep", None)
            if gs is not None:
                return int(gs)
    except Exception as e:
        log.warning(f"Could not read HF config.json: {e}")
    return None


def _refresh_hf_file_info(revision: Optional[str]) -> Dict[str, Dict]:
    info: Dict[str, Dict] = {}
    if not revision:
        return info

    for fn in HF_WEIGHT_FILES:
        try:
            r = requests.head(_hf_resolve_url(revision, fn), timeout=30, allow_redirects=True)
            if r.ok:
                size = int(r.headers.get("content-length", 0) or 0)
                if size > 0:
                    info[fn] = {
                        "size": size,
                        "etag": r.headers.get("etag"),
                    }
        except Exception as e:
            log.warning(f"Could not HEAD HF file {fn}: {e}")

    return info


def _set_hf_revision(revision: Optional[str], step: int) -> bool:
    global current_hf_revision, hf_revision_global_step, hf_files_info

    if not revision:
        return False

    info = _refresh_hf_file_info(revision)
    if len(info) < len(HF_WEIGHT_FILES):
        log.warning("HF file info incomplete; HF direct downloads disabled")
        return False

    with hf_info_lock:
        hf_files_info = info

    current_hf_revision = revision
    hf_revision_global_step = int(step)
    log.info(f"🤗 HF direct weights enabled: revision={revision[:12]} step={hf_revision_global_step}")
    return True


def _build_hf_weights_metadata(fmt: str) -> Optional[dict]:
    if fmt != "bf16":
        return None
    if not HF_ENABLED:
        return None
    if not current_hf_revision:
        return None
    if hf_revision_global_step != global_step:
        return None

    with hf_info_lock:
        info = dict(hf_files_info)

    files = {}
    for role, fn in (("model", "model_bf16.safetensors"), ("engram", "engram_bf16.safetensors")):
        fi = info.get(fn)
        if not fi or not fi.get("size"):
            return None
        files[role] = {
            "path": fn,
            "url": _hf_resolve_url(current_hf_revision, fn),
            "size": int(fi["size"]),
            "dtype": "bf16",
            "layout": "flat",
            "tensorKey": "weights",
        }

    return {
        "source": "huggingface",
        "repo": HF_REPO_ID,
        "revision": current_hf_revision,
        "format": "bf16",
        "files": files,
    }


def upload_to_huggingface(retries: int = 3) -> bool:
    global last_hf_upload_time

    if not HF_ENABLED:
        return False

    with hf_upload_lock:
        for attempt in range(1, retries + 1):
            temp_dir = None
            try:
                log.info(f"🤗 Uploading to HuggingFace: {HF_REPO_ID} (attempt {attempt}/{retries})")
                api = HfApi(token=HF_TOKEN)

                # Unique temp dir so concurrent callers never collide
                temp_dir = Path(tempfile.mkdtemp(prefix="hf_upload_", dir=str(CHECKPOINT_DIR)))

                w = global_weights.cpu().to(torch.bfloat16)
                save_file({"weights": w}, str(temp_dir / "model_bf16.safetensors"))
                del w
                gc.collect()

                w = global_weights.cpu().float()
                save_file({"weights": w}, str(temp_dir / "model_fp32.safetensors"))
                del w
                gc.collect()

                w = global_engram_weights.cpu().to(torch.bfloat16)
                save_file({"weights": w}, str(temp_dir / "engram_bf16.safetensors"))
                del w
                gc.collect()

                w = global_engram_weights.cpu().float()
                save_file({"weights": w}, str(temp_dir / "engram_fp32.safetensors"))
                del w
                gc.collect()

                config_data = {
                    "architecture": MODEL_CONFIG["architecture"],
                    "vocabSize": MODEL_CONFIG["vocabSize"],
                    "dim": MODEL_CONFIG["dim"],
                    "nLayers": MODEL_CONFIG["nLayers"],
                    "nHeads": MODEL_CONFIG["nHeads"],
                    "nKvHeads": MODEL_CONFIG["nKvHeads"],
                    "headDim": MODEL_CONFIG["headDim"],
                    "maxSeqLen": MODEL_CONFIG["maxSeqLen"],
                    "mlpHidden": MODEL_CONFIG["mlpHidden"],
                    "weightTying": MODEL_CONFIG["weightTying"],
                    "globalStep": global_step,
                    "modelSize": MODEL_SIZE,
                    "engramParams": ENG_PARAMS,
                    "uploadedAt": datetime.now(timezone.utc).isoformat(),
                }

                with open(temp_dir / "config.json", 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2)

                commit = api.upload_folder(
                    folder_path=str(temp_dir),
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    commit_message=f"Update weights & engram (step {global_step}, round {round_state['current_round']})",
                )

                rev = _extract_commit_sha(commit) or _latest_hf_revision()
                if rev and _set_hf_revision(rev, global_step):
                    last_hf_upload_time = time.time()
                    log.info(f"✅ Uploaded to {HF_REPO_ID} at step {global_step}, revision={rev[:12]}")
                    return True

                log.warning("Uploaded to HF, but could not verify revision/file sizes")

            except Exception as e:
                log.error(f"❌ HF upload failed (attempt {attempt}): {e}")
                if attempt < retries:
                    time.sleep(5 * attempt)
            finally:
                if temp_dir is not None and temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

        return False


def sync_huggingface_state():
    if not HF_ENABLED:
        return

    try:
        api = HfApi(token=HF_TOKEN)
        files = set(api.list_repo_files(HF_REPO_ID, repo_type="model"))

        required = HF_WEIGHT_FILES + [
            "model_fp32.safetensors",
            "engram_fp32.safetensors",
            "config.json",
        ]
        missing = [f for f in required if f not in files]

        if missing:
            log.warning(f"⚠️ Missing files on HuggingFace: {missing}. Uploading current weights to bootstrap...")
            upload_to_huggingface()
            return

        rev = _latest_hf_revision()
        if not rev:
            log.warning("Could not determine HF commit SHA; HF direct downloads disabled")
            return

        cfg_step = _get_hf_config_step(rev)
        if cfg_step is None:
            log.warning("Could not read HF config globalStep; HF direct downloads disabled")
            return

        if cfg_step == global_step:
            _set_hf_revision(rev, global_step)
        elif cfg_step < global_step and LATEST_CHECKPOINT_FILE.exists() and ENGRAM_CHECKPOINT_FILE.exists():
            log.warning(f"HF step {cfg_step} older than local {global_step}; syncing HF...")
            upload_to_huggingface()
        else:
            log.warning(f"HF step {cfg_step} != local {global_step}; HF direct downloads disabled until sync")

    except Exception as e:
        log.error(f"❌ HF sync failed: {e}")


# ============ STATUS JSON & HUGGINGFACE INTEGRATION ============
def update_and_save_status(push_hf: bool = True):
    loss_history = global_stats["lossHistory"]
    training_loss_change = (
        round(((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100, 1)
        if len(loss_history) >= 2 and loss_history[0] > 0
        else 0.0
    )
    active_gpus = _active_client_count()
    total_tokens = global_stats["totalTokens"]

    with users_lock:
        total_participants = len(users_db["users"])
        users_list = [
            {
                "uuid": u["uuid"],
                "username": u["username"],
                "contributions": u["stats"].get("contributions", 0),
                "totalTokens": u["stats"].get("totalTokens", 0),
                "totalSteps": u["stats"].get("totalSteps", 0),
                "trustScore": compute_trust_score(u["stats"]),
            }
            for u in users_db["users"].values()
        ]

    leaderboard_sorted = sorted(users_list, key=lambda x: x["totalTokens"], reverse=True)

    status_data = {
        "totalParticipants": total_participants,
        "totalContributions": lora_stats.get("total_submissions", 0),
        "activeGpus": active_gpus,
        "totalTokens": total_tokens,
        "trainingLossChange": training_loss_change,
        "globalStep": global_step,
        "lastLoss": global_stats["lastLoss"],
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "currentRound": round_state["current_round"],
        "hfRevision": current_hf_revision,
        "hfRevisionStep": hf_revision_global_step,
        "hfDirectDownloads": bool(_build_hf_weights_metadata("bf16")),
    }

    # Local writes are safe even under concurrent callers
    try:
        with open(BASE_DIR / "status.json", 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)
        with open(BASE_DIR / "leaderboard.json", 'w', encoding='utf-8') as f:
            json.dump(leaderboard_sorted, f, indent=2)
    except Exception as e:
        log.error(f"❌ Failed to save json files to server root: {e}")

    if not (push_hf and HF_ENABLED):
        return

    # Non-blocking: if another push is in progress, skip this one.
    acquired = status_upload_lock.acquire(blocking=False)
    if not acquired:
        log.info("⏭️  Skipping HF status upload (another push in progress)")
        return

    temp_dir = None
    try:
        api = HfApi(token=HF_TOKEN)

        # Unique per call — impossible for two threads to stomp on each other
        temp_dir = Path(tempfile.mkdtemp(prefix="hf_status_", dir=str(BASE_DIR)))

        with open(temp_dir / "status.json", 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)
        with open(temp_dir / "leaderboard.json", 'w', encoding='utf-8') as f:
            json.dump(leaderboard_sorted, f, indent=2)

        def format_tokens(n):
            if n >= 1e12:
                return f"{n/1e12:.1f}T"
            if n >= 1e9:
                return f"{n/1e9:.1f}B"
            if n >= 1e6:
                return f"{n/1e6:.1f}M"
            if n >= 1e3:
                return f"{n/1e3:.1f}K"
            return str(n)

        badge_data = {
            "schemaVersion": 1,
            "label": "Tokens Processed",
            "message": format_tokens(total_tokens),
            "color": "blue",
        }
        with open(temp_dir / "badge_tokens.json", 'w', encoding='utf-8') as f:
            json.dump(badge_data, f, indent=2)

        operations = [
            CommitOperationAdd(path_in_repo="status.json", path_or_fileobj=str(temp_dir / "status.json")),
            CommitOperationAdd(path_in_repo="leaderboard.json", path_or_fileobj=str(temp_dir / "leaderboard.json")),
            CommitOperationAdd(path_in_repo="badge_tokens.json", path_or_fileobj=str(temp_dir / "badge_tokens.json")),
        ]

        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type="model",
            operations=operations,
            commit_message=f"chore: update stats, leaderboard & badge ({badge_data['message']} tokens)",
        )
        log.info("✅ Uploaded status.json, leaderboard.json & badge to HF")
    except Exception as e:
        log.error(f"❌ HF status upload failed: {e}")
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        status_upload_lock.release()


def hf_status_loop():
    if not HF_ENABLED:
        return
    time.sleep(60)
    while True:
        try:
            update_and_save_status(push_hf=True)
        except Exception as e:
            log.error(f"❌ hf_status_loop error: {e}")
        time.sleep(300)


def emergency_save():
    log.info("🛑 Server shutting down, saving final state...")
    save_global_state()
    save_model_weights()
    save_engram_weights()
    save_users_db()
    try:
        update_and_save_status(push_hf=False)
    except Exception:
        pass


atexit.register(emergency_save)


# ============ VECTOR MATH ============
def compute_max_allowed_norm(local_steps, rank=None):
    steps = max(1, int(local_steps or 1))
    base = FL_CONFIG["DELTA_NORM_BASE"] * math.sqrt(steps / 100.0)
    rank_factor = np.sqrt(rank / LORA_CONFIG["FULL_RANK"]) if rank and rank < LORA_CONFIG["FULL_RANK"] else 1.0
    return base * rank_factor


def handle_checkpoints():
    try:
        fp = CHECKPOINT_DIR / f"model_step_{global_step}.bin"
        with open(fp, 'wb') as f:
            chunk_size = 10_000_000
            for i in range(0, MODEL_SIZE, chunk_size):
                chunk = global_weights[i:i + chunk_size].cpu().float().numpy()
                chunk.tofile(f)
                del chunk

        fp_eng = CHECKPOINT_DIR / f"engram_step_{global_step}.bin"
        with open(fp_eng, 'wb') as f:
            chunk_size = 10_000_000
            for i in range(0, ENG_PARAMS, chunk_size):
                chunk = global_engram_weights[i:i + chunk_size].cpu().float().numpy()
                chunk.tofile(f)
                del chunk
        gc.collect()

        save_model_weights()
        save_engram_weights()

        with open(CHECKPOINT_DIR / f"model_step_{global_step}.json", 'w', encoding='utf-8') as f:
            json.dump({
                "globalStep": global_step,
                "savedAt": datetime.now(timezone.utc).isoformat(),
                "modelSize": MODEL_SIZE,
                "engramParams": ENG_PARAMS,
            }, f, indent=2)

        files = sorted(
            [f for f in CHECKPOINT_DIR.iterdir() if f.name.startswith("model_step_") and f.suffix == ".bin"],
            key=lambda f: int(f.stem.split("_")[2])
        )
        while len(files) > FL_CONFIG["MAX_CHECKPOINTS"]:
            old = files.pop(0)
            old.unlink()
            old_eng = CHECKPOINT_DIR / old.name.replace("model_", "engram_")
            if old_eng.exists():
                old_eng.unlink()
            meta = CHECKPOINT_DIR / f"{old.stem}.json"
            if meta.exists():
                meta.unlink()
    except Exception as e:
        log.error(f"❌ checkpoint: {e}")


# ============ CHUNKED UPLOAD STATE ============
_active_uploads: Dict[str, dict] = {}
uploads_lock = threading.RLock()


def _valid_upload_id(uid) -> bool:
    if not isinstance(uid, str) or len(uid) != 32:
        return False
    return all(c in "0123456789abcdef" for c in uid.lower())


def _cleanup_expired_uploads():
    now = time.time()
    with uploads_lock:
        for uid, up in list(_active_uploads.items()):
            if up.get("status") == "finalizing":
                continue
            last_activity = up.get("last_activity", up.get("created", 0))
            if now - last_activity > UPLOAD_TTL_SECONDS:
                try:
                    shutil.rmtree(up["dir"], ignore_errors=True)
                except Exception:
                    pass
                del _active_uploads[uid]

    try:
        for p in UPLOAD_DIR.iterdir():
            if p.is_dir():
                if now - p.stat().st_mtime > UPLOAD_TTL_SECONDS:
                    shutil.rmtree(p, ignore_errors=True)
            else:
                if now - p.stat().st_mtime > UPLOAD_TTL_SECONDS:
                    p.unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Upload dir cleanup warning: {e}")


# ============ BACKGROUND THREADS ============
def cleanup_loop():
    while True:
        time.sleep(FL_CONFIG["CLEANUP_INTERVAL_S"])
        now = time.time()
        now_ms = int(now * 1000)

        for key in list(active_clients.keys()):
            if now - active_clients[key] > FL_CONFIG["IP_TIMEOUT_S"]:
                del active_clients[key]

        for sn in list(blacklisted_subnets.keys()):
            if now > blacklisted_subnets[sn]:
                del blacklisted_subnets[sn]

        task_timeout_ms = 72 * 60 * 60 * 1000
        expired_tasks = [tid for tid, meta in db["tasks"].items() if now_ms - meta.get("createdAt", 0) > task_timeout_ms]
        for tid in expired_tasks:
            del db["tasks"][tid]

        stale_shards = [sid for sid, meta in db["shardMeta"].items() if now_ms - meta.get("createdAt", 0) > FL_CONFIG["SHARD_TIMEOUT_S"] * 1000]
        for sid in stale_shards:
            db["shardMeta"].pop(sid, None)
            db["pendingSubmissions"].pop(sid, None)

        for f in DELTA_TEMP_DIR.glob("delta_*.bin"):
            if time.time() - f.stat().st_mtime > 18000:
                try:
                    f.unlink()
                except Exception:
                    pass

        _cleanup_expired_uploads()

        if expired_tasks or stale_shards:
            log.info(f"🧹 GC: Cleaned {len(expired_tasks)} dead tasks and {len(stale_shards)} stale shards")

        gc.collect()


def round_watchdog():
    while True:
        try:
            time.sleep(30)

            should_end_round = False
            pending_deltas = []
            current_round_ending = 1

            with shard_lock:
                round_elapsed_hours = (time.time() - round_state["round_start_time"]) / 3600

                if round_elapsed_hours >= round_state["round_duration_hours"] * ROUND_CONFIG["ULTIMATUM_THRESHOLD"]:
                    for client_key, client_data in list(round_state["participating_clients"].items()):
                        if not client_data["submitted"] and not client_data.get("ultimatum_sent"):
                            last_seen = active_clients.get(client_key, 0)
                            if time.time() - last_seen < FL_CONFIG["IP_TIMEOUT_S"]:
                                continue

                            log.warning(f"⚠️ Sending ultimatum to {client_key} (silent for {int((time.time()-last_seen)/60)}m)")
                            client_data["ultimatum_sent"] = True
                            client_data["ultimatum_time"] = time.time()

                for client_key in list(round_state["participating_clients"].keys()):
                    client_data = round_state["participating_clients"][client_key]

                    if client_data["submitted"] and client_data.get("ultimatum_sent"):
                        client_data["ultimatum_sent"] = False
                        continue

                    if client_data.get("ultimatum_sent"):
                        time_since_ultimatum = (time.time() - client_data.get("ultimatum_time", 0)) / 60

                        last_seen = active_clients.get(client_key, 0)
                        if time.time() - last_seen < FL_CONFIG["IP_TIMEOUT_S"]:
                            client_data["ultimatum_sent"] = False
                            continue

                        if time_since_ultimatum > ROUND_CONFIG["ULTIMATUM_TIMEOUT_MIN"]:
                            log.warning(f"🚫 Removing {client_key} from round (unresponsive/dead)")
                            del round_state["participating_clients"][client_key]

                timer_expired = round_elapsed_hours >= round_state["round_duration_hours"]
                should_end_round = timer_expired

                if should_end_round:
                    round_state["is_aggregating"] = True
                    pending_deltas = list(round_state["pending_deltas"])
                    round_state["pending_deltas"] = []
                    current_round_ending = round_state["current_round"]
                    log.info(f"⏰ Round {current_round_ending} timer expired ({round_elapsed_hours:.2f}h)! Aggregating {len(pending_deltas)} deltas...")

            if should_end_round:
                try:
                    if pending_deltas:
                        apply_round_aggregation(pending_deltas)
                    else:
                        log.warning("⚠️ Round ended but no deltas were submitted.")

                    hf_ok = False
                    if HF_ENABLED:
                        hf_ok = upload_to_huggingface(retries=3)
                        if hf_ok:
                            log.info("✅ HF weights synced for new round")
                        else:
                            log.warning("❌ HF weights sync failed; clients may fallback to coordinator weights")

                    update_and_save_status(push_hf=True)

                    with shard_lock:
                        round_state["current_round"] += 1
                        round_state["round_start_time"] = time.time()
                        round_state["participating_clients"] = {}
                        round_state["aggregations_this_round"] = 0
                        round_state["last_aggregation_time"] = time.time()
                        round_state["is_aggregating"] = False

                        try:
                            for f in DELTA_TEMP_DIR.glob("delta_*.bin"):
                                f.unlink()
                            log.info("🧹 Wiped leftover deltas for new round.")
                        except Exception:
                            pass

                        log.info(f"🚀 Started round {round_state['current_round']} with updated weights (step {global_step})")

                except Exception as agg_err:
                    log.error(f"❌ Aggregation Failed: {agg_err}")
                    import traceback
                    traceback.print_exc()
                    with shard_lock:
                        round_state["is_aggregating"] = False
                        round_state["current_round"] += 1
                        round_state["round_start_time"] = time.time()
                        round_state["participating_clients"] = {}

        except Exception as e:
            log.error(f"❌ round_watchdog error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)


threading.Thread(target=cleanup_loop, daemon=True).start()
threading.Thread(target=round_watchdog, daemon=True).start()
threading.Thread(target=hf_status_loop, daemon=True).start()


# ============ SUBMISSION PROCESSING ============
def process_decompressed_payload(dec_path: str, client_ip: str, client_key: str, request: Request, init_meta: Optional[dict] = None):
    temp_delta_path = None
    final_delta_path = None

    try:
        with open(dec_path, 'rb') as dec_f:
            payload_len_bytes = dec_f.read(4)
            if len(payload_len_bytes) < 4:
                raise HTTPException(status_code=400, detail="Invalid payload")

            payload_len = struct.unpack('<I', payload_len_bytes)[0]
            payload_bytes = dec_f.read(payload_len)
            if len(payload_bytes) < payload_len:
                raise HTTPException(status_code=400, detail="Incomplete JSON payload")

            body = json.loads(payload_bytes.decode('utf-8'))

            if init_meta:
                for k in ("taskId", "weightFormat", "hasEngram", "engramSparseCount"):
                    if k in init_meta and body.get(k) != init_meta.get(k):
                        raise HTTPException(status_code=400, detail=f"Chunked init metadata mismatch: {k}")

            fmt = body.get("weightFormat", "bf16")
            if fmt not in ("bf16", "fp16", "fp8", "fp32"):
                raise HTTPException(status_code=400, detail=f"Unsupported weight format: {fmt}")

            has_engram = bool(body.get("hasEngram", False))
            try:
                engram_sparse_count = int(body.get("engramSparseCount", 0) or 0)
            except Exception:
                engram_sparse_count = 0

            if engram_sparse_count < 0 or engram_sparse_count > ENG_NUM_BUCKETS:
                raise HTTPException(status_code=400, detail="Invalid engramSparseCount")

            bpp = _bytes_per_param(fmt)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".delta") as temp_delta_file:
                temp_delta_path = temp_delta_file.name

            norm_sq = 0.0
            offset = 0
            has_nan_inf = False

            with open(temp_delta_path, 'wb') as td:
                while offset < MODEL_SIZE:
                    read_params = min(10_000_000, MODEL_SIZE - offset)
                    chunk_bytes = dec_f.read(read_params * bpp)
                    if not chunk_bytes or len(chunk_bytes) < read_params * bpp:
                        raise HTTPException(status_code=400, detail="Incomplete base delta payload")

                    td.write(chunk_bytes)
                    chunk_t = _decode_delta_chunk(bytearray(chunk_bytes), fmt)

                    if torch.isnan(chunk_t).any() or torch.isinf(chunk_t).any():
                        has_nan_inf = True
                        break

                    norm_sq += torch.sum(chunk_t * chunk_t).item()
                    offset += read_params
                    del chunk_t

            if has_nan_inf:
                ban_subnet(client_ip, "submitted NaN/Inf delta")
                raise HTTPException(status_code=400, detail="Invalid delta (NaN/Inf)")

            delta_norm = math.sqrt(norm_sq)

            engram_indices = None
            engram_values = None

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
                if fmt == "fp16":
                    engram_values = torch.frombuffer(val_buf, dtype=torch.uint16).view(torch.float16).to(torch.bfloat16)
                elif fmt == "fp32":
                    engram_values = torch.frombuffer(val_buf, dtype=torch.float32).to(torch.bfloat16)
                elif fmt == "fp8":
                    engram_values = torch.frombuffer(val_buf, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
                else:
                    engram_values = torch.frombuffer(val_buf, dtype=torch.uint16).view(torch.bfloat16)

                engram_values = engram_values.reshape(-1, MODEL_CONFIG["dim"]).contiguous()
                del val_buf

            task_id = body.get("taskId")
            if task_id not in db["tasks"]:
                raise HTTPException(status_code=404, detail="Task expired")

            try:
                rank = int(body.get("loraRank", LORA_CONFIG["FULL_RANK"]) or LORA_CONFIG["FULL_RANK"])
            except Exception:
                rank = LORA_CONFIG["FULL_RANK"]

            if rank <= 0:
                rank = LORA_CONFIG["FULL_RANK"]

            try:
                local_steps = int(body.get("localSteps", 1) or 1)
            except Exception:
                local_steps = 1

            if local_steps <= 0:
                local_steps = 1

            loss_raw = body.get("loss")
            loss_float = None
            try:
                if loss_raw is not None:
                    loss_float = float(loss_raw)
                    if math.isnan(loss_float) or math.isinf(loss_float):
                        loss_float = None
            except Exception:
                loss_float = None

            max_allowed = compute_max_allowed_norm(local_steps, rank)
            abs_max = max(FL_CONFIG["DELTA_NORM_ABSOLUTE_MAX"], max_allowed * 3)

            if delta_norm > abs_max:
                ban_subnet(client_ip, f"delta norm {delta_norm:.0f} > max {abs_max:.0f}")
                raise HTTPException(status_code=400, detail=f"Delta too large ({delta_norm:.2f})")

            ts = int(time.time() * 1000)
            safe_key = client_key.replace(":", "_").replace(".", "_")
            final_delta_path = str(DELTA_TEMP_DIR / f"delta_{safe_key}_{ts}.bin")

            delta_metadata = {
                "nodeKey": client_key,
                "nodeIp": client_ip,
                "delta_norm": delta_norm,
                "rank": rank,
                "loss": loss_float,
                "localSteps": local_steps,
                "tokensProcessed": int(body.get("tokensProcessed", 0) or 0),
                "normalized": False,
                "submittedAt": int(time.time() * 1000),
                "weightFormat": "bf16",
            }

            if delta_norm > max_allowed:
                scale_factor = max_allowed / delta_norm
                delta_metadata["normalized"] = True

                with open(temp_delta_path, 'rb') as in_f, open(final_delta_path, 'wb') as out_f:
                    meta_bytes = json.dumps(delta_metadata).encode('utf-8')
                    out_f.write(struct.pack('<I', len(meta_bytes)))
                    out_f.write(meta_bytes)

                    offset = 0
                    while offset < MODEL_SIZE:
                        read_params = min(10_000_000, MODEL_SIZE - offset)
                        chunk_bytes = in_f.read(read_params * bpp)
                        if not chunk_bytes or len(chunk_bytes) < read_params * bpp:
                            raise HTTPException(status_code=400, detail="Incomplete delta during normalization")

                        chunk_t = _decode_delta_chunk(bytearray(chunk_bytes), fmt)
                        chunk_t.mul_(scale_factor)
                        out_f.write(_encode_float32_to_bf16_bytes(chunk_t))
                        offset += read_params
                        del chunk_t

                    if engram_indices is not None and len(engram_indices) > 0:
                        out_f.write(struct.pack('<I', len(engram_indices)))
                        out_f.write(engram_indices.tobytes())
                        eng_scaled = (engram_values.float() * scale_factor).to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
                        out_f.write(eng_scaled)
                    else:
                        out_f.write(struct.pack('<I', 0))
            else:
                delta_metadata["normalized"] = False

                with open(temp_delta_path, 'rb') as in_f, open(final_delta_path, 'wb') as out_f:
                    meta_bytes = json.dumps(delta_metadata).encode('utf-8')
                    out_f.write(struct.pack('<I', len(meta_bytes)))
                    out_f.write(meta_bytes)

                    offset = 0
                    while offset < MODEL_SIZE:
                        read_params = min(10_000_000, MODEL_SIZE - offset)
                        chunk_bytes = in_f.read(read_params * bpp)
                        if not chunk_bytes or len(chunk_bytes) < read_params * bpp:
                            raise HTTPException(status_code=400, detail="Incomplete delta during copy")

                        if fmt == "bf16":
                            out_f.write(chunk_bytes)
                        else:
                            chunk_t = _decode_delta_chunk(bytearray(chunk_bytes), fmt)
                            out_f.write(_encode_float32_to_bf16_bytes(chunk_t))
                            del chunk_t

                        offset += read_params

                    if engram_indices is not None and len(engram_indices) > 0:
                        out_f.write(struct.pack('<I', len(engram_indices)))
                        out_f.write(engram_indices.tobytes())
                        out_f.write(engram_values.contiguous().view(torch.uint16).numpy().tobytes())
                    else:
                        out_f.write(struct.pack('<I', 0))

            if temp_delta_path and os.path.exists(temp_delta_path):
                os.unlink(temp_delta_path)
            temp_delta_path = None

            submitted_path = None
            with shard_lock:
                if client_key not in round_state["participating_clients"]:
                    round_state["participating_clients"][client_key] = {
                        "joined": time.time(),
                        "submitted": False,
                        "ultimatum_sent": False,
                    }

                round_state["participating_clients"][client_key]["submitted"] = True
                round_state["participating_clients"][client_key]["ultimatum_sent"] = False
                round_state["aggregations_this_round"] += 1
                round_state["pending_deltas"].append(final_delta_path)
                submitted_path = final_delta_path
                final_delta_path = None

                log.info(f"📥 Round {round_state['current_round']}: {client_key} submitted → saved to disk (norm={delta_norm:.2f})")

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
                tokens_to_add = delta_metadata["tokensProcessed"]
                steps_to_add = local_steps
                with users_lock:
                    user_record = users_db["users"].get(auth_user["uuid"])
                    if user_record:
                        user_record["stats"]["contributions"] = user_record["stats"].get("contributions", 0) + 1
                        user_record["stats"]["totalTokens"] = user_record["stats"].get("totalTokens", 0) + tokens_to_add
                        user_record["stats"]["totalSteps"] = user_record["stats"].get("totalSteps", 0) + steps_to_add
                        save_users_db()
                        log.info(
                            f"✅ User {user_record['username']} credited: "
                            f"+{tokens_to_add:,} tokens, +{steps_to_add} steps "
                            f"(total: {user_record['stats']['totalTokens']:,})"
                        )
                    else:
                        log.warning(f"⚠️ Auth user {auth_user['username']} not found in DB — token may be stale")
            else:
                log.info(f"📝 Anonymous submission from {client_ip} (no auth header or invalid token)")

            save_global_state()
            update_and_save_status(push_hf=False)

            return {
                "status": "success",
                "globalStep": global_step,
                "normalized": delta_metadata["normalized"],
                "round": round_state["current_round"],
                "loss": loss_float,
                "savedPath": submitted_path,
            }

    finally:
        if temp_delta_path and os.path.exists(temp_delta_path):
            try:
                os.unlink(temp_delta_path)
            except Exception:
                pass
        if final_delta_path and os.path.exists(final_delta_path):
            try:
                os.unlink(final_delta_path)
            except Exception:
                pass


def process_compressed_payload(compressed_path: str, client_ip: str, client_key: str, request: Request, init_meta: Optional[dict] = None):
    temp_decompressed = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dec") as dec_file:
            temp_decompressed = dec_file.name

        dec_size = 0

        with open(compressed_path, 'rb') as f:
            magic = f.read(2)
            f.seek(0)

            with open(temp_decompressed, 'wb') as dec:
                if len(magic) >= 2 and magic[0] == 0x1f and magic[1] == 0x8b:
                    with gzip.open(f, 'rb') as gz:
                        while True:
                            chunk = gz.read(1024 * 1024)
                            if not chunk:
                                break
                            dec_size += len(chunk)
                            if dec_size > MAX_DECOMPRESSED_BYTES:
                                raise HTTPException(status_code=413, detail="Decompressed payload too large")
                            dec.write(chunk)
                else:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        dec_size += len(chunk)
                        if dec_size > MAX_DECOMPRESSED_BYTES:
                            raise HTTPException(status_code=413, detail="Payload too large")
                        dec.write(chunk)

        return process_decompressed_payload(temp_decompressed, client_ip, client_key, request, init_meta)

    finally:
        if temp_decompressed and os.path.exists(temp_decompressed):
            try:
                os.unlink(temp_decompressed)
            except Exception:
                pass


# ============ FASTAPI ============
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="CrowdGPT Continuous FL Coordinator")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Task-Id", "X-Shard-Id", "X-Chunk-Idx", "X-Global-Step",
        "X-Model-Size", "X-Quantized", "X-Round", "X-Status",
    ],
)


@app.get("/fl/heartbeat")
async def heartbeat(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        return Response(content='{"status":"blocked"}', media_type="application/json")

    client_key = get_client_key(request)
    active_clients[client_key] = time.time()
    return Response(content='{"status":"ok"}', media_type="application/json")


@app.get("/fl/round_status")
async def get_round_status(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")

    client_key = get_client_key(request)
    active_clients[client_key] = time.time()

    with shard_lock:
        if client_key not in round_state["participating_clients"]:
            round_state["participating_clients"][client_key] = {
                "joined": time.time(),
                "submitted": False,
                "ultimatum_sent": False,
            }
            log.info(f"👥 Client {client_key} joined round {round_state['current_round']}")

    in_cooldown = (time.time() - round_state["last_aggregation_time"]) < ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"]
    round_elapsed_hours = (time.time() - round_state["round_start_time"]) / 3600

    time_left_in_server_round = round_state["round_duration_hours"] - round_elapsed_hours
    if time_left_in_server_round < 0.25:
        client_max_round = round_state["round_duration_hours"]
    else:
        client_max_round = round_elapsed_hours + ROUND_CONFIG["TARGET_SUBMISSION_HOURS"]

    return {
        "current_round": round_state["current_round"],
        "round_elapsed_hours": round_elapsed_hours,
        "max_round_hours": client_max_round,
        "round_duration_hours": round_state["round_duration_hours"],
        "clients_in_round": len(round_state["participating_clients"]),
        "submitted_clients": sum(1 for c in round_state["participating_clients"].values() if c["submitted"]),
        "in_cooldown": in_cooldown,
        "global_step": global_step,
        "next_round_ready": not in_cooldown and round_state["last_aggregation_time"] > 0,
        "is_aggregating": round_state.get("is_aggregating", False),
        "hfDirectDownloads": bool(_build_hf_weights_metadata("bf16")),
    }


@app.get("/fl/task")
async def get_task(request: Request, format: str = "bf16", skip_weights: bool = False):
    if round_state.get("is_aggregating", False):
        return Response(
            content=json.dumps({"status": "wait", "message": "Server aggregating and uploading to HF, please wait"}).encode(),
            media_type="application/json",
            headers={"X-Status": "wait"},
        )

    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")

    client_key = get_client_key(request)
    active_clients[client_key] = time.time()

    in_cooldown = (time.time() - round_state["last_aggregation_time"]) < ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"]
    if in_cooldown:
        wait_time = int(ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"] - (time.time() - round_state["last_aggregation_time"]))
        return Response(
            content=json.dumps({"status": "wait", "message": f"Server aggregating, wait {wait_time}s"}).encode(),
            media_type="application/json",
            headers={"X-Status": "wait"},
        )

    shard = get_or_advance_shard(client_key, force_new=skip_weights)
    shard_id = shard["shardId"]
    task_id = f"round_{round_state['current_round']}_{shard_id}"

    with shard_lock:
        if client_key in round_state["participating_clients"]:
            round_state["participating_clients"][client_key]["submitted"] = False
            round_state["participating_clients"][client_key]["ultimatum_sent"] = False

    db["tasks"][task_id] = {
        "shardId": shard_id,
        "chunkIdx": shard["chunkIdx"],
        "round": round_state["current_round"],
        "requestedByKey": client_key,
        "requestedByIp": client_ip,
        "createdAt": int(time.time() * 1000),
    }

    metadata = {
        "taskId": task_id,
        "globalStep": global_step,
        "currentRound": round_state["current_round"],
        "modelSize": MODEL_SIZE,
        "engramParams": ENG_PARAMS,
        "modelConfig": MODEL_CONFIG,
        "weightFormat": format,
        "datasetConfig": {
            "repoId": DATASET_CONFIG["repoId"],
            "chunkIdx": shard["chunkIdx"],
            "totalChunks": dataset_info["totalChunks"],
            "subChunkSize": DATASET_CONFIG["subChunkMB"] * 1024 * 1024,
            "epoch": data_position["epoch"],
            "tokensPerSample": DATASET_CONFIG["tokensPerSample"],
            "tokenBytes": DATASET_CONFIG["tokenBytes"],
        },
        "shardConfig": {
            "shardId": shard_id,
            "slot": len(shard["assignedClients"]),
            "total": FL_CONFIG["CONSENSUS_REQUIRED"],
        },
        "roundConfig": {
            "maxHours": round_state["round_duration_hours"],
            "targetDurationHours": round_state["round_duration_hours"],
        },
    }

    hf_weights = _build_hf_weights_metadata(format)
    if hf_weights:
        metadata["weights"] = hf_weights
    else:
        metadata["weights"] = {
            "source": "coordinator",
            "reason": "hf_unavailable_or_step_mismatch",
        }

    if skip_weights:
        return JSONResponse(
            metadata,
            headers={
                "X-Task-Id": task_id,
                "X-Round": str(round_state["current_round"]),
                "X-Global-Step": str(global_step),
            },
        )

    metadata_json = json.dumps(metadata).encode('utf-8')
    metadata_length = struct.pack('<I', len(metadata_json))

    def weight_stream():
        yield metadata_length + metadata_json
        chunk_size = 10_000_000

        for i in range(0, MODEL_SIZE, chunk_size):
            chunk = global_weights[i:i + chunk_size].cpu()
            if format == "bf16":
                yield chunk.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
            else:
                yield chunk.float().numpy().astype(np.float16).tobytes()
            del chunk

        for i in range(0, ENG_PARAMS, chunk_size):
            chunk = global_engram_weights[i:i + chunk_size].cpu()
            if format == "bf16":
                yield chunk.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
            else:
                yield chunk.float().numpy().astype(np.float16).tobytes()
            del chunk

    return StreamingResponse(
        weight_stream(),
        media_type="application/octet-stream",
        headers={
            "X-Task-Id": task_id,
            "X-Round": str(round_state["current_round"]),
            "X-Global-Step": str(global_step),
        },
    )


@app.post("/fl/submit")
async def submit_delta(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")

    if round_state.get("is_aggregating", False):
        raise HTTPException(status_code=503, detail="Server is aggregating, try again later")

    client_key = get_client_key(request)
    active_clients[client_key] = time.time()

    temp_compressed = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin.gz") as tmp_file:
            temp_compressed = tmp_file.name

        size = 0
        with open(temp_compressed, 'wb') as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_DIRECT_BODY_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Direct upload too large. Use chunked upload: /fl/submit/init"
                    )
                f.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Empty upload")

        return process_compressed_payload(temp_compressed, client_ip, client_key, request, init_meta=None)

    finally:
        if temp_compressed and os.path.exists(temp_compressed):
            try:
                os.unlink(temp_compressed)
            except Exception:
                pass


@app.post("/fl/submit/init")
async def submit_init(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")

    if round_state.get("is_aggregating", False):
        raise HTTPException(status_code=503, detail="Server is aggregating, try again later")

    client_key = get_client_key(request)
    active_clients[client_key] = time.time()

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON init payload")

    required = [
        "taskId",
        "totalBytes",
        "chunkSize",
        "chunks",
        "checksum",
        "weightFormat",
        "hasEngram",
        "engramSparseCount",
    ]
    for k in required:
        if k not in data:
            raise HTTPException(status_code=400, detail=f"Missing init field: {k}")

    task_id = str(data["taskId"])
    if task_id not in db["tasks"]:
        raise HTTPException(status_code=404, detail="Task expired")

    try:
        total_bytes = int(data["totalBytes"])
        chunk_size = int(data["chunkSize"])
        chunks = int(data["chunks"])
        engram_sparse_count = int(data.get("engramSparseCount", 0) or 0)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid numeric init fields")

    if total_bytes <= 0 or total_bytes > MAX_TOTAL_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Invalid totalBytes")

    if chunk_size <= 0 or chunk_size > MAX_CHUNK_BYTES:
        raise HTTPException(status_code=400, detail=f"Invalid chunkSize (max {MAX_CHUNK_BYTES})")

    if chunks <= 0 or chunks > MAX_CHUNKS:
        raise HTTPException(status_code=400, detail=f"Invalid chunks (max {MAX_CHUNKS})")

    if (chunks - 1) * chunk_size >= total_bytes or chunks * chunk_size < total_bytes:
        raise HTTPException(status_code=400, detail="Inconsistent chunks/chunkSize/totalBytes")

    checksum = str(data["checksum"]).lower()
    if len(checksum) != 64 or not all(c in "0123456789abcdef" for c in checksum):
        raise HTTPException(status_code=400, detail="Invalid checksum, expected sha256 hex")

    fmt = str(data.get("weightFormat", "bf16"))
    if fmt not in ("bf16", "fp16", "fp8", "fp32"):
        raise HTTPException(status_code=400, detail=f"Unsupported weight format: {fmt}")

    has_engram = bool(data.get("hasEngram", False))
    if engram_sparse_count < 0 or engram_sparse_count > ENG_NUM_BUCKETS:
        raise HTTPException(status_code=400, detail="Invalid engramSparseCount")

    auth_user = get_current_user(request)

    upload_id = uuid.uuid4().hex
    up_dir = UPLOAD_DIR / upload_id
    up_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "taskId": task_id,
        "totalBytes": total_bytes,
        "chunkSize": chunk_size,
        "chunks": chunks,
        "checksum": checksum,
        "weightFormat": fmt,
        "hasEngram": has_engram,
        "engramSparseCount": engram_sparse_count,
        "loss": data.get("loss"),
        "localSteps": data.get("localSteps"),
        "tokensProcessed": data.get("tokensProcessed"),
        "loraRank": data.get("loraRank"),
        "isDelta": data.get("isDelta", True),
    }

    with uploads_lock:
        _active_uploads[upload_id] = {
            "dir": up_dir,
            "created": time.time(),
            "last_activity": time.time(),
            "status": "open",
            "client_key": client_key,
            "client_ip": client_ip,
            "user_uuid": auth_user["uuid"] if auth_user else None,
            "chunks": chunks,
            "chunk_size": chunk_size,
            "total_bytes": total_bytes,
            "checksum": checksum,
            "meta": meta,
            "received": set(),
            "chunk_sizes": {},
            "bytes_received": 0,
        }

    return {
        "uploadId": upload_id,
        "chunkSize": chunk_size,
        "chunks": chunks,
        "totalBytes": total_bytes,
    }


@app.post("/fl/submit/chunk")
async def submit_chunk(request: Request, uploadId: str, index: int):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")

    if not _valid_upload_id(uploadId):
        raise HTTPException(status_code=400, detail="Invalid uploadId")

    with uploads_lock:
        up = _active_uploads.get(uploadId)
        if not up:
            raise HTTPException(status_code=404, detail="Unknown uploadId")
        if up.get("status") != "open":
            raise HTTPException(status_code=409, detail="Upload is not open")
        if index < 0 or index >= up["chunks"]:
            raise HTTPException(status_code=400, detail="Chunk index out of range")

        # Verify ownership by user UUID when available, fall back to client_key
        cur_user = get_current_user(request)
        if up.get("user_uuid") is not None:
            if not cur_user or cur_user["uuid"] != up["user_uuid"]:
                raise HTTPException(status_code=403, detail="Upload owner mismatch")
        else:
            # anonymous upload: check by client_key
            cur_key = get_client_key(request)
            if up.get("client_key") and cur_key != up["client_key"]:
                raise HTTPException(status_code=403, detail="Upload owner mismatch")

        up_dir = up["dir"]
        up["last_activity"] = time.time()

    part_tmp = up_dir / f"part_{index:06d}.tmp"
    part_final = up_dir / f"part_{index:06d}.bin"

    h = hashlib.sha256()
    size = 0

    try:
        with open(part_tmp, 'wb') as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_CHUNK_BYTES:
                    raise HTTPException(status_code=413, detail="Chunk too large")
                f.write(chunk)
                h.update(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Empty chunk")

        expected_sha = request.headers.get("X-Chunk-Sha256")
        if expected_sha:
            if expected_sha.lower() != h.hexdigest():
                raise HTTPException(status_code=400, detail="Chunk checksum mismatch")

        os.replace(part_tmp, part_final)

        with uploads_lock:
            up = _active_uploads.get(uploadId)
            if not up:
                raise HTTPException(status_code=404, detail="Unknown uploadId")

            old_size = up["chunk_sizes"].get(index, 0)
            up["chunk_sizes"][index] = size
            up["bytes_received"] = up["bytes_received"] - old_size + size
            up["received"].add(index)
            up["last_activity"] = time.time()

        return {
            "ok": True,
            "index": index,
            "size": size,
            "receivedChunks": len(up["received"]),
            "totalChunks": up["chunks"],
        }

    finally:
        try:
            if part_tmp.exists():
                part_tmp.unlink()
        except Exception:
            pass


@app.post("/fl/submit/finalize")
async def submit_finalize(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON finalize payload")

    upload_id = data.get("uploadId")
    if not _valid_upload_id(upload_id):
        raise HTTPException(status_code=400, detail="Invalid uploadId")

    with uploads_lock:
        up = _active_uploads.get(upload_id)
        if not up:
            raise HTTPException(status_code=404, detail="Unknown uploadId")
        if up.get("status") != "open":
            raise HTTPException(status_code=409, detail="Upload is not open")

        cur_user = get_current_user(request)
        if up.get("user_uuid") is not None:
            if not cur_user or cur_user["uuid"] != up["user_uuid"]:
                raise HTTPException(status_code=403, detail="Upload owner mismatch")
        else:
            cur_key = get_client_key(request)
            if up.get("client_key") and cur_key != up["client_key"]:
                raise HTTPException(status_code=403, detail="Upload owner mismatch")

        if len(up["received"]) != up["chunks"]:
            raise HTTPException(status_code=400, detail="Missing chunks")

        up["status"] = "finalizing"
        up["last_activity"] = time.time()

    client_key = up.get("client_key") or get_client_key(request)

    compressed_path = UPLOAD_DIR / f"{upload_id}_compressed.bin.gz"
    completed = False

    try:
        h = hashlib.sha256()
        total = 0

        with open(compressed_path, 'wb') as out:
            for i in range(up["chunks"]):
                part = up["dir"] / f"part_{i:06d}.bin"
                if not part.exists():
                    raise HTTPException(status_code=400, detail=f"Missing chunk {i}")

                with open(part, 'rb') as inp:
                    while True:
                        b = inp.read(4 * 1024 * 1024)
                        if not b:
                            break
                        total += len(b)
                        if total > up["total_bytes"]:
                            raise HTTPException(status_code=400, detail="Reassembled payload too large")
                        out.write(b)
                        h.update(b)

        if total != up["total_bytes"]:
            raise HTTPException(status_code=400, detail="Total size mismatch")

        if h.hexdigest() != up["checksum"]:
            raise HTTPException(status_code=400, detail="Payload checksum mismatch")

        result = process_compressed_payload(str(compressed_path), client_ip, client_key, request, up["meta"])
        completed = True
        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"❌ Finalize failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Finalize failed: {str(e)}")

    finally:
        try:
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)
        except Exception:
            pass

        with uploads_lock:
            up2 = _active_uploads.get(upload_id)
            if up2:
                if completed:
                    try:
                        shutil.rmtree(up2["dir"], ignore_errors=True)
                    except Exception:
                        pass
                    del _active_uploads[upload_id]
                else:
                    up2["status"] = "open"
                    up2["last_activity"] = time.time()


def apply_round_aggregation(submission_paths: List[str]):
    global global_weights, global_step

    if not submission_paths:
        log.warning("⚠️ No deltas to aggregate")
        return

    log.info(f"🧮 Aggregating {len(submission_paths)} deltas from disk (direct in-place)...")

    total_tokens = 0
    all_metadata = []

    for path in submission_paths:
        try:
            with open(path, 'rb') as f:
                meta_len = struct.unpack('<I', f.read(4))[0]
                metadata = json.loads(f.read(meta_len).decode('utf-8'))
            all_metadata.append((path, metadata))
            total_tokens += metadata.get("tokensProcessed", 0)
        except Exception as e:
            log.error(f"❌ Failed to read metadata from {path}: {e}")
            try:
                os.remove(path)
            except Exception:
                pass

    if total_tokens == 0 or not all_metadata:
        log.warning("⚠️ No valid deltas to aggregate (0 tokens)")
        for path in submission_paths:
            try:
                os.remove(path)
            except Exception:
                pass
        return

    log.info(f"📊 Total tokens across {len(all_metadata)} submissions: {total_tokens:,}")

    with shard_lock:
        global_weights.mul_(1.0 - FL_CONFIG["WEIGHT_DECAY"])

    engram_accumulator = {}

    for path, metadata in all_metadata:
        try:
            _, delta_bf16, eng_indices, eng_values_bf16 = load_delta_from_disk(path)
            weight = metadata.get("tokensProcessed", 0) / total_tokens
            scaled_lr = FL_CONFIG["SERVER_LR"] * weight

            if weight > 0:
                with shard_lock:
                    global_weights.add_(delta_bf16.float(), alpha=scaled_lr)

                if eng_indices is not None and len(eng_indices) > 0:
                    eng_values_fp32 = eng_values_bf16.float()
                    for idx, val in zip(eng_indices.tolist(), eng_values_fp32):
                        idx = int(idx)
                        if idx not in engram_accumulator:
                            engram_accumulator[idx] = [torch.zeros(MODEL_CONFIG["dim"], dtype=torch.float32), 0.0]
                        engram_accumulator[idx][0] += val * weight
                        engram_accumulator[idx][1] += weight

            del delta_bf16
            if eng_indices is not None:
                del eng_indices
            if eng_values_bf16 is not None:
                del eng_values_bf16
            gc.collect()

            try:
                os.remove(path)
            except Exception:
                pass

        except Exception as e:
            log.error(f"❌ Failed to process delta {path}: {e}")
            import traceback
            traceback.print_exc()
            try:
                os.remove(path)
            except Exception:
                pass

    with shard_lock:
        norm_sq = 0.0
        chunk_size = 10_000_000
        for i in range(0, MODEL_SIZE, chunk_size):
            chunk = global_weights[i:i + chunk_size].float()
            norm_sq += torch.sum(chunk * chunk).item()
            del chunk
        norm = math.sqrt(norm_sq)

        if norm > FL_CONFIG["MAX_WEIGHT_NORM"]:
            global_weights.mul_(FL_CONFIG["MAX_WEIGHT_NORM"] / norm)
            log.warning(f"⚠️ Global weight norm {norm:.2f} clipped to {FL_CONFIG['MAX_WEIGHT_NORM']}")

        if engram_accumulator:
            log.info(f"🧠 Updating {len(engram_accumulator)} engram buckets")
            for idx, (val_sum, weight_sum) in engram_accumulator.items():
                if weight_sum > 0:
                    avg_val = val_sum / weight_sum
                    start_p = idx * MODEL_CONFIG["dim"]
                    end_p = start_p + MODEL_CONFIG["dim"]
                    global_engram_weights[start_p:end_p] += avg_val * FL_CONFIG["SERVER_LR"]
            del engram_accumulator
            gc.collect()

        with stats_lock:
            global_step += 1

    avg_loss = sum(
        m.get("loss", 0) * m.get("tokensProcessed", 0)
        for _, m in all_metadata
        if m.get("loss")
    ) / max(total_tokens, 1)

    log.info(f"✅ Aggregated step {global_step} from {len(all_metadata)} submissions (tokens={total_tokens:,}, avg_loss={avg_loss:.4f})")

    handle_checkpoints()
    gc.collect()


# ============ AUTH ROUTES ============
@app.post("/auth/register")
async def register(request: Request):
    data = await request.json()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not username or not (3 <= len(username) <= 20):
        raise HTTPException(400, "Username must be 3-20 characters")
    if not username.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "Invalid characters")
    if contains_bad_word(username):
        raise HTTPException(400, "Prohibited words")
    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email")
    if not password or len(password) < 6:
        raise HTTPException(400, "Password too short")

    with users_lock:
        for u in users_db["users"].values():
            if u["username_lower"] == username.lower():
                raise HTTPException(400, "Username taken")
            if u.get("email", "").lower() == email.lower():
                raise HTTPException(400, "Email registered")

        user_uuid = str(uuid.uuid4())
        salt = secrets.token_hex(16)
        token = generate_token()

        users_db["users"][user_uuid] = {
            "uuid": user_uuid,
            "username": username,
            "username_lower": username.lower(),
            "email": email,
            "password_hash": hash_password(password, salt),
            "salt": salt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "contributions": 0,
                "totalTokens": 0,
                "totalSteps": 0,
            },
        }
        users_db["tokens"][token] = user_uuid
        save_users_db()
        log.info(f"✅ New user registered: {username} ({user_uuid[:8]}...)")

    return {"token": token, "uuid": user_uuid, "username": username}


@app.post("/auth/login")
async def auth_login(request: Request):
    data = await request.json()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    with users_lock:
        user = next((u for u in users_db["users"].values() if u["username_lower"] == username.lower()), None)
        if not user:
            raise HTTPException(401, "Invalid credentials")
        if hash_password(password, user["salt"]) != user["password_hash"]:
            raise HTTPException(401, "Invalid credentials")

        user_uuid = user["uuid"]
        token = generate_token()
        users_db["tokens"][token] = user_uuid
        save_users_db()
        log.info(f"✅ User logged in: {username}")

    return {"token": token, "uuid": user["uuid"], "username": user["username"]}


@app.get("/auth/me")
async def auth_me(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {
        "uuid": user["uuid"],
        "username": user["username"],
        "email": user.get("email"),
        "stats": user["stats"],
        "trustScore": compute_trust_score(user["stats"]),
        "globalRank": compute_global_rank(user["uuid"]),
    }


# ============ ADMIN ROUTES ============
@app.get("/api/checkpoints")
async def list_checkpoints(secret: str = ""):
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid secret")
    checkpoints = []
    for f in CHECKPOINT_DIR.iterdir():
        if f.name.startswith("model_step_") and f.suffix == ".bin":
            try:
                checkpoints.append({
                    "step": int(f.stem.split("_")[2]),
                    "size": f.stat().st_size,
                    "mtime": f.stat().st_mtime,
                })
            except Exception:
                continue
    return {"current_step": global_step, "checkpoints": sorted(checkpoints, key=lambda x: x["step"], reverse=True)}


@app.post("/api/admin/reload")
async def reload_checkpoint(step: int = -1, secret: str = ""):
    global global_weights, global_step
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid secret")

    if step == -1:
        load_global_state()
        load_model_weights()
        load_engram_weights()
    else:
        cp_file = CHECKPOINT_DIR / f"model_step_{step}.bin"
        if not cp_file.exists():
            raise HTTPException(404, "Not found")
        buf = np.fromfile(cp_file, dtype=np.float32)
        if len(buf) != MODEL_SIZE:
            raise HTTPException(400, "Size mismatch")
        global_weights = torch.from_numpy(buf.copy())

        cp_eng_file = CHECKPOINT_DIR / f"engram_step_{step}.bin"
        if cp_eng_file.exists():
            buf_eng = np.fromfile(cp_eng_file, dtype=np.float32)
            if len(buf_eng) == ENG_PARAMS:
                global_engram_weights = torch.from_numpy(buf_eng.copy())

        global_step = step

    save_latest_checkpoint()
    sync_huggingface_state()
    update_and_save_status(push_hf=True)
    return {"status": "reloaded", "step": global_step}


@app.post("/api/admin/push_hf")
async def force_hf_push(secret: str = ""):
    global last_hf_upload_time
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid secret")
    last_hf_upload_time = 0
    ok = upload_to_huggingface()
    update_and_save_status(push_hf=True)
    return {"status": "uploaded" if ok else "failed"}


@app.post("/api/admin/reset")
async def admin_reset(secret: str = ""):
    global global_weights, global_engram_weights, global_step, global_stats, lora_stats
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid secret")

    torch.manual_seed(42)
    global_weights = (torch.randn(MODEL_SIZE) * 0.02).float()
    global_engram_weights = (torch.randn(ENG_PARAMS) * 0.02).float()
    global_step = 0
    global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
    lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}

    save_latest_checkpoint()
    sync_huggingface_state()
    update_and_save_status(push_hf=True)
    return {"status": "reset"}


# ============ PUBLIC STATS ============
@app.get("/api/stats")
async def get_stats(request: Request):
    recent = global_stats["lossHistory"][-10:]
    avg_loss = sum(recent) / len(recent) if recent else 0.0
    loss_history = global_stats["lossHistory"]
    training_loss_change = (
        round(((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100, 1)
        if len(loss_history) >= 2 and loss_history[0] > 0
        else 0.0
    )

    with users_lock:
        total_users = len(users_db["users"])

    active_gpus = _active_client_count()
    auth_user = get_current_user(request)

    your_contribution = 0.0
    if auth_user:
        total_tokens = global_stats["totalTokens"]
        if total_tokens > 0:
            your_contribution = round((auth_user["stats"].get("totalTokens", 0) / total_tokens) * 100, 4)

    return {
        "totalParticipants": total_users,
        "totalContributions": lora_stats.get("total_submissions", 0),
        "activeGpus": active_gpus,
        "totalTokens": global_stats["totalTokens"],
        "trainingLossChange": training_loss_change,
        "yourContribution": your_contribution,
        "activeNow": active_gpus,
        "totalSteps": global_stats["totalSteps"],
        "globalStep": global_step,
        "modelSize": MODEL_SIZE,
        "lastLoss": global_stats["lastLoss"],
        "avgLoss": avg_loss,
        "weightNorm": float(torch.norm(global_weights.float()).item()) if global_weights is not None else 0.0,
        "loraStats": lora_stats,
        "totalUsers": total_users,
        "activeContributors": active_gpus,
        "lossHistory": global_stats["lossHistory"][-20:],
        "datasetStats": {
            "totalChunks": dataset_info["totalChunks"],
            "currentChunk": data_position["currentChunkIdx"],
            "currentEpoch": data_position["epoch"],
            "totalShardsProcessed": dataset_stats["totalShardsProcessed"],
            "epochsCompleted": dataset_stats["epochsCompleted"],
        },
        "currentRound": round_state["current_round"],
        "huggingface": {
            "repoId": HF_REPO_ID,
            "enabled": HF_ENABLED,
            "uploadIntervalHours": HF_UPLOAD_INTERVAL_HOURS,
            "revision": current_hf_revision,
            "revisionStep": hf_revision_global_step,
            "directDownloads": bool(_build_hf_weights_metadata("bf16")),
        } if HF_ENABLED else {},
    }


@app.get("/api/leaderboard")
async def get_leaderboard():
    lb_file = BASE_DIR / "leaderboard.json"
    if lb_file.exists():
        try:
            with open(lb_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    with users_lock:
        users_list = [
            {
                "uuid": u["uuid"],
                "username": u["username"],
                "contributions": u["stats"].get("contributions", 0),
                "totalTokens": u["stats"].get("totalTokens", 0),
                "totalSteps": u["stats"].get("totalSteps", 0),
                "trustScore": compute_trust_score(u["stats"]),
            }
            for u in users_db["users"].values()
        ]

    return sorted(users_list, key=lambda x: x["totalTokens"], reverse=True)


@app.get("/api/checkpoint")
async def get_checkpoint(request: Request, format: str = "bf16"):
    if not LATEST_CHECKPOINT_FILE.exists():
        raise HTTPException(status_code=404, detail="No checkpoint")

    w = np.fromfile(LATEST_CHECKPOINT_FILE, dtype=np.float32)[:MODEL_SIZE]
    if format == "fp8":
        c = compress_weights_fp8(w)
    elif format == "fp16":
        c = compress_weights_fp16(w)
    else:
        c = compress_weights_bf16(w)

    return Response(
        content=c,
        media_type="application/octet-stream",
        headers={
            "X-Global-Step": str(global_step),
            "X-Model-Size": str(MODEL_SIZE),
        },
    )


@app.get("/api/engram_checkpoint")
async def get_engram_checkpoint(request: Request, format: str = "bf16"):
    if not ENGRAM_CHECKPOINT_FILE.exists():
        raise HTTPException(status_code=404, detail="No engram checkpoint")

    w = np.fromfile(ENGRAM_CHECKPOINT_FILE, dtype=np.float32)[:ENG_PARAMS]
    if format == "fp8":
        c = compress_weights_fp8(w)
    elif format == "fp16":
        c = compress_weights_fp16(w)
    else:
        c = compress_weights_bf16(w)

    return Response(
        content=c,
        media_type="application/octet-stream",
        headers={
            "X-Global-Step": str(global_step),
            "X-Engram-Size": str(ENG_PARAMS),
        },
    )


@app.get("/api/model_info")
async def get_model_info():
    return {
        "architecture": MODEL_CONFIG["architecture"],
        "parameters": MODEL_SIZE,
        "engramParams": ENG_PARAMS,
        "config": MODEL_CONFIG,
        "globalStep": global_step,
        "currentRound": round_state["current_round"],
        "huggingface": {
            "repoId": HF_REPO_ID,
            "enabled": HF_ENABLED,
            "revision": current_hf_revision,
            "revisionStep": hf_revision_global_step,
            "directDownloads": bool(_build_hf_weights_metadata("bf16")),
        } if HF_ENABLED else {},
    }


@app.post("/api/reset_weights")
async def reset_weights(secret: str = ""):
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid secret")

    global global_weights, global_engram_weights, global_step, global_stats, lora_stats

    torch.manual_seed(42)
    global_weights = (torch.randn(MODEL_SIZE) * 0.02).float()
    global_engram_weights = (torch.randn(ENG_PARAMS) * 0.02).float()
    global_step = 0
    global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
    lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}

    save_latest_checkpoint()
    sync_huggingface_state()
    update_and_save_status(push_hf=True)
    return {"status": "reset"}


# ============ STARTUP ============
@app.on_event("startup")
async def startup():
    load_dataset_state()
    load_users_db()
    load_global_state()
    initialize_model()
    load_model_weights()
    load_engram_weights()
    initialize_dataset()

    round_state["round_start_time"] = time.time()
    round_state["current_round"] = global_step + 1

    try:
        for f in DELTA_TEMP_DIR.glob("*.bin"):
            f.unlink()
        log.info("🧹 Cleared old pending delta files.")
    except Exception:
        pass

    try:
        for p in UPLOAD_DIR.iterdir():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        log.info("🧹 Cleared old pending upload directories.")
    except Exception:
        pass

    sync_huggingface_state()
    update_and_save_status(push_hf=True)

    log.info("=" * 60)
    log.info(f"🚀 CrowdGPT Continuous FL Coordinator on port {PORT}")
    log.info(f"⚡ Mode: UPLOAD-SAFE CONTINUOUS AGGREGATION (Rolling Window)")
    log.info(f"⏱️  Server round: {round_state['round_duration_hours']}h | Client target: {ROUND_CONFIG['TARGET_SUBMISSION_HOURS']}h rolling")
    log.info(f"🧠 RAM Optimizations: Zero-Copy Streaming + Disk-Based Aggregation + Chunked Math")
    log.info(f"🧮 Master weights: FP32 in-memory accumulation (bf16 only at wire/export boundaries)")
    log.info(f"🛡️  Admin watchdog: active (secret={ADMIN_SECRET[:8]}...)")
    log.info(f"👥 Client identity: user UUID (primary) → IP (fallback)")
    log.info(f"🤗 HF direct downloads: {'enabled' if _build_hf_weights_metadata('bf16') else 'disabled/fallback'}")
    log.info(f"📦 Chunked uploads: max chunk={MAX_CHUNK_BYTES/(1024*1024):.0f} MB, max total={MAX_TOTAL_UPLOAD_BYTES/(1024*1024):.0f} MB")
    log.info(f"🔄 Initialized round {round_state['current_round']}")
    log.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", access_log=False)
