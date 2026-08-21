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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

# 🔧 Windows encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ============ LOGGING ============
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ============ CONFIGURATION ============
PORT = int(os.environ.get("PORT", 3000))
BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"
LATEST_CHECKPOINT_FILE = CHECKPOINT_DIR / "model_latest.bin"
GLOBAL_STATE_FILE = CHECKPOINT_DIR / "global_state.json"
DATASET_CACHE_FILE = DATA_DIR / "dataset_info.json"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============ MODEL ============
MODEL_CONFIG = {
    "vocabSize": 50257, "dim": 768, "nLayers": 8, "nHeads": 12,
    "nKvHeads": 4, "headDim": 64, "maxSeqLen": 64, "architecture": "SotaGPT",
    "features": {"groupedQueryAttention": True, "rotaryPositionEmbedding": True,
                 "swiGLU": True, "weightTying": True, "chunkedAttention": True}
}

LORA_CONFIG = {
    "FULL_RANK": MODEL_CONFIG["dim"],
    "RANK_TIERS": {"minimal": 32, "low": 64, "medium": 128, "high": 256,
                   "ultra": 512, "extreme": 1024, "max": 2048, "full": MODEL_CONFIG["dim"]},
    "DEFAULT_RANK": MODEL_CONFIG["dim"],
    "WEIGHT_BY_RANK": True,
    "LOG_EFFECTIVE_RANK": True
}

DATASET_CONFIG = {
    "url": "https://huggingface.co/datasets/Vxtzq/TinyStoriesBIN/resolve/main/tinystories_tokens.bin",
    "chunkSizeMB": 1,
    "tokensPerSample": MODEL_CONFIG["maxSeqLen"] + 1,
    "tokenBytes": 2,
}

# 🎯 Training modes — the server enforces ONE mode per shard (same settings for the whole group)
MODE_CONFIGS = {
    "quick":    {"localSteps": 500,    "batchSize": 4, "seqLen": 64},
    "balanced": {"localSteps": 5000,   "batchSize": 4, "seqLen": 64},
    "deep":     {"localSteps": 50000,  "batchSize": 4, "seqLen": 64},
    "ultra":    {"localSteps": 100000, "batchSize": 4, "seqLen": 64},
}

def calculate_model_size(cfg: dict) -> int:
    size = cfg["vocabSize"] * cfg["dim"]
    for _ in range(cfg["nLayers"]):
        d = cfg["dim"]
        size += d*2 + d*(cfg["nHeads"]*cfg["headDim"]) + d*(cfg["nKvHeads"]*cfg["headDim"])*2
        size += d*d + d*2
        size += d*4*d + d*4*d + 4*d*d
    size += cfg["dim"]*2
    return size

MODEL_SIZE = calculate_model_size(MODEL_CONFIG)
log.info(f"🧮 SotaGPT size: {MODEL_SIZE:,} params (~{MODEL_SIZE*4/1024/1024:.2f} MB)")

# ============ QUANTIZATION ============
def float32_to_bf16_array(a): return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16).view(torch.uint16).numpy()
def bf16_array_to_float32(a): return torch.from_numpy(a.copy()).view(torch.uint16).view(torch.bfloat16).to(torch.float32).numpy()
def float32_to_fp16_array(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return np.clip(a, -65504.0, 65504.0).astype(np.float16).view(np.uint16)
def fp16_array_to_float32(a): return a.view(np.float16).astype(np.float32)
def float32_to_fp8_array(a): return torch.from_numpy(a.astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
def fp8_array_to_float32(a): return torch.from_numpy(a.copy()).view(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).numpy()

def compress_weights_bf16(w): return float32_to_bf16_array(w).tobytes()
def compress_weights_fp16(w): return float32_to_fp16_array(w).tobytes()
def compress_weights_fp8(w): return float32_to_fp8_array(w).tobytes()

# ============ GLOBAL STATE (ANONYMOUS SWARM) ============
db = {
    "tasks": {},              # taskId -> {shardId, chunkIdx, mode, requestedByIp}
    "pendingSubmissions": {}, # shardId -> [ {nodeIp, delta, rank, ...} ]
    "shardMeta": {},          # shardId -> {mode, createdAt, required, chunkIdx}
}

active_ips: Dict[str, float] = {}      # ip -> last_seen (seconds)
blacklisted_subnets: Dict[str, float] = {}  # subnet -> expiry timestamp

global_weights: Optional[torch.Tensor] = None
global_step = 0
global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}

dataset_info = {"url": DATASET_CONFIG["url"], "totalSize": 0, "totalChunks": 0,
                "chunkSizeBytes": DATASET_CONFIG["chunkSizeMB"]*1024*1024,
                "initialized": False, "lastChecked": None}
data_position = {"currentChunkIdx": 0, "epoch": 0, "totalSamplesSeen": 0}
current_data_shard = {"chunkIdx": 0, "assignedIPs": [], "shardId": None, "createdAt": None}
dataset_stats = {"totalShardsProcessed": 0, "epochsCompleted": 0, "samplesProcessed": 0, "lastUpdate": None}

FL_CONFIG = {
    "CONSENSUS_REQUIRED": 3,      # distinct IPs needed to fill a shard / aggregate
    "CHECKPOINT_INTERVAL": 100,
    "MAX_CHECKPOINTS": 10,
    "DELTA_NORM_BASE": 500.0,
    "DELTA_NORM_ABSOLUTE_MAX": 5000.0,
    "MAX_WEIGHT_NORM": 1000.0,
    "WEIGHT_DECAY": 0.0001,
    "SERVER_LR": 0.1,             # 🎯 NEW: Scales sequential updates to prevent divergence
    "IP_TIMEOUT_S": 600,          # consider an IP "active" if seen in last 10 min
    "SHARD_TIMEOUT_S": 1800,      # force-aggregate a shard after 30 min (avoid stuck shards)
    "CLEANUP_INTERVAL_S": 60,
}

shard_lock = threading.Lock()  # guards shard/submission mutations

# ============ SUBNET HELPERS ============
def get_subnet(ip: str) -> str:
    if not ip or ip == "unknown":
        return "unknown"
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3]) + ".0/24"
    return ip  # IPv6 or weird -> use as-is

def is_blacklisted(ip: str) -> bool:
    subnet = get_subnet(ip)
    if subnet in blacklisted_subnets:
        if time.time() < blacklisted_subnets[subnet]:
            return True
        del blacklisted_subnets[subnet]
    return False

def ban_subnet(ip: str, reason: str, hours: int = 24):
    subnet = get_subnet(ip)
    blacklisted_subnets[subnet] = time.time() + hours*3600
    log.warning(f"🔨 BANNED {subnet} for {hours}h — {reason}")

def get_client_ip(request: Request) -> str:
    # honor X-Forwarded-For if behind a proxy, else direct client
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

# ============ DATASET STATE ============
def save_dataset_state():
    try:
        with open(DATASET_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({"datasetInfo": dataset_info, "dataPosition": data_position,
                       "datasetStats": dataset_stats}, f, indent=2)
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
    try:
        res = requests.head(dataset_info["url"], allow_redirects=True, timeout=30)
        res.raise_for_status()
        total_size = int(res.headers.get('content-length', 0))
        chunk_size = dataset_info["chunkSizeBytes"]
        dataset_info["totalSize"] = total_size
        dataset_info["totalChunks"] = (total_size + chunk_size - 1) // chunk_size
        dataset_info["initialized"] = True
        dataset_info["lastChecked"] = datetime.now(timezone.utc).isoformat()
        log.info(f"✅ dataset: {total_size/1024/1024:.1f} MB, {dataset_info['totalChunks']} chunks")
        save_dataset_state()
    except Exception as e:
        log.error(f"❌ initialize_dataset: {e}")

# ============ SHARD MANAGEMENT (the core "same chunk" logic) ============
def _active_ip_count() -> int:
    now = time.time()
    return sum(1 for ts in active_ips.values() if now - ts < FL_CONFIG["IP_TIMEOUT_S"])

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

def get_or_advance_shard(client_ip: str, requested_mode: str = "quick") -> dict:
    """
    Assigns client_ip to the current shard.
    - Solo mode (< CONSENSUS_REQUIRED active IPs): advance chunk on every request.
    - Consensus mode: a shard fills to CONSENSUS_REQUIRED distinct IPs, then advances.
    The FIRST node to a brand-new shard sets its training mode (enforces same settings).
    """
    global current_data_shard
    if not dataset_info["initialized"]:
        initialize_dataset()

    with shard_lock:
        solo = _active_ip_count() < FL_CONFIG["CONSENSUS_REQUIRED"]
        is_new_client = (current_data_shard["shardId"] is None) or (client_ip not in current_data_shard["assignedIPs"])

        need_new_shard = False
        if current_data_shard["shardId"] is None:
            need_new_shard = True
        elif solo:
            need_new_shard = True
        elif is_new_client and len(current_data_shard["assignedIPs"]) >= FL_CONFIG["CONSENSUS_REQUIRED"]:
            need_new_shard = True

        if need_new_shard:
            if current_data_shard["shardId"] is not None:
                _advance_chunk_pointer()
            mode = requested_mode if requested_mode in MODE_CONFIGS else "quick"
            shard_id = f"shard_{data_position['epoch']}_{data_position['currentChunkIdx']}_{int(time.time()*1000)}"
            current_data_shard = {
                "chunkIdx": data_position["currentChunkIdx"],
                "assignedIPs": [],
                "shardId": shard_id,
                "createdAt": int(time.time()*1000),
            }
            db["shardMeta"][shard_id] = {
                "mode": mode,                       # first node locks the mode for this shard
                "createdAt": int(time.time()*1000),
                "required": 1 if solo else FL_CONFIG["CONSENSUS_REQUIRED"],
                "chunkIdx": data_position["currentChunkIdx"],
            }
            db["pendingSubmissions"][shard_id] = []
            log.info(f"📦 New shard {shard_id} chunk={current_data_shard['chunkIdx']} "
                     f"mode={mode} ({'solo' if solo else 'consensus'})")

        if client_ip not in current_data_shard["assignedIPs"]:
            current_data_shard["assignedIPs"].append(client_ip)

        save_dataset_state()
        return current_data_shard

# ============ MODEL WEIGHTS ============
def initialize_model():
    global global_weights
    torch.manual_seed(42)
    global_weights = torch.randn(MODEL_SIZE) * 0.02
    log.info(f"🎲 initialized {MODEL_SIZE:,} random weights")

def load_latest_checkpoint() -> bool:
    global global_weights, global_step
    try:
        if LATEST_CHECKPOINT_FILE.exists() and GLOBAL_STATE_FILE.exists():
            with open(GLOBAL_STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            global_step = state.get("globalStep", 0)
            buf = np.fromfile(LATEST_CHECKPOINT_FILE, dtype=np.float32)[:MODEL_SIZE]
            if len(buf) != MODEL_SIZE:
                return False
            global_weights = torch.from_numpy(buf.copy())
            log.info(f"✅ loaded checkpoint step {global_step}")
            return True
    except Exception as e:
        log.error(f"❌ load checkpoint: {e}")
    return False

def save_latest_checkpoint():
    try:
        global_weights.cpu().numpy().tofile(LATEST_CHECKPOINT_FILE)
        with open(GLOBAL_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump({"globalStep": global_step, "savedAt": datetime.now(timezone.utc).isoformat(),
                       "modelSize": MODEL_SIZE, "modelConfig": MODEL_CONFIG, "loraStats": lora_stats}, f, indent=2)
    except Exception as e:
        log.error(f"❌ save checkpoint: {e}")

# ============ VECTOR MATH / SEQUENTIAL AGGREGATION ============
def compute_max_allowed_norm(local_steps, rank=None):
    steps = max(1, int(local_steps or 1))
    base = FL_CONFIG["DELTA_NORM_BASE"] * math.sqrt(steps / 100.0)
    rank_factor = np.sqrt(rank / LORA_CONFIG["FULL_RANK"]) if rank and rank < LORA_CONFIG["FULL_RANK"] else 1.0
    return base * rank_factor

def apply_sequential_update(delta: np.ndarray, client_ip: str, rank: int):
    """🎯 NEW: Applies delta immediately to global weights (sequential FedSGD style)"""
    global global_weights, global_step
    
    if np.isnan(delta).any() or np.isinf(delta).any():
        log.error("🚨 Sequential update rejected: NaN/Inf in delta")
        return

    delta_tensor = torch.from_numpy(delta)
    
    with shard_lock:
        # 1. Weight decay
        global_weights.mul_(1.0 - FL_CONFIG["WEIGHT_DECAY"])
        
        # 2. Apply scaled sequential update
        global_weights.add_(delta_tensor, alpha=FL_CONFIG["SERVER_LR"])
        
        # 3. Global norm clip to prevent explosion
        norm = torch.norm(global_weights).item()
        if norm > FL_CONFIG["MAX_WEIGHT_NORM"]:
            global_weights.mul_(FL_CONFIG["MAX_WEIGHT_NORM"] / norm)
            log.warning(f"⚠️ global weight norm {norm:.2f} clipped")
            
        global_step += 1
        
    log.info(f"✅ Sequential update step {global_step} from {client_ip} (rank={rank})")
    save_latest_checkpoint()
    handle_checkpoints()

def handle_checkpoints():
    if global_step % FL_CONFIG["CHECKPOINT_INTERVAL"] != 0:
        return
    try:
        fp = CHECKPOINT_DIR / f"model_step_{global_step}.bin"
        global_weights.cpu().numpy().tofile(fp)
        with open(CHECKPOINT_DIR / f"model_step_{global_step}.json", 'w', encoding='utf-8') as f:
            json.dump({"globalStep": global_step, "savedAt": datetime.now(timezone.utc).isoformat(),
                       "modelSize": MODEL_SIZE}, f, indent=2)
        files = sorted([f for f in CHECKPOINT_DIR.iterdir()
                        if f.name.startswith("model_step_") and f.suffix == ".bin"],
                       key=lambda f: int(f.stem.split("_")[2]))
        while len(files) > FL_CONFIG["MAX_CHECKPOINTS"]:
            old = files.pop(0); old.unlink()
            meta = CHECKPOINT_DIR / f"{old.stem}.json"
            if meta.exists(): meta.unlink()
    except Exception as e:
        log.error(f"❌ checkpoint: {e}")

# ============ BACKGROUND THREADS ============
def cleanup_loop():
    while True:
        time.sleep(FL_CONFIG["CLEANUP_INTERVAL_S"])
        now = time.time()
        for ip in list(active_ips.keys()):
            if now - active_ips[ip] > FL_CONFIG["IP_TIMEOUT_S"]:
                del active_ips[ip]
        for sn in list(blacklisted_subnets.keys()):
            if now > blacklisted_subnets[sn]:
                del blacklisted_subnets[sn]

def aggregation_watchdog():
    """Force-cleans shards that timed out (keeps stats/watchdog logic intact)"""
    while True:
        time.sleep(FL_CONFIG["CLEANUP_INTERVAL_S"])
        now_ms = int(time.time()*1000)
        with shard_lock:
            for shard_id in list(db["pendingSubmissions"].keys()):
                subs = db["pendingSubmissions"].get(shard_id, [])
                if not subs:
                    continue
                meta = db["shardMeta"].get(shard_id, {})
                age_s = (now_ms - meta.get("createdAt", now_ms)) / 1000
                # In sequential mode, we just clean up old shards to prevent memory leaks
                if age_s > FL_CONFIG["SHARD_TIMEOUT_S"]:
                    log.info(f"🧹 Watchdog cleaning stale shard {shard_id} (age={age_s}s)")
                    db["pendingSubmissions"][shard_id] = []

threading.Thread(target=cleanup_loop, daemon=True).start()
threading.Thread(target=aggregation_watchdog, daemon=True).start()

# ============ FASTAPI ============
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="CrowdGPT Swarm Coordinator")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Task-Id", "X-Shard-Id", "X-Chunk-Idx", "X-Global-Step",
                                   "X-Model-Size", "X-Quantized"])

# ============ FL ROUTES (PERMISSIONLESS) ============
@app.get("/fl/task")
async def get_task(request: Request, mode: str = "quick", format: str = "bf16"):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")
    active_ips[client_ip] = time.time()

    shard = get_or_advance_shard(client_ip, requested_mode=mode)
    shard_id = shard["shardId"]
    meta = db["shardMeta"].get(shard_id, {})
    shard_mode = meta.get("mode", mode if mode in MODE_CONFIGS else "quick")  # enforce same settings
    cfg = MODE_CONFIGS[shard_mode]

    task_id = f"{shard_id}_{secrets.token_hex(4)}"
    db["tasks"][task_id] = {
        "shardId": shard_id,
        "chunkIdx": shard["chunkIdx"],
        "mode": shard_mode,
        "requestedByIp": client_ip,
        "createdAt": int(time.time()*1000),
    }

    chunk_offset = shard["chunkIdx"] * dataset_info["chunkSizeBytes"]
    metadata = {
        "taskId": task_id,
        "globalStep": global_step,
        "modelSize": MODEL_SIZE,
        "modelConfig": MODEL_CONFIG,
        "trainingConfig": cfg,                      # ⬅ same settings for the whole shard
        "weightFormat": format,
        "datasetConfig": {                          # ⬅ client uses these to fetch its chunk
            "url": dataset_info["url"],
            "chunkIdx": shard["chunkIdx"],
            "chunkOffset": chunk_offset,
            "chunkSize": dataset_info["chunkSizeBytes"],
            "totalChunks": dataset_info["totalChunks"],
            "totalSize": dataset_info["totalSize"],
            "epoch": data_position["epoch"],
            "tokensPerSample": DATASET_CONFIG["tokensPerSample"],
            "tokenBytes": DATASET_CONFIG["tokenBytes"],
        },
        "shardConfig": {
            "shardId": shard_id,
            "slot": len(shard["assignedIPs"]),
            "total": meta.get("required", FL_CONFIG["CONSENSUS_REQUIRED"]),
        },
        "loraConfig": {
            "defaultRank": LORA_CONFIG["DEFAULT_RANK"],
            "fullRank": LORA_CONFIG["FULL_RANK"],
            "rankTiers": LORA_CONFIG["RANK_TIERS"],
            "weightByRank": LORA_CONFIG["WEIGHT_BY_RANK"],
        },
    }

    metadata_json = json.dumps(metadata).encode('utf-8')
    metadata_length = struct.pack('<I', len(metadata_json))

    weights_np = global_weights.cpu().numpy()
    if format == "fp8":
        compressed = compress_weights_fp8(weights_np)
    elif format == "fp16":
        compressed = compress_weights_fp16(weights_np)
    else:
        compressed = compress_weights_bf16(weights_np)

    log.info(f"📤 task {task_id} shard={shard_id} chunk={shard['chunkIdx']} mode={shard_mode} ip={client_ip}")
    return Response(content=metadata_length + metadata_json + compressed,
                    media_type="application/octet-stream",
                    headers={"X-Task-Id": task_id, "X-Shard-Id": shard_id,
                             "X-Chunk-Idx": str(shard["chunkIdx"]), "X-Global-Step": str(global_step),
                             "X-Model-Size": str(MODEL_SIZE), "X-Quantized": format})

@app.post("/fl/submit")
async def submit_delta(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip):
        raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")
    active_ips[client_ip] = time.time()

    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty payload")

    # decompress + parse
    try:
        buf = gzip.decompress(raw) if (len(raw) >= 2 and raw[0] == 0x1f and raw[1] == 0x8b) else raw
        payload_len = struct.unpack('<I', buf[:4])[0]
        body = json.loads(buf[4:4+payload_len].decode('utf-8'))
        delta_bytes = buf[4+payload_len:]
        fmt = body.get("weightFormat", "bf16")
        if fmt == "fp8":
            delta = fp8_array_to_float32(np.frombuffer(delta_bytes, dtype=np.uint8))
        elif fmt == "fp16":
            delta = fp16_array_to_float32(np.frombuffer(delta_bytes, dtype=np.uint16))
        elif fmt == "fp32":
            delta = np.frombuffer(delta_bytes, dtype=np.float32)
        else:
            delta = bf16_array_to_float32(np.frombuffer(delta_bytes, dtype=np.uint16))
    except Exception as e:
        log.error(f"❌ parse failed: {e}")
        raise HTTPException(status_code=400, detail="Binary parsing failed")

    task_id = body.get("taskId")
    if task_id not in db["tasks"]:
        raise HTTPException(status_code=404, detail="Task expired")
    task = db["tasks"][task_id]
    shard_id = task["shardId"]

    if len(delta) != MODEL_SIZE:
        raise HTTPException(status_code=400, detail=f"Size mismatch: expected {MODEL_SIZE}, got {len(delta)}")
    if np.isnan(delta).any() or np.isinf(delta).any():
        ban_subnet(client_ip, "submitted NaN/Inf delta")
        raise HTTPException(status_code=400, detail="Invalid delta (NaN/Inf)")

    rank = body.get("loraRank", LORA_CONFIG["FULL_RANK"])
    local_steps = body.get("localSteps", 1)
    loss = body.get("loss")

    # norm gate
    delta_norm = float(np.linalg.norm(delta))
    max_allowed = compute_max_allowed_norm(local_steps, rank)
    abs_max = max(FL_CONFIG["DELTA_NORM_ABSOLUTE_MAX"], max_allowed * 3)
    if delta_norm > abs_max:
        ban_subnet(client_ip, f"delta norm {delta_norm:.0f} > max {abs_max:.0f}")
        raise HTTPException(status_code=400, detail=f"Delta too large ({delta_norm:.2f})")
    
    normalized = False
    if delta_norm > max_allowed:
        delta = delta * (max_allowed / delta_norm)
        normalized = True

    # 🎯 SEQUENTIAL AGGREGATION: Apply immediately
    apply_sequential_update(delta, client_ip, rank)

    # Still track in shard for stats/watchdog compatibility
    with shard_lock:
        subs = db["pendingSubmissions"].setdefault(shard_id, [])
        if any(s["nodeIp"] == client_ip for s in subs):
            # Don't reject, just don't double-count stats if they retry
            pass
        else:
            subs.append({
                "nodeIp": client_ip,
                "delta_norm": delta_norm,
                "rank": rank,
                "loss": loss,
                "localSteps": local_steps,
                "tokensProcessed": body.get("tokensProcessed", 0),
                "normalized": normalized,
                "submittedAt": int(time.time()*1000),
            })

        global_stats["totalTokens"] += body.get("tokensProcessed", 0)
        global_stats["totalSteps"] += local_steps or 0
        if loss and loss > 0 and math.isfinite(loss):
            global_stats["lastLoss"] = loss
            global_stats["lossHistory"].append(loss)
            global_stats["lossHistory"] = global_stats["lossHistory"][-100:]

        lora_stats["total_submissions"] += 1
        lora_stats["rank_distribution"][str(rank)] = lora_stats["rank_distribution"].get(str(rank), 0) + 1
        lora_stats["last_update"] = datetime.now(timezone.utc).isoformat()

        required = db["shardMeta"].get(shard_id, {}).get("required", FL_CONFIG["CONSENSUS_REQUIRED"])
        log.info(f"📥 {shard_id}: tracking submission {len(subs)}/{required} (ip={client_ip}, norm={delta_norm:.2f})")

    return {"status": "success", "globalStep": global_step, "normalized": normalized,
            "shardId": shard_id, "chunkIdx": task["chunkIdx"], "loss": loss}

# ============ PUBLIC STATS ============
@app.get("/api/stats")
async def get_stats(request: Request):
    recent = global_stats["lossHistory"][-10:]
    avg_loss = sum(recent)/len(recent) if recent else 0.0
    return {
        "activeNow": _active_ip_count(),
        "totalTokens": global_stats["totalTokens"],
        "totalSteps": global_stats["totalSteps"],
        "globalStep": global_step,
        "modelSize": MODEL_SIZE,
        "lastLoss": global_stats["lastLoss"],
        "avgLoss": avg_loss,
        "weightNorm": float(torch.norm(global_weights).item()) if global_weights is not None else 0.0,
        "loraStats": lora_stats,
        "datasetStats": {
            "totalChunks": dataset_info["totalChunks"],
            "currentChunk": data_position["currentChunkIdx"],
            "currentEpoch": data_position["epoch"],
            "totalShardsProcessed": dataset_stats["totalShardsProcessed"],
            "epochsCompleted": dataset_stats["epochsCompleted"],
        },
        "currentShard": {
            "shardId": current_data_shard["shardId"],
            "chunkIdx": current_data_shard["chunkIdx"],
            "assignedIPs": len(current_data_shard["assignedIPs"]),
            "required": db["shardMeta"].get(current_data_shard["shardId"], {}).get("required", FL_CONFIG["CONSENSUS_REQUIRED"]),
        },
    }

@app.get("/api/checkpoint")
async def get_checkpoint(request: Request, format: str = "bf16"):
    if not LATEST_CHECKPOINT_FILE.exists():
        raise HTTPException(status_code=404, detail="No checkpoint yet")
    w = np.fromfile(LATEST_CHECKPOINT_FILE, dtype=np.float32)[:MODEL_SIZE]
    if format == "fp8":
        c = compress_weights_fp8(w)
    elif format == "fp16":
        c = compress_weights_fp16(w)
    else:
        c = compress_weights_bf16(w)
    return Response(content=c, media_type="application/octet-stream",
                    headers={"X-Global-Step": str(global_step), "X-Model-Size": str(MODEL_SIZE)})

# Fallbacks so old web UI doesn't hard-crash
@app.get("/api/leaderboard")
async def get_leaderboard():
    return []

@app.get("/api/model_info")
async def get_model_info():
    return {"architecture": MODEL_CONFIG["architecture"], "parameters": MODEL_SIZE,
            "config": MODEL_CONFIG, "globalStep": global_step}

@app.post("/api/reset_weights")
async def reset_weights():
    global global_weights, global_step
    torch.manual_seed(42)
    global_weights = torch.randn(MODEL_SIZE) * 0.02
    global_step = 0
    save_latest_checkpoint()
    return {"status": "reset"}

# ============ STARTUP ============
@app.on_event("startup")
async def startup():
    load_dataset_state()
    initialize_model()
    if not load_latest_checkpoint():
        save_latest_checkpoint()
    initialize_dataset()
    log.info("="*60)
    log.info(f"🚀 CrowdGPT Swarm Coordinator on port {PORT}")
    log.info(f"⚡ Mode: SEQUENTIAL AGGREGATION (SERVER_LR={FL_CONFIG['SERVER_LR']})")
    log.info(f"🧩 consensus_tracking={FL_CONFIG['CONSENSUS_REQUIRED']} shard_timeout={FL_CONFIG['SHARD_TIMEOUT_S']}s")
    log.info("="*60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", access_log=False)
