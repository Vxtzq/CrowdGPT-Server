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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

# ============ LOGGING (must be first) ============
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ============ IMPORTS ============
import numpy as np
import torch
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

try:
    from huggingface_hub import HfApi, login, hf_hub_download
    from safetensors.torch import save_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    log.warning("⚠️ huggingface_hub not installed - HF uploads disabled")

# ============ CONFIGURATION ============
PORT = 5006
BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"
LATEST_CHECKPOINT_FILE = CHECKPOINT_DIR / "model_latest.bin"
GLOBAL_STATE_FILE = CHECKPOINT_DIR / "global_state.json"
DATASET_CACHE_FILE = DATA_DIR / "dataset_info.json"
USERS_DB_FILE = BASE_DIR / "users.json"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

import socket
hostname = socket.gethostname()
try:
    local_ip = socket.gethostbyname(hostname)
except:
    local_ip = "unknown"
print(f"🌐 Container hostname: {hostname}")
print(f"🌐 Container local IP: {local_ip}")
print(f"🌐 Listening on: 0.0.0.0:{PORT}")
print("="*60)

# HuggingFace
HF_TOKEN = "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"
HF_REPO_ID = "Vxtzq/Crowd-v1"
DATASET_REPO_ID = "Vxtzq/CrowdGPT"
HF_UPLOAD_INTERVAL_HOURS = 5
HF_ENABLED = HF_AVAILABLE and bool(HF_TOKEN)

# Admin secret for watchdog endpoints
ADMIN_SECRET = "crowdgpt-admin-9f3k2m8x"

# ============ MODEL ============
MODEL_CONFIG = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
    "architecture": "SotaGPT",
    "features": {"groupedQueryAttention": True, "rotaryPositionEmbedding": True,
                 "swiGLU": True, "weightTying": True, "chunkedAttention": True}
}

LORA_CONFIG = {
    "FULL_RANK": MODEL_CONFIG["dim"],
    "RANK_TIERS": {"minimal": 32, "low": 64, "medium": 128, "high": 256,
                   "ultra": 512, "extreme": 1024, "max": 2048, "full": MODEL_CONFIG["dim"]},
    "DEFAULT_RANK": MODEL_CONFIG["dim"], "WEIGHT_BY_RANK": True, "LOG_EFFECTIVE_RANK": True
}

# IMPORTANT: Update totalChunks to match the actual number of chunk_*.bin files!
DATASET_CONFIG = {
    "repoId": DATASET_REPO_ID,
    "totalChunks": 100,  # <-- UPDATE THIS VALUE
    "tokensPerSample": MODEL_CONFIG["maxSeqLen"] + 1, 
    "tokenBytes": 4,
}

MODE_CONFIGS = {
    "quick":    {"localSteps": 50000,   "batchSize": 8, "seqLen": 64},
    "balanced": {"localSteps": 110000,  "batchSize": 8, "seqLen": 64},
    "deep":     {"localSteps": 1000000, "batchSize": 8, "seqLen": 64},
    "ultra":    {"localSteps": 2000000, "batchSize": 8, "seqLen": 64},
}

def calculate_model_size(cfg: dict) -> int:
    size = cfg["vocabSize"] * cfg["dim"]
    for _ in range(cfg["nLayers"]):
        d = cfg["dim"]
        size += d * 2
        size += d * (cfg["nHeads"] * cfg["headDim"])
        size += d * (cfg["nKvHeads"] * cfg["headDim"])
        size += d * (cfg["nKvHeads"] * cfg["headDim"])
        size += d * d
        size += d * 2
        mlp_hidden = cfg.get("mlpHidden", 4 * d)
        size += d * mlp_hidden + d * mlp_hidden + mlp_hidden * d
    size += cfg["dim"] * 2
    if not cfg.get("weightTying", False):
        size += cfg["dim"] * cfg["vocabSize"]
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

# ============ GLOBAL STATE ============
db = {"tasks": {}, "pendingSubmissions": {}, "shardMeta": {}}
active_ips: Dict[str, float] = {}
blacklisted_subnets: Dict[str, float] = {}
global_weights: Optional[torch.Tensor] = None
global_step = 0
global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}

dataset_info = {
    "repoId": DATASET_CONFIG["repoId"],
    "totalChunks": 0,
    "initialized": False, 
    "lastChecked": None
}
data_position = {"currentChunkIdx": 0, "epoch": 0, "totalSamplesSeen": 0}
current_data_shard = {"chunkIdx": 0, "assignedIPs": [], "shardId": None, "createdAt": None}
dataset_stats = {"totalShardsProcessed": 0, "epochsCompleted": 0, "samplesProcessed": 0, "lastUpdate": None}

FL_CONFIG = {
    "CONSENSUS_REQUIRED": 3, "CHECKPOINT_INTERVAL": 100, "MAX_CHECKPOINTS": 10,
    "DELTA_NORM_BASE": 500.0, "DELTA_NORM_ABSOLUTE_MAX": 5000.0,
    "MAX_WEIGHT_NORM": 1000.0, "WEIGHT_DECAY": 0.0001, "SERVER_LR": 0.1,
    "IP_TIMEOUT_S": 600, "SHARD_TIMEOUT_S": 1800, "CLEANUP_INTERVAL_S": 60,
}
shard_lock = threading.Lock()

# ============ AUTH & LEADERBOARD ============
users_db = {"users": {}, "tokens": {}}
users_lock = threading.Lock()

BAD_WORDS = [
    "fuck", "shit", "ass", "bitch", "dick", "cock", "pussy", "cunt",
    "bastard", "damn", "piss", "crap", "twat",
    "nigger", "nigga", "faggot", "fag", "retard", "tranny",
    "whore", "slut", "rape", "pedo", "nazi", "hitler",
    "porn", "xxx", "hentai",
    "admin", "administrator", "root", "system", "moderator", "mod",
    "staff", "official", "crowdgpt", "vxtzq", "nicolas",
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

def generate_token() -> str: return secrets.token_urlsafe(32)

def contains_bad_word(text: str) -> bool:
    if not text: return False
    lower = text.lower()
    normalized = lower.translate(str.maketrans("013@$", "oieas"))
    for word in BAD_WORDS:
        if word in lower or word in normalized: return True
    return any(c in text for c in ['\n', '\r', '\t'])

def get_current_user(request: Request) -> Optional[dict]:
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "): return None
    token = auth_header[7:].strip()
    with users_lock:
        user_uuid = users_db["tokens"].get(token)
        if not user_uuid: return None
        return users_db["users"].get(user_uuid)

def compute_trust_score(stats: dict) -> float:
    return round(min(1.0, 0.5 + (stats.get("contributions", 0) / 200.0)), 2)

def compute_global_rank(user_uuid: str) -> int:
    with users_lock:
        sorted_users = sorted(users_db["users"].values(), key=lambda u: u["stats"].get("totalTokens", 0), reverse=True)
        for i, u in enumerate(sorted_users):
            if u["uuid"] == user_uuid: return i + 1
    return 0

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
    blacklisted_subnets[subnet] = time.time() + hours*3600
    log.warning(f"🔨 BANNED {subnet} for {hours}h — {reason}")

def get_client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd: return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

# ============ DATASET STATE ============
def save_dataset_state():
    try:
        with open(DATASET_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({"datasetInfo": dataset_info, "dataPosition": data_position, "datasetStats": dataset_stats}, f, indent=2)
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
    
    # Default to configured value
    dataset_info["totalChunks"] = DATASET_CONFIG.get("totalChunks", 100)
    
    # Try to auto-detect chunks if a local 'chunks' directory exists on the server
    chunks_dir = BASE_DIR / "chunks"
    if chunks_dir.exists():
        local_chunks = sorted(list(chunks_dir.glob("chunk_*.bin")))
        if local_chunks:
            dataset_info["totalChunks"] = len(local_chunks)
            log.info(f"✅ Auto-detected {dataset_info['totalChunks']} local chunks in server chunks/ directory")
        else:
            log.info(f"⚠️ Server chunks/ dir empty, using configured totalChunks: {dataset_info['totalChunks']}")
    else:
        log.info(f"✅ Dataset initialized with configured totalChunks: {dataset_info['totalChunks']} (clients will use local chunks/)")
        
    dataset_info["initialized"] = True
    dataset_info["lastChecked"] = datetime.now(timezone.utc).isoformat()
    save_dataset_state()

# ============ SHARD MANAGEMENT ============
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
    global current_data_shard
    if not dataset_info["initialized"]: initialize_dataset()
    with shard_lock:
        solo = _active_ip_count() < FL_CONFIG["CONSENSUS_REQUIRED"]
        is_new_client = (current_data_shard["shardId"] is None) or (client_ip not in current_data_shard["assignedIPs"])
        need_new_shard = False
        if current_data_shard["shardId"] is None: need_new_shard = True
        elif solo: need_new_shard = True
        elif is_new_client and len(current_data_shard["assignedIPs"]) >= FL_CONFIG["CONSENSUS_REQUIRED"]: need_new_shard = True
        if need_new_shard:
            if current_data_shard["shardId"] is not None: _advance_chunk_pointer()
            mode = requested_mode if requested_mode in MODE_CONFIGS else "quick"
            shard_id = f"shard_{data_position['epoch']}_{data_position['currentChunkIdx']}_{int(time.time()*1000)}"
            current_data_shard = {"chunkIdx": data_position["currentChunkIdx"], "assignedIPs": [], "shardId": shard_id, "createdAt": int(time.time()*1000)}
            db["shardMeta"][shard_id] = {"mode": mode, "createdAt": int(time.time()*1000), "required": 1 if solo else FL_CONFIG["CONSENSUS_REQUIRED"], "chunkIdx": data_position["currentChunkIdx"]}
            db["pendingSubmissions"][shard_id] = []
            log.info(f"📦 New shard {shard_id} chunk={current_data_shard['chunkIdx']} mode={mode} ({'solo' if solo else 'consensus'})")
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
            if len(buf) != MODEL_SIZE: return False
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

# ============ HUGGINGFACE INTEGRATION ============
last_hf_upload_time = 0

def upload_to_huggingface():
    global last_hf_upload_time
    if not HF_ENABLED: return
    now = time.time()
    if now - last_hf_upload_time < HF_UPLOAD_INTERVAL_HOURS * 3600: return
    try:
        log.info(f"🤗 Uploading to HuggingFace: {HF_REPO_ID}...")
        api = HfApi(token=HF_TOKEN)
        temp_dir = CHECKPOINT_DIR / "hf_upload_temp"
        temp_dir.mkdir(exist_ok=True)
        weights_fp32 = global_weights.cpu()
        weights_bf16 = weights_fp32.to(torch.bfloat16)
        save_file({"weights": weights_fp32}, str(temp_dir / "model_fp32.safetensors"))
        save_file({"weights": weights_bf16}, str(temp_dir / "model_bf16.safetensors"))
        with open(temp_dir / "config.json", 'w') as f:
            json.dump({
                "architecture": MODEL_CONFIG["architecture"], "vocabSize": MODEL_CONFIG["vocabSize"],
                "dim": MODEL_CONFIG["dim"], "nLayers": MODEL_CONFIG["nLayers"],
                "nHeads": MODEL_CONFIG["nHeads"], "nKvHeads": MODEL_CONFIG["nKvHeads"],
                "headDim": MODEL_CONFIG["headDim"], "maxSeqLen": MODEL_CONFIG["maxSeqLen"],
                "mlpHidden": MODEL_CONFIG["mlpHidden"], "weightTying": MODEL_CONFIG["weightTying"],
                "globalStep": global_step, "modelSize": MODEL_SIZE,
                "uploadedAt": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        api.upload_folder(folder_path=str(temp_dir), repo_id=HF_REPO_ID, repo_type="model",
                          commit_message=f"Update weights (step {global_step})")
        shutil.rmtree(temp_dir)
        last_hf_upload_time = now
        log.info(f"✅ Uploaded to {HF_REPO_ID} at step {global_step}")
    except Exception as e:
        log.error(f"❌ HF upload failed: {e}")

def hf_upload_loop():
    if not HF_ENABLED:
        log.warning("⚠️ HF uploads disabled")
        return
    log.info(f"🤗 HF uploads enabled: {HF_REPO_ID} every {HF_UPLOAD_INTERVAL_HOURS}h")
    time.sleep(300)
    while True:
        upload_to_huggingface()
        time.sleep(HF_UPLOAD_INTERVAL_HOURS * 3600)

# ============ NEW: HF STATUS SYNC ============
def update_hf_status():
    if not HF_ENABLED:
        return
    try:
        api = HfApi(token=HF_TOKEN)
        
        # Calculate real loss change
        loss_history = global_stats["lossHistory"]
        if len(loss_history) >= 2:
            first_loss = loss_history[0]
            last_loss = loss_history[-1]
            training_loss_change = round(((last_loss - first_loss) / first_loss) * 100, 1) if first_loss > 0 else 0.0
        else:
            training_loss_change = 0.0
            
        now = time.time()
        active_gpus = sum(1 for ts in active_ips.values() if now - ts < FL_CONFIG["IP_TIMEOUT_S"])
        
        with users_lock:
            total_participants = len(users_db["users"])
            
        status_data = {
            "totalParticipants": total_participants,
            "activeGpus": active_gpus,
            "totalTokens": global_stats["totalTokens"],
            "trainingLossChange": training_loss_change,
            "globalStep": global_step,
            "lastLoss": global_stats["lastLoss"],
            "updatedAt": datetime.now(timezone.utc).isoformat()
        }
        
        # Write locally first
        status_file = BASE_DIR / "status.json"
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)
            
        # Upload to HF
        api.upload_file(
            path_or_fileobj=str(status_file),
            path_in_repo="status.json",
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="chore: auto-update status.json"
        )
        log.info(f"✅ Uploaded status.json to HF (step {global_step}, {active_gpus} GPUs)")
    except Exception as e:
        log.error(f"❌ HF status upload failed: {e}")

def hf_status_loop():
    if not HF_ENABLED:
        log.warning("⚠️ HF status updates disabled")
        return
    log.info(f"🤗 HF status updates enabled: {HF_REPO_ID} every 5 minutes")
    time.sleep(60) # Wait 60s on startup to let things settle
    while True:
        update_hf_status()
        time.sleep(300) # 5 minutes

# ============ VECTOR MATH ============
def compute_max_allowed_norm(local_steps, rank=None):
    steps = max(1, int(local_steps or 1))
    base = FL_CONFIG["DELTA_NORM_BASE"] * math.sqrt(steps / 100.0)
    rank_factor = np.sqrt(rank / LORA_CONFIG["FULL_RANK"]) if rank and rank < LORA_CONFIG["FULL_RANK"] else 1.0
    return base * rank_factor

def apply_sequential_update(delta: np.ndarray, client_ip: str, rank: int):
    global global_weights, global_step
    if np.isnan(delta).any() or np.isinf(delta).any():
        log.error("🚨 Sequential update rejected: NaN/Inf in delta")
        return
    delta_tensor = torch.from_numpy(delta)
    with shard_lock:
        global_weights.mul_(1.0 - FL_CONFIG["WEIGHT_DECAY"])
        global_weights.add_(delta_tensor, alpha=FL_CONFIG["SERVER_LR"])
        norm = torch.norm(global_weights).item()
        if norm > FL_CONFIG["MAX_WEIGHT_NORM"]:
            global_weights.mul_(FL_CONFIG["MAX_WEIGHT_NORM"] / norm)
            log.warning(f"⚠️ global weight norm {norm:.2f} clipped")
        global_step += 1
    log.info(f"✅ Sequential update step {global_step} from {client_ip} (rank={rank})")
    save_latest_checkpoint()
    handle_checkpoints()

def handle_checkpoints():
    if global_step % FL_CONFIG["CHECKPOINT_INTERVAL"] != 0: return
    try:
        fp = CHECKPOINT_DIR / f"model_step_{global_step}.bin"
        global_weights.cpu().numpy().tofile(fp)
        with open(CHECKPOINT_DIR / f"model_step_{global_step}.json", 'w', encoding='utf-8') as f:
            json.dump({"globalStep": global_step, "savedAt": datetime.now(timezone.utc).isoformat(), "modelSize": MODEL_SIZE}, f, indent=2)
        files = sorted([f for f in CHECKPOINT_DIR.iterdir() if f.name.startswith("model_step_") and f.suffix == ".bin"],
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
            if now - active_ips[ip] > FL_CONFIG["IP_TIMEOUT_S"]: del active_ips[ip]
        for sn in list(blacklisted_subnets.keys()):
            if now > blacklisted_subnets[sn]: del blacklisted_subnets[sn]

def aggregation_watchdog():
    while True:
        time.sleep(FL_CONFIG["CLEANUP_INTERVAL_S"])
        now_ms = int(time.time()*1000)
        with shard_lock:
            for shard_id in list(db["pendingSubmissions"].keys()):
                subs = db["pendingSubmissions"].get(shard_id, [])
                if not subs: continue
                meta = db["shardMeta"].get(shard_id, {})
                age_s = (now_ms - meta.get("createdAt", now_ms)) / 1000
                if age_s > FL_CONFIG["SHARD_TIMEOUT_S"]:
                    log.info(f"🧹 Watchdog cleaning stale shard {shard_id} (age={age_s}s)")
                    db["pendingSubmissions"][shard_id] = []

threading.Thread(target=cleanup_loop, daemon=True).start()
threading.Thread(target=aggregation_watchdog, daemon=True).start()
threading.Thread(target=hf_upload_loop, daemon=True).start()
threading.Thread(target=hf_status_loop, daemon=True).start()

# ============ FASTAPI ============
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="CrowdGPT Swarm Coordinator")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Task-Id", "X-Shard-Id", "X-Chunk-Idx", "X-Global-Step", "X-Model-Size", "X-Quantized"])

# ============ FL ROUTES ============
@app.get("/fl/task")
async def get_task(request: Request, mode: str = "quick", format: str = "bf16", skip_weights: bool = False):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip): raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")
    active_ips[client_ip] = time.time()
    shard = get_or_advance_shard(client_ip, requested_mode=mode)
    shard_id = shard["shardId"]
    meta = db["shardMeta"].get(shard_id, {})
    shard_mode = meta.get("mode", mode if mode in MODE_CONFIGS else "quick")
    cfg = MODE_CONFIGS[shard_mode]
    task_id = f"{shard_id}_{secrets.token_hex(4)}"
    db["tasks"][task_id] = {"shardId": shard_id, "chunkIdx": shard["chunkIdx"], "mode": shard_mode, "requestedByIp": client_ip, "createdAt": int(time.time()*1000)}
    
    # chunk_offset is no longer relevant for local chunk reading
    chunk_offset = 0 
    
    metadata = {
        "taskId": task_id, 
        "globalStep": global_step, 
        "modelSize": MODEL_SIZE, 
        "modelConfig": MODEL_CONFIG,
        "trainingConfig": cfg, 
        "weightFormat": format,
        "datasetConfig": {
            "chunkIdx": shard["chunkIdx"],
            "totalChunks": dataset_info["totalChunks"],
            "epoch": data_position["epoch"],
            "tokensPerSample": DATASET_CONFIG["tokensPerSample"],
            "tokenBytes": DATASET_CONFIG["tokenBytes"],
            "repoId": dataset_info.get("repoId", DATASET_REPO_ID)
        },
        "shardConfig": {"shardId": shard_id, "slot": len(shard["assignedIPs"]),
                        "total": meta.get("required", FL_CONFIG["CONSENSUS_REQUIRED"])},
        "loraConfig": {"defaultRank": LORA_CONFIG["DEFAULT_RANK"], "fullRank": LORA_CONFIG["FULL_RANK"],
                       "rankTiers": LORA_CONFIG["RANK_TIERS"], "weightByRank": LORA_CONFIG["WEIGHT_BY_RANK"]}
    }
    if HF_ENABLED: metadata["huggingface"] = {"repoId": HF_REPO_ID, "enabled": True}
    metadata["tokenizerConfig"] = {"name": "Qwen/Qwen3-0.6B", "vocabSize": MODEL_CONFIG["vocabSize"]}
    metadata_json = json.dumps(metadata).encode('utf-8')
    metadata_length = struct.pack('<I', len(metadata_json))
    
    if skip_weights:
        log.info(f"📤 task {task_id} shard={shard_id} chunk={shard['chunkIdx']} mode={shard_mode} ip={client_ip} (skipping weights)")
        return Response(content=metadata_length + metadata_json, media_type="application/octet-stream",
            headers={"X-Task-Id": task_id, "X-Shard-Id": shard_id, "X-Chunk-Idx": str(shard["chunkIdx"]),
                     "X-Global-Step": str(global_step), "X-Model-Size": str(MODEL_SIZE), "X-Quantized": format})

    weights_np = global_weights.cpu().numpy()
    if format == "fp8": compressed = compress_weights_fp8(weights_np)
    elif format == "fp16": compressed = compress_weights_fp16(weights_np)
    else: compressed = compress_weights_bf16(weights_np)
    
    log.info(f"📤 task {task_id} shard={shard_id} chunk={shard['chunkIdx']} mode={shard_mode} ip={client_ip} (sending {len(compressed)/(1024*1024):.0f}MB)")
    return Response(content=metadata_length + metadata_json + compressed, media_type="application/octet-stream",
        headers={"X-Task-Id": task_id, "X-Shard-Id": shard_id, "X-Chunk-Idx": str(shard["chunkIdx"]),
                 "X-Global-Step": str(global_step), "X-Model-Size": str(MODEL_SIZE), "X-Quantized": format})

@app.post("/fl/submit")
async def submit_delta(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip): raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")
    active_ips[client_ip] = time.time()
    raw = await request.body()
    if not raw: raise HTTPException(status_code=400, detail="Empty payload")
    try:
        buf = gzip.decompress(raw) if (len(raw) >= 2 and raw[0] == 0x1f and raw[1] == 0x8b) else raw
        payload_len = struct.unpack('<I', buf[:4])[0]
        body = json.loads(buf[4:4+payload_len].decode('utf-8'))
        delta_bytes = buf[4+payload_len:]
        fmt = body.get("weightFormat", "bf16")
        if fmt == "fp8": delta = fp8_array_to_float32(np.frombuffer(delta_bytes, dtype=np.uint8))
        elif fmt == "fp16": delta = fp16_array_to_float32(np.frombuffer(delta_bytes, dtype=np.uint16))
        elif fmt == "fp32": delta = np.frombuffer(delta_bytes, dtype=np.float32)
        else: delta = bf16_array_to_float32(np.frombuffer(delta_bytes, dtype=np.uint16))
    except Exception as e:
        log.error(f"❌ parse failed: {e}")
        raise HTTPException(status_code=400, detail="Binary parsing failed")
    task_id = body.get("taskId")
    if task_id not in db["tasks"]: raise HTTPException(status_code=404, detail="Task expired")
    task = db["tasks"][task_id]
    shard_id = task["shardId"]
    if len(delta) != MODEL_SIZE: raise HTTPException(status_code=400, detail=f"Size mismatch: expected {MODEL_SIZE}, got {len(delta)}")
    if np.isnan(delta).any() or np.isinf(delta).any():
        ban_subnet(client_ip, "submitted NaN/Inf delta")
        raise HTTPException(status_code=400, detail="Invalid delta (NaN/Inf)")
    rank = body.get("loraRank", LORA_CONFIG["FULL_RANK"])
    local_steps = body.get("localSteps", 1)
    loss = body.get("loss")
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
    apply_sequential_update(delta, client_ip, rank)
    with shard_lock:
        subs = db["pendingSubmissions"].setdefault(shard_id, [])
        if not any(s["nodeIp"] == client_ip for s in subs):
            subs.append({"nodeIp": client_ip, "delta_norm": delta_norm, "rank": rank, "loss": loss,
                         "localSteps": local_steps, "tokensProcessed": body.get("tokensProcessed", 0),
                         "normalized": normalized, "submittedAt": int(time.time()*1000)})
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
    auth_user = get_current_user(request)
    if auth_user:
        with users_lock:
            auth_user["stats"]["contributions"] = auth_user["stats"].get("contributions", 0) + 1
            auth_user["stats"]["totalTokens"] = auth_user["stats"].get("totalTokens", 0) + body.get("tokensProcessed", 0)
            auth_user["stats"]["totalSteps"] = auth_user["stats"].get("totalSteps", 0) + (local_steps or 0)
            save_users_db()
    return {"status": "success", "globalStep": global_step, "normalized": normalized, "shardId": shard_id, "chunkIdx": task["chunkIdx"], "loss": loss}

# ============ AUTH ROUTES ============
@app.post("/auth/register")
async def register(request: Request):
    data = await request.json()
    username, email, password = data.get("username", "").strip(), data.get("email", "").strip(), data.get("password", "")
    if not username or not (3 <= len(username) <= 20): raise HTTPException(400, "Username must be 3-20 characters")
    if not username.replace("_", "").replace("-", "").isalnum(): raise HTTPException(400, "Username can only contain letters, numbers, _ and -")
    if contains_bad_word(username): raise HTTPException(400, "Username contains prohibited words")
    if not email or "@" not in email or "." not in email.split("@")[-1]: raise HTTPException(400, "Invalid email")
    if not password or len(password) < 6: raise HTTPException(400, "Password must be at least 6 characters")
    with users_lock:
        for u in users_db["users"].values():
            if u["username_lower"] == username.lower(): raise HTTPException(400, "Username already taken")
            if u.get("email", "").lower() == email.lower(): raise HTTPException(400, "Email already registered")
        user_uuid = str(uuid.uuid4())
        salt = secrets.token_hex(16)
        users_db["users"][user_uuid] = {"uuid": user_uuid, "username": username, "username_lower": username.lower(),
                                        "email": email, "password_hash": hash_password(password, salt), "salt": salt,
                                        "created_at": datetime.now(timezone.utc).isoformat(),
                                        "stats": {"contributions": 0, "totalTokens": 0, "totalSteps": 0}}
        users_db["tokens"][generate_token()] = user_uuid
        save_users_db()
    log.info(f"✅ Registered user {username} ({user_uuid[:8]})")
    return {"token": generate_token(), "uuid": user_uuid, "username": username}

@app.post("/auth/login")
async def auth_login(request: Request):
    data = await request.json()
    username, password = data.get("username", "").strip(), data.get("password", "")
    with users_lock:
        user = next((u for u in users_db["users"].values() if u["username_lower"] == username.lower()), None)
        if not user: raise HTTPException(401, "Invalid credentials")
        if hash_password(password, user["salt"]) != user["password_hash"]: raise HTTPException(401, "Invalid credentials")
        token = generate_token()
        users_db["tokens"][token] = user["uuid"]
        save_users_db()
    log.info(f"✅ Login: {username}")
    return {"token": token, "uuid": user["uuid"], "username": user["username"]}

@app.get("/auth/me")
async def auth_me(request: Request):
    user = get_current_user(request)
    if not user: raise HTTPException(401, "Unauthorized")
    return {"uuid": user["uuid"], "username": user["username"], "email": user.get("email"),
            "stats": user["stats"], "trustScore": compute_trust_score(user["stats"]), "globalRank": compute_global_rank(user["uuid"])}

# ============ ADMIN / WATCHDOG ROUTES ============
@app.get("/api/checkpoints")
async def list_checkpoints(secret: str = ""):
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    checkpoints = []
    for f in CHECKPOINT_DIR.iterdir():
        if f.name.startswith("model_step_") and f.suffix == ".bin":
            try:
                step = int(f.stem.split("_")[2])
                checkpoints.append({"step": step, "size": f.stat().st_size, "mtime": f.stat().st_mtime})
            except (IndexError, ValueError): continue
    checkpoints.sort(key=lambda x: x["step"], reverse=True)
    return {"current_step": global_step, "checkpoints": checkpoints, "count": len(checkpoints)}

@app.post("/api/admin/reload")
async def reload_checkpoint(step: int = -1, secret: str = ""):
    global global_weights, global_step
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    if step == -1:
        if not load_latest_checkpoint(): raise HTTPException(404, "No checkpoint found")
        target_step = global_step
    else:
        cp_file = CHECKPOINT_DIR / f"model_step_{step}.bin"
        if not cp_file.exists(): raise HTTPException(404, f"Checkpoint step {step} not found")
        buf = np.fromfile(cp_file, dtype=np.float32)
        if len(buf) != MODEL_SIZE: raise HTTPException(400, f"Checkpoint size mismatch: {len(buf)} vs {MODEL_SIZE}")
        global_weights = torch.from_numpy(buf.copy())
        global_step = step
        target_step = step
    log.warning(f"🔄 ADMIN RELOAD to step {target_step}")
    save_latest_checkpoint()
    return {"status": "reloaded", "step": target_step, "weight_norm": float(torch.norm(global_weights).item())}

@app.post("/api/admin/push_hf")
async def force_hf_push(secret: str = ""):
    global last_hf_upload_time
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    if not HF_ENABLED: raise HTTPException(503, "HF uploads disabled")
    last_hf_upload_time = 0
    upload_to_huggingface()
    return {"status": "uploaded", "step": global_step}

@app.post("/api/admin/reset")
async def admin_reset(secret: str = ""):
    global global_weights, global_step
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    torch.manual_seed(42)
    global_weights = torch.randn(MODEL_SIZE) * 0.02
    global_step = 0
    save_latest_checkpoint()
    log.warning("🔄 ADMIN RESET to random weights")
    return {"status": "reset", "step": 0}

# ============ PUBLIC STATS & LEADERBOARD ============
@app.get("/api/stats")
async def get_stats(request: Request):
    recent = global_stats["lossHistory"][-10:]
    avg_loss = sum(recent)/len(recent) if recent else 0.0
    
    loss_history = global_stats["lossHistory"]
    if len(loss_history) >= 2:
        first_loss = loss_history[0]
        last_loss = loss_history[-1]
        training_loss_change = round(((last_loss - first_loss) / first_loss) * 100, 1) if first_loss > 0 else 0.0
    else:
        training_loss_change = 0.0
        
    with users_lock: 
        total_users = len(users_db["users"])
        
    active_gpus = _active_ip_count()
    
    auth_user = get_current_user(request)
    your_contribution = 0.0
    if auth_user:
        total_tokens = global_stats["totalTokens"]
        user_tokens = auth_user["stats"].get("totalTokens", 0)
        if total_tokens > 0:
            your_contribution = round((user_tokens / total_tokens) * 100, 4)
    
    return {
        "totalParticipants": total_users,
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
        "weightNorm": float(torch.norm(global_weights).item()) if global_weights is not None else 0.0,
        "loraStats": lora_stats, 
        "totalUsers": total_users, 
        "activeContributors": active_gpus,
        "lossHistory": global_stats["lossHistory"][-20:],
        "datasetStats": {"totalChunks": dataset_info["totalChunks"], "currentChunk": data_position["currentChunkIdx"],
                         "currentEpoch": data_position["epoch"], "totalShardsProcessed": dataset_stats["totalShardsProcessed"],
                         "epochsCompleted": dataset_stats["epochsCompleted"]},
        "currentShard": {"shardId": current_data_shard["shardId"], "chunkIdx": current_data_shard["chunkIdx"],
                         "assignedIPs": len(current_data_shard["assignedIPs"]),
                         "required": db["shardMeta"].get(current_data_shard["shardId"], {}).get("required", FL_CONFIG["CONSENSUS_REQUIRED"])},
        "huggingface": {"repoId": HF_REPO_ID, "enabled": HF_ENABLED, "uploadIntervalHours": HF_UPLOAD_INTERVAL_HOURS} if HF_ENABLED else {},
    }

@app.get("/api/leaderboard")
async def get_leaderboard():
    with users_lock:
        users_list = [{"uuid": u["uuid"], "username": u["username"],
                       "contributions": u["stats"].get("contributions", 0),
                       "totalTokens": u["stats"].get("totalTokens", 0),
                       "totalSteps": u["stats"].get("totalSteps", 0),
                       "trustScore": compute_trust_score(u["stats"])} for u in users_db["users"].values()]
    users_list.sort(key=lambda x: x["totalTokens"], reverse=True)
    return users_list

@app.get("/api/checkpoint")
async def get_checkpoint(request: Request, format: str = "bf16"):
    if not LATEST_CHECKPOINT_FILE.exists(): raise HTTPException(status_code=404, detail="No checkpoint yet")
    w = np.fromfile(LATEST_CHECKPOINT_FILE, dtype=np.float32)[:MODEL_SIZE]
    if format == "fp8": c = compress_weights_fp8(w)
    elif format == "fp16": c = compress_weights_fp16(w)
    else: c = compress_weights_bf16(w)
    return Response(content=c, media_type="application/octet-stream",
                    headers={"X-Global-Step": str(global_step), "X-Model-Size": str(MODEL_SIZE)})

@app.get("/api/model_info")
async def get_model_info():
    return {"architecture": MODEL_CONFIG["architecture"], "parameters": MODEL_SIZE,
            "config": MODEL_CONFIG, "globalStep": global_step,
            "huggingface": {"repoId": HF_REPO_ID, "enabled": HF_ENABLED} if HF_ENABLED else {}}

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
    load_users_db()
    initialize_model()
    if not load_latest_checkpoint():
        save_latest_checkpoint()
    initialize_dataset()
    log.info("="*60)
    log.info(f"🚀 CrowdGPT Swarm Coordinator on port {PORT}")
    log.info(f"⚡ Mode: SEQUENTIAL AGGREGATION (SERVER_LR={FL_CONFIG['SERVER_LR']})")
    log.info(f"🧩 consensus_tracking={FL_CONFIG['CONSENSUS_REQUIRED']} shard_timeout={FL_CONFIG['SHARD_TIMEOUT_S']}s")
    log.info(f"👥 Users loaded: {len(users_db['users'])}")
    if HF_ENABLED:
        log.info(f"🤗 HF uploads: {HF_REPO_ID} every {HF_UPLOAD_INTERVAL_HOURS}h")
        log.info(f"📊 HF status.json updates: every 5 minutes")
    else:
        log.warning("⚠️ HF uploads disabled")
    log.info(f"🛡️  Admin watchdog: active (secret={ADMIN_SECRET[:8]}...)")
    log.info("="*60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", access_log=False)
