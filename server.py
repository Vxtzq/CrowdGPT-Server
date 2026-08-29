#!/usr/bin/env python3
"""
CrowdGPT Server - Continuous FL Coordinator (Ultimate Docker-OOM-Proof Edition)
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
from fastapi.responses import StreamingResponse # 🚨 NEW: For zero-copy streaming
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
        if not chunk: break
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

import socket
hostname = socket.gethostname()
try: local_ip = socket.gethostbyname(hostname)
except: local_ip = "unknown"
print(f"🌐 Container hostname: {hostname}")
print(f"🌐 Container local IP: {local_ip}")
print(f"🌐 Listening on: 0.0.0.0:{PORT}")
print("="*60)

HF_TOKEN = os.getenv("HF_TOKEN", "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")
HF_REPO_ID = "Vxtzq/Crowd-v1"
DATASET_REPO_ID = "Vxtzq/CrowdGPT"
HF_UPLOAD_INTERVAL_HOURS = 2
HF_ENABLED = HF_AVAILABLE and bool(HF_TOKEN)
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

DATASET_CONFIG = {
    "repoId": DATASET_REPO_ID, "totalChunks": 260, "subChunkMB": 10,
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
log.info(f"🧮 SotaGPT size: {MODEL_SIZE:,} params (~{MODEL_SIZE*4/1024/1024:.2f} MB in FP32, ~{MODEL_SIZE*2/1024/1024:.2f} MB in BF16)")

ENG_NUM_BUCKETS = 227865
ENG_PARAMS = ENG_NUM_BUCKETS * MODEL_CONFIG["dim"]
log.info(f"🧠 Engram size: {ENG_PARAMS:,} params (~{ENG_PARAMS*4/1024/1024:.2f} MB in FP32, ~{ENG_PARAMS*2/1024/1024:.2f} MB in BF16)")

# ============ QUANTIZATION ============
def float32_to_bf16_array(a): return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16).view(torch.uint16).numpy()
def bf16_array_to_float32(a): return torch.from_numpy(a.copy()).view(torch.uint16).view(torch.bfloat16).to(torch.float32).numpy()
def float32_to_fp16_array(a): return np.clip(np.ascontiguousarray(a, dtype=np.float32), -65504.0, 65504.0).astype(np.float16).view(np.uint16)
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
global_engram_weights: Optional[torch.Tensor] = None
global_step = 0
global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}

dataset_info = {"repoId": DATASET_CONFIG["repoId"], "totalChunks": 0, "subChunkBytes": DATASET_CONFIG["subChunkMB"] * 1024 * 1024, "initialized": False, "lastChecked": None}
data_position = {"currentChunkIdx": 0, "epoch": 0, "totalSamplesSeen": 0}
current_data_shard = {"chunkIdx": 0, "assignedIPs": [], "shardId": None, "createdAt": None}
dataset_stats = {"totalShardsProcessed": 0, "epochsCompleted": 0, "samplesProcessed": 0, "lastUpdate": None}

round_state = {
    "current_round": 1, "round_start_time": time.time(), "round_duration_hours": 2.0,
    "participating_clients": {}, "aggregations_this_round": 0,
    "last_aggregation_time": 0, "pending_deltas": [],
    "is_aggregating": False  # 🚨 NEW: Locks clients out during HF upload
}

FL_CONFIG = {
    "CONSENSUS_REQUIRED": 1, "CHECKPOINT_INTERVAL": 1, "MAX_CHECKPOINTS": 10,
    "DELTA_NORM_BASE": 500.0, "DELTA_NORM_ABSOLUTE_MAX": 5000.0,
    "MAX_WEIGHT_NORM": 1000.0, "WEIGHT_DECAY": 0.0001, "SERVER_LR": 0.1,
    "IP_TIMEOUT_S": 900, "SHARD_TIMEOUT_S": 1800, "CLEANUP_INTERVAL_S": 60,
}

ROUND_CONFIG = {
    "MAX_ROUND_HOURS": 2.0, "MIN_CLIENTS_TO_AGGREGATE": 1, "CLIENT_TIMEOUT_HOURS": 2.5,
    "AGGREGATION_COOLDOWN_SEC": 120, "ULTIMATUM_THRESHOLD": 0.90, "ULTIMATUM_TIMEOUT_MIN": 5,
}

shard_lock = threading.RLock()

# ============ AUTH & LEADERBOARD ============
users_db = {"users": {}, "tokens": {}}
users_lock = threading.Lock()

BAD_WORDS = ["fuck", "shit", "ass", "bitch", "dick", "cock", "pussy", "cunt", "bastard", "damn", "piss", "crap", "twat", "nigger", "nigga", "faggot", "fag", "retard", "tranny", "whore", "slut", "rape", "pedo", "nazi", "hitler", "porn", "xxx", "hentai", "admin", "administrator", "root", "system", "moderator", "mod", "staff", "official"]

def load_users_db():
    global users_db
    try:
        if USERS_DB_FILE.exists():
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f: users_db = json.load(f)
            log.info(f"👥 Loaded {len(users_db['users'])} users from disk")
    except Exception as e:
        log.error(f"❌ load_users_db: {e}")
        users_db = {"users": {}, "tokens": {}}

def save_users_db():
    try:
        with open(USERS_DB_FILE, 'w', encoding='utf-8') as f: json.dump(users_db, f, indent=2)
    except Exception as e: log.error(f"❌ save_users_db: {e}")

def hash_password(password: str, salt: str) -> str: return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
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

def compute_trust_score(stats: dict) -> float: return round(min(1.0, 0.5 + (stats.get("contributions", 0) / 200.0)), 2)
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
    except Exception as e: log.error(f"❌ save_dataset_state: {e}")

def load_dataset_state():
    global dataset_info, data_position, dataset_stats
    try:
        if DATASET_CACHE_FILE.exists():
            with open(DATASET_CACHE_FILE, 'r', encoding='utf-8') as f: state = json.load(f)
            dataset_info.update(state.get("datasetInfo", {}))
            data_position.update(state.get("dataPosition", {}))
            dataset_stats.update(state.get("datasetStats", {}))
            log.info(f"📂 dataset state: epoch={data_position['epoch']} chunk={data_position['currentChunkIdx']}")
    except Exception as e: log.error(f"❌ load_dataset_state: {e}")

def initialize_dataset():
    if dataset_info["initialized"]: return
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

def get_or_advance_shard(client_ip: str) -> dict:
    global current_data_shard
    if not dataset_info["initialized"]: initialize_dataset()
    with shard_lock:
        is_new_client = (current_data_shard["shardId"] is None) or (client_ip not in current_data_shard["assignedIPs"])
        need_new_shard = current_data_shard["shardId"] is None or (is_new_client and len(current_data_shard["assignedIPs"]) >= FL_CONFIG["CONSENSUS_REQUIRED"])
        if need_new_shard:
            if current_data_shard["shardId"] is not None: _advance_chunk_pointer()
            shard_id = f"round_{round_state['current_round']}_{data_position['currentChunkIdx']}_{int(time.time()*1000)}"
            current_data_shard = {"chunkIdx": data_position["currentChunkIdx"], "assignedIPs": [], "shardId": shard_id, "createdAt": int(time.time()*1000)}
            db["shardMeta"][shard_id] = {"createdAt": int(time.time()*1000), "required": FL_CONFIG["CONSENSUS_REQUIRED"], "chunkIdx": data_position["currentChunkIdx"]}
            db["pendingSubmissions"][shard_id] = []
            log.info(f"📦 New shard {shard_id} chunk={current_data_shard['chunkIdx']}")
        if client_ip not in current_data_shard["assignedIPs"]:
            current_data_shard["assignedIPs"].append(client_ip)
        save_dataset_state()
        return current_data_shard

# ============ MODEL WEIGHTS & STATE ============
def initialize_model():
    global global_weights, global_engram_weights
    torch.manual_seed(42)
    global_weights = (torch.randn(MODEL_SIZE) * 0.02).to(torch.bfloat16)
    global_engram_weights = (torch.randn(ENG_PARAMS) * 0.02).to(torch.bfloat16)
    log.info(f"🎲 initialized {MODEL_SIZE:,} base weights + {ENG_PARAMS:,} engram weights (BF16)")

def load_global_state():
    global global_step, global_stats, lora_stats
    if not GLOBAL_STATE_FILE.exists():
        log.warning("⚠️ global_state.json not found. Starting fresh.")
        return
    try:
        with open(GLOBAL_STATE_FILE, 'r', encoding='utf-8') as f: state = json.load(f)
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
            "globalStep": global_step, "savedAt": datetime.now(timezone.utc).isoformat(),
            "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS, "modelConfig": MODEL_CONFIG, 
            "globalStats": global_stats, "loraStats": lora_stats
        }
        temp_file = GLOBAL_STATE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f: json.dump(state_data, f, indent=2)
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
            global_weights = torch.from_numpy(buf.copy()).to(torch.bfloat16)
            log.info("✅ SUCCESS: Loaded base model weights.")
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
            global_engram_weights = torch.from_numpy(buf.copy()).to(torch.bfloat16)
            log.info("✅ SUCCESS: Loaded engram weights.")
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
                chunk = global_weights[i:i+chunk_size].cpu().float().numpy()
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
                chunk = global_engram_weights[i:i+chunk_size].cpu().float().numpy()
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

# ============ STATUS JSON & HUGGINGFACE INTEGRATION ============
last_hf_upload_time = 0

def bootstrap_huggingface():
    if not HF_ENABLED: return
    try:
        api = HfApi(token=HF_TOKEN)
        log.info("🔍 Checking HuggingFace repo for existing weights...")
        try: files = api.list_repo_files(HF_REPO_ID, repo_type="model")
        except Exception: files = []
            
        required_files = ["model_bf16.safetensors", "model_fp32.safetensors", "engram_bf16.safetensors", "engram_fp32.safetensors", "config.json"]
        missing_files = [f for f in required_files if f not in files]
        
        if not missing_files:
            log.info("✅ All required weight files found on HuggingFace.")
            return
            
        log.warning(f"⚠️ Missing files on HuggingFace: {missing_files}. Uploading current weights to bootstrap (Memory-safe mode)...")
        temp_dir = CHECKPOINT_DIR / "hf_bootstrap_temp"
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        w = global_weights.cpu().clone()
        save_file({"weights": w}, str(temp_dir / "model_bf16.safetensors"))
        del w; gc.collect()
        
        w = global_weights.cpu().float()
        save_file({"weights": w}, str(temp_dir / "model_fp32.safetensors"))
        del w; gc.collect()
        
        w = global_engram_weights.cpu().clone()
        save_file({"weights": w}, str(temp_dir / "engram_bf16.safetensors"))
        del w; gc.collect()
        
        w = global_engram_weights.cpu().float()
        save_file({"weights": w}, str(temp_dir / "engram_fp32.safetensors"))
        del w; gc.collect()
        
        with open(temp_dir / "config.json", 'w') as f:
            json.dump({
                "architecture": MODEL_CONFIG["architecture"], "vocabSize": MODEL_CONFIG["vocabSize"], 
                "dim": MODEL_CONFIG["dim"], "nLayers": MODEL_CONFIG["nLayers"], 
                "nHeads": MODEL_CONFIG["nHeads"], "nKvHeads": MODEL_CONFIG["nKvHeads"], 
                "headDim": MODEL_CONFIG["headDim"], "maxSeqLen": MODEL_CONFIG["maxSeqLen"], 
                "mlpHidden": MODEL_CONFIG["mlpHidden"], "weightTying": MODEL_CONFIG["weightTying"], 
                "globalStep": global_step, "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS, 
                "uploadedAt": datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
            
        api.upload_folder(folder_path=str(temp_dir), repo_id=HF_REPO_ID, repo_type="model", commit_message="chore: bootstrap initial weights and engram")
        shutil.rmtree(temp_dir)
        log.info(f"✅ Successfully bootstrapped missing files to {HF_REPO_ID}")
    except Exception as e:
        log.error(f"❌ HuggingFace bootstrap failed: {e}")

def upload_to_huggingface():
    global last_hf_upload_time
    if not HF_ENABLED: return
    try:
        log.info(f"🤗 Uploading to HuggingFace: {HF_REPO_ID} (Memory-safe mode)...")
        api = HfApi(token=HF_TOKEN)
        temp_dir = CHECKPOINT_DIR / "hf_upload_temp"
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        w = global_weights.cpu().clone()
        save_file({"weights": w}, str(temp_dir / "model_bf16.safetensors"))
        del w; gc.collect()
        
        w = global_weights.cpu().float()
        save_file({"weights": w}, str(temp_dir / "model_fp32.safetensors"))
        del w; gc.collect()
        
        w = global_engram_weights.cpu().clone()
        save_file({"weights": w}, str(temp_dir / "engram_bf16.safetensors"))
        del w; gc.collect()
        
        w = global_engram_weights.cpu().float()
        save_file({"weights": w}, str(temp_dir / "engram_fp32.safetensors"))
        del w; gc.collect()
        
        with open(temp_dir / "config.json", 'w') as f:
            json.dump({"architecture": MODEL_CONFIG["architecture"], "vocabSize": MODEL_CONFIG["vocabSize"], "dim": MODEL_CONFIG["dim"], "nLayers": MODEL_CONFIG["nLayers"], "nHeads": MODEL_CONFIG["nHeads"], "nKvHeads": MODEL_CONFIG["nKvHeads"], "headDim": MODEL_CONFIG["headDim"], "maxSeqLen": MODEL_CONFIG["maxSeqLen"], "mlpHidden": MODEL_CONFIG["mlpHidden"], "weightTying": MODEL_CONFIG["weightTying"], "globalStep": global_step, "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS, "uploadedAt": datetime.now(timezone.utc).isoformat()}, f, indent=2)
            
        api.upload_folder(folder_path=str(temp_dir), repo_id=HF_REPO_ID, repo_type="model", commit_message=f"Update weights & engram (step {global_step}, round {round_state['current_round']})")
        shutil.rmtree(temp_dir)
        last_hf_upload_time = time.time()
        log.info(f"✅ Uploaded to {HF_REPO_ID} at step {global_step}")
    except Exception as e: 
        log.error(f"❌ HF upload failed: {e}")
        if 'temp_dir' in locals() and temp_dir.exists(): shutil.rmtree(temp_dir)

def update_and_save_status():
    loss_history = global_stats["lossHistory"]
    training_loss_change = round(((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100, 1) if len(loss_history) >= 2 and loss_history[0] > 0 else 0.0
    active_gpus = sum(1 for ts in active_ips.values() if time.time() - ts < FL_CONFIG["IP_TIMEOUT_S"])
    total_tokens = global_stats["totalTokens"]
    
    with users_lock:
        total_participants = len(users_db["users"])
        users_list = [
            {"uuid": u["uuid"], "username": u["username"], "contributions": u["stats"].get("contributions", 0), 
             "totalTokens": u["stats"].get("totalTokens", 0), "totalSteps": u["stats"].get("totalSteps", 0), 
             "trustScore": compute_trust_score(u["stats"])} 
            for u in users_db["users"].values()
        ]
    leaderboard_sorted = sorted(users_list, key=lambda x: x["totalTokens"], reverse=True)
    
    status_data = {
        "totalParticipants": total_participants, "totalContributions": lora_stats.get("total_submissions", 0), 
        "activeGpus": active_gpus, "totalTokens": total_tokens, "trainingLossChange": training_loss_change, 
        "globalStep": global_step, "lastLoss": global_stats["lastLoss"], "updatedAt": datetime.now(timezone.utc).isoformat(),
        "currentRound": round_state["current_round"]
    }
    
    try:
        with open(BASE_DIR / "status.json", 'w', encoding='utf-8') as f: json.dump(status_data, f, indent=2)
        with open(BASE_DIR / "leaderboard.json", 'w', encoding='utf-8') as f: json.dump(leaderboard_sorted, f, indent=2)
    except Exception as e:
        log.error(f"❌ Failed to save json files to server root: {e}")

    if HF_ENABLED:
        try:
            api = HfApi(token=HF_TOKEN)
            temp_dir = BASE_DIR / "hf_status_temp"
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            
            with open(temp_dir / "status.json", 'w', encoding='utf-8') as f: json.dump(status_data, f, indent=2)
            with open(temp_dir / "leaderboard.json", 'w', encoding='utf-8') as f: json.dump(leaderboard_sorted, f, indent=2)
                
            def format_tokens(n):
                if n >= 1e12: return f"{n/1e12:.1f}T"
                if n >= 1e9: return f"{n/1e9:.1f}B"
                if n >= 1e6: return f"{n/1e6:.1f}M"
                if n >= 1e3: return f"{n/1e3:.1f}K"
                return str(n)
                
            badge_data = {"schemaVersion": 1, "label": "Tokens Processed", "message": format_tokens(total_tokens), "color": "blue"}
            with open(temp_dir / "badge_tokens.json", 'w', encoding='utf-8') as f: json.dump(badge_data, f, indent=2)
                
            operations = [
                CommitOperationAdd(path_in_repo="status.json", path_or_fileobj=str(temp_dir / "status.json")),
                CommitOperationAdd(path_in_repo="leaderboard.json", path_or_fileobj=str(temp_dir / "leaderboard.json")),
                CommitOperationAdd(path_in_repo="badge_tokens.json", path_or_fileobj=str(temp_dir / "badge_tokens.json"))
            ]
            
            api.create_commit(repo_id=HF_REPO_ID, repo_type="model", operations=operations, commit_message=f"chore: update stats, leaderboard & badge ({badge_data['message']} tokens)")
            shutil.rmtree(temp_dir)
            log.info(f"✅ Uploaded status.json, leaderboard.json & badge to HF")
        except Exception as e: 
            log.error(f"❌ HF status upload failed: {e}")

def hf_status_loop():
    if not HF_ENABLED: return
    time.sleep(60)
    while True:
        update_and_save_status()
        time.sleep(300)

def emergency_save():
    log.info("🛑 Server shutting down, saving final state...")
    save_global_state()
    save_model_weights()
    save_engram_weights()
    save_users_db()
    update_and_save_status()

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
                chunk = global_weights[i:i+chunk_size].cpu().float().numpy()
                chunk.tofile(f)
                del chunk
                
        fp_eng = CHECKPOINT_DIR / f"engram_step_{global_step}.bin"
        with open(fp_eng, 'wb') as f:
            chunk_size = 10_000_000
            for i in range(0, ENG_PARAMS, chunk_size):
                chunk = global_engram_weights[i:i+chunk_size].cpu().float().numpy()
                chunk.tofile(f)
                del chunk
        gc.collect()
        
        with open(CHECKPOINT_DIR / f"model_step_{global_step}.json", 'w', encoding='utf-8') as f:
            json.dump({"globalStep": global_step, "savedAt": datetime.now(timezone.utc).isoformat(), "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS}, f, indent=2)
            
        files = sorted([f for f in CHECKPOINT_DIR.iterdir() if f.name.startswith("model_step_") and f.suffix == ".bin"], key=lambda f: int(f.stem.split("_")[2]))
        while len(files) > FL_CONFIG["MAX_CHECKPOINTS"]:
            old = files.pop(0); old.unlink()
            old_eng = CHECKPOINT_DIR / old.name.replace("model_", "engram_")
            if old_eng.exists(): old_eng.unlink()
            meta = CHECKPOINT_DIR / f"{old.stem}.json"
            if meta.exists(): meta.unlink()
    except Exception as e: log.error(f"❌ checkpoint: {e}")

# ============ BACKGROUND THREADS ============
def cleanup_loop():
    while True:
        time.sleep(FL_CONFIG["CLEANUP_INTERVAL_S"])
        now = time.time()
        now_ms = int(now * 1000)
        for ip in list(active_ips.keys()):
            if now - active_ips[ip] > FL_CONFIG["IP_TIMEOUT_S"]: del active_ips[ip]
        for sn in list(blacklisted_subnets.keys()):
            if now > blacklisted_subnets[sn]: del blacklisted_subnets[sn]
            
        task_timeout_ms = 72 * 60 * 60 * 1000
        expired_tasks = [tid for tid, meta in db["tasks"].items() if now_ms - meta.get("createdAt", 0) > task_timeout_ms]
        for tid in expired_tasks: del db["tasks"][tid]
            
        stale_shards = [sid for sid, meta in db["shardMeta"].items() if now_ms - meta.get("createdAt", 0) > FL_CONFIG["SHARD_TIMEOUT_S"] * 1000]
        for sid in stale_shards:
            db["shardMeta"].pop(sid, None)
            db["pendingSubmissions"].pop(sid, None)
            
        if expired_tasks or stale_shards:
            log.info(f"🧹 GC: Cleaned {len(expired_tasks)} dead tasks and {len(stale_shards)} stale shards")
            
        gc.collect()

def round_watchdog():
    while True:
        time.sleep(30)
        with shard_lock:
            round_elapsed_hours = (time.time() - round_state["round_start_time"]) / 3600
            round_elapsed_fraction = round_elapsed_hours / round_state["round_duration_hours"]
            
            if round_elapsed_fraction >= ROUND_CONFIG["ULTIMATUM_THRESHOLD"]:
                for client_ip, client_data in round_state["participating_clients"].items():
                    if not client_data["submitted"] and not client_data.get("ultimatum_sent"):
                        log.warning(f"⚠️ Sending ultimatum to {client_ip}")
                        client_data["ultimatum_sent"] = True
                        client_data["ultimatum_time"] = time.time()
            
            for client_ip in list(round_state["participating_clients"].keys()):
                client_data = round_state["participating_clients"][client_ip]
                if client_data.get("ultimatum_sent"):
                    time_since_ultimatum = (time.time() - client_data.get("ultimatum_time", 0)) / 60
                    if time_since_ultimatum > ROUND_CONFIG["ULTIMATUM_TIMEOUT_MIN"] and not client_data["submitted"]:
                        log.warning(f"🚫 Removing {client_ip} from round")
                        del round_state["participating_clients"][client_ip]
            
            has_clients = len(round_state["participating_clients"]) > 0
            all_submitted = has_clients and all(c["submitted"] for c in round_state["participating_clients"].values())
            
            should_end_round = (
                has_clients and (
                    round_elapsed_hours >= round_state["round_duration_hours"] or
                    (all_submitted and len(round_state["participating_clients"]) >= ROUND_CONFIG["MIN_CLIENTS_TO_AGGREGATE"])
                )
            )
            
            if should_end_round:
                round_state["is_aggregating"] = True  # 🚨 LOCK CLIENTS OUT
                log.info(f"⏰ Round {round_state['current_round']} complete! Aggregating and uploading to HF...")
                
                if round_state["pending_deltas"]:
                    apply_round_aggregation(round_state["pending_deltas"])
                    round_state["pending_deltas"] = []
                
                if HF_ENABLED:
                    upload_to_huggingface()
                
                round_state["current_round"] += 1
                round_state["round_start_time"] = time.time()
                round_state["participating_clients"] = {}
                round_state["aggregations_this_round"] = 0
                round_state["last_aggregation_time"] = time.time()
                round_state["is_aggregating"] = False  # 🚨 UNLOCK
                log.info(f"🚀 Started round {round_state['current_round']}")

def aggregation_watchdog():
    while True: time.sleep(FL_CONFIG["CLEANUP_INTERVAL_S"])

threading.Thread(target=cleanup_loop, daemon=True).start()
threading.Thread(target=aggregation_watchdog, daemon=True).start()
threading.Thread(target=round_watchdog, daemon=True).start()
threading.Thread(target=hf_status_loop, daemon=True).start()

# ============ FASTAPI ============
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="CrowdGPT Continuous FL Coordinator")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Task-Id", "X-Shard-Id", "X-Chunk-Idx", "X-Global-Step", "X-Model-Size", "X-Quantized", "X-Round", "X-Status"])

@app.get("/fl/heartbeat")
async def heartbeat(request: Request):
    client_ip = get_client_ip(request)
    if not is_blacklisted(client_ip): active_ips[client_ip] = time.time()
    return Response(content='{"status":"ok"}', media_type="application/json")

@app.get("/fl/round_status")
async def get_round_status(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip): raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")
    active_ips[client_ip] = time.time()
    
    with shard_lock:
        if client_ip not in round_state["participating_clients"]:
            round_state["participating_clients"][client_ip] = {"joined": time.time(), "submitted": False, "ultimatum_sent": False}
            log.info(f"👥 Client {client_ip} joined round {round_state['current_round']}")
    
    in_cooldown = (time.time() - round_state["last_aggregation_time"]) < ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"]
    round_elapsed_hours = (time.time() - round_state["round_start_time"]) / 3600
    
    return {
        "current_round": round_state["current_round"], "round_elapsed_hours": round_elapsed_hours,
        "max_round_hours": round_state["round_duration_hours"],
        "clients_in_round": len(round_state["participating_clients"]),
        "submitted_clients": sum(1 for c in round_state["participating_clients"].values() if c["submitted"]),
        "in_cooldown": in_cooldown, "global_step": global_step,
        "next_round_ready": not in_cooldown and round_state["last_aggregation_time"] > 0,
        "is_aggregating": round_state.get("is_aggregating", False)  # 🚨 NEW
    }

@app.get("/fl/task")
async def get_task(request: Request, format: str = "bf16", skip_weights: bool = False):
    if round_state.get("is_aggregating", False):
        return Response(content=json.dumps({"status": "wait", "message": "Server aggregating and uploading to HF, please wait"}).encode(), media_type="application/json", headers={"X-Status": "wait"})
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip): raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")
    active_ips[client_ip] = time.time()
    
    in_cooldown = (time.time() - round_state["last_aggregation_time"]) < ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"]
    if in_cooldown:
        wait_time = int(ROUND_CONFIG["AGGREGATION_COOLDOWN_SEC"] - (time.time() - round_state["last_aggregation_time"]))
        return Response(content=json.dumps({"status": "wait", "message": f"Server aggregating, wait {wait_time}s"}).encode(), media_type="application/json", headers={"X-Status": "wait"})
    
    shard = get_or_advance_shard(client_ip)
    shard_id = shard["shardId"]
    task_id = f"round_{round_state['current_round']}_{shard_id}"
    
    db["tasks"][task_id] = {"shardId": shard_id, "chunkIdx": shard["chunkIdx"], "round": round_state["current_round"], "requestedByIp": client_ip, "createdAt": int(time.time()*1000)}
    
    metadata = {
        "taskId": task_id, "globalStep": global_step, "currentRound": round_state["current_round"],
        "modelSize": MODEL_SIZE, "engramParams": ENG_PARAMS, "modelConfig": MODEL_CONFIG, "weightFormat": format,
        "datasetConfig": {"repoId": DATASET_CONFIG["repoId"], "chunkIdx": shard["chunkIdx"], "totalChunks": dataset_info["totalChunks"], "subChunkSize": DATASET_CONFIG["subChunkMB"] * 1024 * 1024, "epoch": data_position["epoch"], "tokensPerSample": DATASET_CONFIG["tokensPerSample"], "tokenBytes": DATASET_CONFIG["tokenBytes"]},
        "shardConfig": {"shardId": shard_id, "slot": len(shard["assignedIPs"]), "total": FL_CONFIG["CONSENSUS_REQUIRED"]},
        "roundConfig": {"maxHours": round_state["round_duration_hours"], "targetDurationHours": round_state["round_duration_hours"]}
    }
    if HF_ENABLED: metadata["huggingface"] = {"repoId": HF_REPO_ID, "enabled": True}
    
    metadata_json = json.dumps(metadata).encode('utf-8')
    metadata_length = struct.pack('<I', len(metadata_json))
    
    if skip_weights:
        return Response(content=metadata_length + metadata_json, media_type="application/octet-stream", headers={"X-Task-Id": task_id, "X-Round": str(round_state["current_round"]), "X-Global-Step": str(global_step)})
    
    # 🚨 ZERO-COPY STREAMING: Eliminates 2.6GB RAM spike!
    def weight_stream():
        yield metadata_length + metadata_json
        chunk_size = 10_000_000
        for i in range(0, MODEL_SIZE, chunk_size):
            chunk = global_weights[i:i+chunk_size].cpu()
            if format == "bf16":
                yield chunk.view(torch.uint16).numpy().tobytes()
            else:
                yield chunk.float().numpy().astype(np.float16).tobytes()
            del chunk
            
    return StreamingResponse(weight_stream(), media_type="application/octet-stream", headers={"X-Task-Id": task_id, "X-Round": str(round_state["current_round"]), "X-Global-Step": str(global_step)})

@app.post("/fl/submit")
async def submit_delta(request: Request):
    client_ip = get_client_ip(request)
    if is_blacklisted(client_ip): raise HTTPException(status_code=403, detail="Subnet temporarily blacklisted")
    active_ips[client_ip] = time.time()
    
    temp_compressed = None
    temp_decompressed = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin.gz") as tmp_file:
            temp_compressed = tmp_file.name
            async for chunk in request.stream(): tmp_file.write(chunk)
        
        with open(temp_compressed, 'rb') as f:
            magic = f.read(2)
            f.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dec") as dec_file:
                temp_decompressed = dec_file.name
                if len(magic) >= 2 and magic[0] == 0x1f and magic[1] == 0x8b:
                    with gzip.open(f, 'rb') as gz:
                        while True:
                            chunk = gz.read(1024 * 1024)
                            if not chunk: break
                            dec_file.write(chunk)
                else:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk: break
                        dec_file.write(chunk)
        
        with open(temp_decompressed, 'rb') as dec_f:
            payload_len_bytes = dec_f.read(4)
            if len(payload_len_bytes) < 4: raise HTTPException(status_code=400, detail="Invalid payload")
            payload_len = struct.unpack('<I', payload_len_bytes)[0]
            payload_bytes = dec_f.read(payload_len)
            body = json.loads(payload_bytes.decode('utf-8'))
            
            fmt = body.get("weightFormat", "bf16")
            has_engram = body.get("hasEngram", False)
            engram_sparse_count = body.get("engramSparseCount", 0)
            bytes_per_param = 2 if fmt in ("bf16", "fp16") else (1 if fmt == "fp8" else 4)
            
            # 🚨 ZERO-COPY CHUNKED DECODING: Eliminates 3.9GB RAM spike!
            delta = torch.zeros(MODEL_SIZE, dtype=torch.bfloat16)
            chunk_params = 10_000_000
            
            offset = 0
            while offset < MODEL_SIZE:
                read_params = min(chunk_params, MODEL_SIZE - offset)
                chunk_bytes = dec_f.read(read_params * bytes_per_param)
                if not chunk_bytes or len(chunk_bytes) < read_params * bytes_per_param:
                    raise HTTPException(status_code=400, detail="Incomplete base delta payload")
                
                buf = bytearray(chunk_bytes) # Writable buffer for torch
                
                if fmt == "fp8":
                    chunk_t = torch.frombuffer(buf, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
                elif fmt == "fp16":
                    chunk_t = torch.frombuffer(buf, dtype=torch.uint16).view(torch.float16).to(torch.bfloat16)
                elif fmt == "fp32":
                    chunk_t = torch.frombuffer(buf, dtype=torch.float32).to(torch.bfloat16)
                else: # bf16
                    chunk_t = torch.frombuffer(buf, dtype=torch.uint16).view(torch.bfloat16)
                    
                delta[offset:offset+read_params] = chunk_t
                offset += read_params
                del buf, chunk_t, chunk_bytes
                
            engram_indices = None
            engram_values = None
            if has_engram and engram_sparse_count > 0:
                idx_bytes_len = engram_sparse_count * 4
                val_bytes_len = engram_sparse_count * MODEL_CONFIG["dim"] * bytes_per_param
                idx_bytes = read_exact(dec_f, idx_bytes_len)
                val_bytes = read_exact(dec_f, val_bytes_len)
                if len(idx_bytes) < idx_bytes_len or len(val_bytes) < val_bytes_len: raise HTTPException(status_code=400, detail=f"Incomplete engram payload")
                
                engram_indices = np.frombuffer(bytearray(idx_bytes), dtype=np.uint32)
                
                val_buf = bytearray(val_bytes)
                if fmt == "fp16":
                    engram_values = torch.frombuffer(val_buf, dtype=torch.uint16).view(torch.float16).to(torch.bfloat16)
                elif fmt == "fp32":
                    engram_values = torch.frombuffer(val_buf, dtype=torch.float32).to(torch.bfloat16)
                else: # bf16
                    engram_values = torch.frombuffer(val_buf, dtype=torch.uint16).view(torch.bfloat16)
                    
                engram_values = engram_values.reshape(-1, MODEL_CONFIG["dim"])
                del val_buf
                
    except Exception as e:
        log.error(f"❌ Upload stream failed: {e}")
        raise HTTPException(status_code=400, detail=f"Upload stream failed: {str(e)}")
    finally:
        if temp_compressed and os.path.exists(temp_compressed): os.unlink(temp_compressed)
        if temp_decompressed and os.path.exists(temp_decompressed): os.unlink(temp_decompressed)

    if not delta.size: raise HTTPException(status_code=400, detail="Empty delta")
    
    task_id = body.get("taskId")
    if task_id not in db["tasks"]: raise HTTPException(status_code=404, detail="Task expired")
    task = db["tasks"][task_id]
    
    if len(delta) != MODEL_SIZE: raise HTTPException(status_code=400, detail=f"Size mismatch")
    if torch.isnan(delta).any() or torch.isinf(delta).any():
        ban_subnet(client_ip, "submitted NaN/Inf delta")
        raise HTTPException(status_code=400, detail="Invalid delta")
    
    norm_sq = 0.0
    chunk_size = 10_000_000
    for i in range(0, MODEL_SIZE, chunk_size):
        chunk = delta[i:i+chunk_size].float()
        norm_sq += torch.sum(chunk * chunk).item()
        del chunk
    delta_norm = math.sqrt(norm_sq)
    
    rank = body.get("loraRank", LORA_CONFIG["FULL_RANK"])
    local_steps = body.get("localSteps", 1)
    loss = body.get("loss")
    
    max_allowed = compute_max_allowed_norm(local_steps, rank)
    abs_max = max(FL_CONFIG["DELTA_NORM_ABSOLUTE_MAX"], max_allowed * 3)
    if delta_norm > abs_max:
        ban_subnet(client_ip, f"delta norm {delta_norm:.0f} > max {abs_max:.0f}")
        raise HTTPException(status_code=400, detail=f"Delta too large ({delta_norm:.2f})")
    
    normalized = False
    if delta_norm > max_allowed:
        delta.mul_(max_allowed / delta_norm)
        normalized = True
    
    with shard_lock:
        if client_ip in round_state["participating_clients"]:
            round_state["participating_clients"][client_ip]["submitted"] = True
            round_state["aggregations_this_round"] += 1
            round_state["pending_deltas"].append({
                "nodeIp": client_ip, "delta": delta, "engram_indices": engram_indices, "engram_values": engram_values,
                "delta_norm": delta_norm, "rank": rank, "loss": loss, "localSteps": local_steps, 
                "tokensProcessed": body.get("tokensProcessed", 0), "normalized": normalized, "submittedAt": int(time.time()*1000)
            })
            log.info(f"📥 Round {round_state['current_round']}: Client {client_ip} submitted")
    
    global_stats["totalTokens"] += body.get("tokensProcessed", 0)
    global_stats["totalSteps"] += local_steps or 0
    if loss and loss > 0 and math.isfinite(loss):
        global_stats["lastLoss"] = loss
        global_stats["lossHistory"].append(loss)
        global_stats["lossHistory"] = global_stats["lossHistory"][-100:]
    
    lora_stats["total_submissions"] += 1
    lora_stats["rank_distribution"][str(rank)] = lora_stats["rank_distribution"].get(str(rank), 0) + 1
    lora_stats["last_update"] = datetime.now(timezone.utc).isoformat()
    
    auth_user = get_current_user(request)
    if auth_user:
        tokens_to_add = body.get("tokensProcessed", 0)
        steps_to_add = local_steps or 0
        with users_lock:
            user_record = users_db["users"].get(auth_user["uuid"])
            if user_record:
                user_record["stats"]["contributions"] = user_record["stats"].get("contributions", 0) + 1
                user_record["stats"]["totalTokens"] = user_record["stats"].get("totalTokens", 0) + tokens_to_add
                user_record["stats"]["totalSteps"] = user_record["stats"].get("totalSteps", 0) + steps_to_add
                save_users_db()
    
    save_global_state()
    update_and_save_status()
    
    return {"status": "success", "globalStep": global_step, "normalized": normalized, "round": round_state["current_round"], "loss": loss}

def apply_round_aggregation(submissions: List[dict]):
    global global_weights, global_step
    if not submissions: return
    
    total_tokens = sum(s.get("tokensProcessed", 0) for s in submissions)
    if total_tokens == 0:
        log.warning("⚠️ No tokens processed this round, skipping aggregation")
        return
    
    log.info(f"🧮 Aggregating {len(submissions)} deltas ({total_tokens:,} total tokens)...")
    
    aggregated_delta = torch.zeros(MODEL_SIZE, dtype=torch.bfloat16)
    for sub in submissions:
        weight = sub.get("tokensProcessed", 0) / total_tokens
        aggregated_delta.add_(sub["delta"], alpha=weight)
        
    if torch.isnan(aggregated_delta).any() or torch.isinf(aggregated_delta).any():
        log.error("🚨 Aggregated base delta contains NaN/Inf, skipping update")
        del aggregated_delta; gc.collect()
        return
    
    with shard_lock:
        global_weights.mul_(1.0 - FL_CONFIG["WEIGHT_DECAY"])
        global_weights.add_(aggregated_delta, alpha=FL_CONFIG["SERVER_LR"])
        
        norm_sq = 0.0
        chunk_size = 10_000_000
        for i in range(0, MODEL_SIZE, chunk_size):
            chunk = global_weights[i:i+chunk_size].float()
            norm_sq += torch.sum(chunk * chunk).item()
            del chunk
        norm = math.sqrt(norm_sq)
        
        if norm > FL_CONFIG["MAX_WEIGHT_NORM"]:
            global_weights.mul_(FL_CONFIG["MAX_WEIGHT_NORM"] / norm)
            log.warning(f"⚠️ Global weight norm {norm:.2f} clipped")
        
        del aggregated_delta
        gc.collect()
        
        all_indices = []
        all_values = []
        token_weights = []
        
        for sub in submissions:
            if sub.get("engram_indices") is not None and len(sub["engram_indices"]) > 0:
                all_indices.append(sub["engram_indices"])
                all_values.append(sub["engram_values"])
                token_weights.append(sub.get("tokensProcessed", 0) / total_tokens)
                
        if all_indices:
            indices_t = torch.from_numpy(np.concatenate(all_indices).astype(np.int64))
            
            weighted_values = []
            for vals, w in zip(all_values, token_weights):
                weighted_values.append(vals * w)
            values_t = torch.cat(weighted_values, dim=0)
            del all_indices, all_values, weighted_values, token_weights
            gc.collect()
            
            unique_buckets = torch.unique(indices_t)
            
            engram_update = torch.zeros(ENG_PARAMS, dtype=torch.bfloat16)
            bucket_indices = indices_t * MODEL_CONFIG["dim"]
            dim_offsets = torch.arange(MODEL_CONFIG["dim"], dtype=torch.int64)
            flat_indices = (bucket_indices.unsqueeze(1) + dim_offsets.unsqueeze(0)).flatten()
            values_flat = values_t.flatten()
            
            engram_update.scatter_add_(0, flat_indices, values_flat)
            del flat_indices, values_flat, bucket_indices, dim_offsets, indices_t, values_t
            gc.collect()
            
            tensor_indices = (unique_buckets * MODEL_CONFIG["dim"]).unsqueeze(1) + torch.arange(MODEL_CONFIG["dim"])
            tensor_indices = tensor_indices.flatten().to(torch.long)
            
            agg_values = engram_update[tensor_indices]
            del engram_update
            gc.collect()
            
            current_vals = global_engram_weights[tensor_indices]
            current_vals.mul_(1.0 - FL_CONFIG["WEIGHT_DECAY"])
            current_vals.add_(agg_values, alpha=FL_CONFIG["SERVER_LR"])
            global_engram_weights[tensor_indices] = current_vals
            
            num_buckets = len(unique_buckets)
            del tensor_indices, agg_values, current_vals, unique_buckets
            gc.collect()
            
            log.info(f"🧠 Engram updated: {num_buckets} buckets modified (token-weighted)")
        
        global_step += 1
    
    avg_loss = sum(s["loss"] * s.get("tokensProcessed", 0) for s in submissions if s.get("loss")) / max(total_tokens, 1)
    log.info(f"✅ Aggregated step {global_step} from {len(submissions)} clients (tokens={total_tokens:,}, avg_loss={avg_loss:.4f})")
    handle_checkpoints()
    gc.collect()

# ============ AUTH ROUTES ============
@app.post("/auth/register")
async def register(request: Request):
    data = await request.json()
    username, email, password = data.get("username", "").strip(), data.get("email", "").strip(), data.get("password", "")
    if not username or not (3 <= len(username) <= 20): raise HTTPException(400, "Username must be 3-20 characters")
    if not username.replace("_", "").replace("-", "").isalnum(): raise HTTPException(400, "Invalid characters")
    if contains_bad_word(username): raise HTTPException(400, "Prohibited words")
    if not email or "@" not in email: raise HTTPException(400, "Invalid email")
    if not password or len(password) < 6: raise HTTPException(400, "Password too short")
    with users_lock:
        for u in users_db["users"].values():
            if u["username_lower"] == username.lower(): raise HTTPException(400, "Username taken")
            if u.get("email", "").lower() == email.lower(): raise HTTPException(400, "Email registered")
        user_uuid = str(uuid.uuid4())
        salt = secrets.token_hex(16)
        users_db["users"][user_uuid] = {"uuid": user_uuid, "username": username, "username_lower": username.lower(), "email": email, "password_hash": hash_password(password, salt), "salt": salt, "created_at": datetime.now(timezone.utc).isoformat(), "stats": {"contributions": 0, "totalTokens": 0, "totalSteps": 0}}
        users_db["tokens"][generate_token()] = user_uuid
        save_users_db()
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
    return {"token": token, "uuid": user["uuid"], "username": user["username"]}

@app.get("/auth/me")
async def auth_me(request: Request):
    user = get_current_user(request)
    if not user: raise HTTPException(401, "Unauthorized")
    return {"uuid": user["uuid"], "username": user["username"], "email": user.get("email"), "stats": user["stats"], "trustScore": compute_trust_score(user["stats"]), "globalRank": compute_global_rank(user["uuid"])}

# ============ ADMIN ROUTES ============
@app.get("/api/checkpoints")
async def list_checkpoints(secret: str = ""):
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    checkpoints = []
    for f in CHECKPOINT_DIR.iterdir():
        if f.name.startswith("model_step_") and f.suffix == ".bin":
            try: checkpoints.append({"step": int(f.stem.split("_")[2]), "size": f.stat().st_size, "mtime": f.stat().st_mtime})
            except: continue
    return {"current_step": global_step, "checkpoints": sorted(checkpoints, key=lambda x: x["step"], reverse=True)}

@app.post("/api/admin/reload")
async def reload_checkpoint(step: int = -1, secret: str = ""):
    global global_weights, global_step
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    if step == -1:
        load_global_state(); load_model_weights(); load_engram_weights()
    else:
        cp_file = CHECKPOINT_DIR / f"model_step_{step}.bin"
        if not cp_file.exists(): raise HTTPException(404, "Not found")
        buf = np.fromfile(cp_file, dtype=np.float32)
        if len(buf) != MODEL_SIZE: raise HTTPException(400, "Size mismatch")
        global_weights = torch.from_numpy(buf.copy()).to(torch.bfloat16)
        cp_eng_file = CHECKPOINT_DIR / f"engram_step_{step}.bin"
        if cp_eng_file.exists():
            buf_eng = np.fromfile(cp_eng_file, dtype=np.float32)
            if len(buf_eng) == ENG_PARAMS: global_engram_weights = torch.from_numpy(buf_eng.copy()).to(torch.bfloat16)
        global_step = step
    save_latest_checkpoint(); update_and_save_status()
    return {"status": "reloaded", "step": global_step}

@app.post("/api/admin/push_hf")
async def force_hf_push(secret: str = ""):
    global last_hf_upload_time
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    last_hf_upload_time = 0
    upload_to_huggingface()
    return {"status": "uploaded"}

@app.post("/api/admin/reset")
async def admin_reset(secret: str = ""):
    global global_weights, global_engram_weights, global_step, global_stats, lora_stats
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    torch.manual_seed(42)
    global_weights = (torch.randn(MODEL_SIZE) * 0.02).to(torch.bfloat16)
    global_engram_weights = (torch.randn(ENG_PARAMS) * 0.02).to(torch.bfloat16)
    global_step = 0
    global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
    lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}
    save_latest_checkpoint(); update_and_save_status()
    return {"status": "reset"}

# ============ PUBLIC STATS ============
@app.get("/api/stats")
async def get_stats(request: Request):
    recent = global_stats["lossHistory"][-10:]
    avg_loss = sum(recent)/len(recent) if recent else 0.0
    loss_history = global_stats["lossHistory"]
    training_loss_change = round(((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100, 1) if len(loss_history) >= 2 and loss_history[0] > 0 else 0.0
    with users_lock: total_users = len(users_db["users"])
    active_gpus = _active_ip_count()
    auth_user = get_current_user(request)
    your_contribution = 0.0
    if auth_user:
        total_tokens = global_stats["totalTokens"]
        if total_tokens > 0: your_contribution = round((auth_user["stats"].get("totalTokens", 0) / total_tokens) * 100, 4)
    return {"totalParticipants": total_users, "totalContributions": lora_stats.get("total_submissions", 0), "activeGpus": active_gpus, "totalTokens": global_stats["totalTokens"], "trainingLossChange": training_loss_change, "yourContribution": your_contribution, "activeNow": active_gpus, "totalSteps": global_stats["totalSteps"], "globalStep": global_step, "modelSize": MODEL_SIZE, "lastLoss": global_stats["lastLoss"], "avgLoss": avg_loss, "weightNorm": float(torch.norm(global_weights.float()).item()) if global_weights is not None else 0.0, "loraStats": lora_stats, "totalUsers": total_users, "activeContributors": active_gpus, "lossHistory": global_stats["lossHistory"][-20:], "datasetStats": {"totalChunks": dataset_info["totalChunks"], "currentChunk": data_position["currentChunkIdx"], "currentEpoch": data_position["epoch"], "totalShardsProcessed": dataset_stats["totalShardsProcessed"], "epochsCompleted": dataset_stats["epochsCompleted"]}, "currentRound": round_state["current_round"], "huggingface": {"repoId": HF_REPO_ID, "enabled": HF_ENABLED, "uploadIntervalHours": HF_UPLOAD_INTERVAL_HOURS} if HF_ENABLED else {}}

@app.get("/api/leaderboard")
async def get_leaderboard():
    lb_file = BASE_DIR / "leaderboard.json"
    if lb_file.exists():
        try:
            with open(lb_file, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception: pass
    with users_lock:
        users_list = [{"uuid": u["uuid"], "username": u["username"], "contributions": u["stats"].get("contributions", 0), "totalTokens": u["stats"].get("totalTokens", 0), "totalSteps": u["stats"].get("totalSteps", 0), "trustScore": compute_trust_score(u["stats"])} for u in users_db["users"].values()]
    return sorted(users_list, key=lambda x: x["totalTokens"], reverse=True)

@app.get("/api/checkpoint")
async def get_checkpoint(request: Request, format: str = "bf16"):
    if not LATEST_CHECKPOINT_FILE.exists(): raise HTTPException(status_code=404, detail="No checkpoint")
    w = np.fromfile(LATEST_CHECKPOINT_FILE, dtype=np.float32)[:MODEL_SIZE]
    if format == "fp8": c = compress_weights_fp8(w)
    elif format == "fp16": c = compress_weights_fp16(w)
    else: c = compress_weights_bf16(w)
    return Response(content=c, media_type="application/octet-stream", headers={"X-Global-Step": str(global_step), "X-Model-Size": str(MODEL_SIZE)})

@app.get("/api/engram_checkpoint")
async def get_engram_checkpoint(request: Request, format: str = "bf16"):
    if not ENGRAM_CHECKPOINT_FILE.exists(): raise HTTPException(status_code=404, detail="No engram checkpoint")
    w = np.fromfile(ENGRAM_CHECKPOINT_FILE, dtype=np.float32)[:ENG_PARAMS]
    if format == "fp8": c = compress_weights_fp8(w)
    elif format == "fp16": c = compress_weights_fp16(w)
    else: c = compress_weights_bf16(w)
    return Response(content=c, media_type="application/octet-stream", headers={"X-Global-Step": str(global_step), "X-Engram-Size": str(ENG_PARAMS)})

@app.get("/api/model_info")
async def get_model_info():
    return {"architecture": MODEL_CONFIG["architecture"], "parameters": MODEL_SIZE, "engramParams": ENG_PARAMS, "config": MODEL_CONFIG, "globalStep": global_step, "currentRound": round_state["current_round"], "huggingface": {"repoId": HF_REPO_ID, "enabled": HF_ENABLED} if HF_ENABLED else {}}

@app.post("/api/reset_weights")
async def reset_weights(secret: str = ""):
    if secret != ADMIN_SECRET: raise HTTPException(403, "Invalid secret")
    global global_weights, global_engram_weights, global_step, global_stats, lora_stats
    torch.manual_seed(42)
    global_weights = (torch.randn(MODEL_SIZE) * 0.02).to(torch.bfloat16)
    global_engram_weights = (torch.randn(ENG_PARAMS) * 0.02).to(torch.bfloat16)
    global_step = 0
    global_stats = {"totalTokens": 0, "totalSteps": 0, "lastLoss": 0.0, "lossHistory": []}
    lora_stats = {"total_submissions": 0, "rank_distribution": {}, "avg_effective_rank": 0, "last_update": None}
    save_latest_checkpoint(); update_and_save_status()
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
    bootstrap_huggingface()
    update_and_save_status()
    
    log.info("="*60)
    log.info(f"🚀 CrowdGPT Continuous FL Coordinator on port {PORT}")
    log.info(f"⚡ Mode: CONTINUOUS AGGREGATION (Round duration: {round_state['round_duration_hours']}h)")
    log.info(f"🧠 RAM Optimizations: Zero-Copy Streaming + Chunked Math + PyTorch Scatter")
    log.info(f"🛡️  Admin watchdog: active (secret={ADMIN_SECRET[:8]}...)")
    log.info(f"🔄 Initialized round {round_state['current_round']}")
    log.info("="*60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", access_log=False)
