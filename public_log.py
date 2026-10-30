#!/usr/bin/env python3
"""
public_log.py — sanitized PUBLIC event log + loss history for the CrowdGPT server.

Drop next to server.py (apply_public_log.py wires it in).  Nothing private ever goes in:
  * no IPs / subnets / client keys / tokens / secrets / emails (IPv4/IPv6 patterns are scrubbed from every string),
  * users appear by their real USERNAME (the leaderboard already shows these). Set
    CROWDGPT_PUBLIC_LOG_USERNAMES=0 to show a stable pseudonym "user-ab12cd" instead.

Files written next to server.py:
  public_events.jsonl   append-only, rotated at 20 MB (public_events.jsonl.1)
  public_log.json       last 1000 events          (static copy, pushed to HF every 5 min)
  loss_history.json     {"rounds": [...], "pools": [last 2000]}   (what your site graphs)
  public_log_salt.txt   secret salt for the pseudonyms (do NOT publish)
"""
import json, os, re, secrets, threading, time, hashlib, collections
from datetime import datetime, timezone
from pathlib import Path

_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?\b")
_IPV6 = re.compile(r"\b(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\b")
_BAD_KEYS = ("ip", "key", "secret", "token", "email", "password", "subnet", "client")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scrub(s: str, limit: int = 140) -> str:
    s = _IPV4.sub("[ip]", str(s))
    s = _IPV6.sub("[ip]", s)
    return s.replace("\n", " ").replace("\r", " ")[:limit]


class PublicLog:
    def __init__(self, base_dir, max_events_mem=2000, rotate_bytes=20 * 1024 * 1024):
        self.base = Path(base_dir)
        self.events_file = self.base / "public_events.jsonl"
        self.public_file = self.base / "public_log.json"
        self.loss_file = self.base / "loss_history.json"
        self.salt_file = self.base / "public_log_salt.txt"
        self.rotate_bytes = rotate_bytes
        self.show_usernames = os.getenv("CROWDGPT_PUBLIC_LOG_USERNAMES", "1") == "1"
        self._lock = threading.Lock()
        self._events = collections.deque(maxlen=max_events_mem)
        self.resolver = None              # server sets: uuid -> username
        self._salt = self._load_salt()
        self._round_samples = []          # (loss, tokens, pool_id) of cross-checked deltas of the CURRENT round
        self.rounds = []                  # one entry per finished round
        self.pools = collections.deque(maxlen=2000)   # one entry per surviving pool
        self._load()

    # ---------- persistence ----------
    def _load_salt(self) -> str:
        try:
            if self.salt_file.exists():
                return self.salt_file.read_text().strip()
            s = secrets.token_hex(16)
            self.salt_file.write_text(s)
            try: os.chmod(self.salt_file, 0o600)
            except Exception: pass
            return s
        except Exception:
            return secrets.token_hex(16)

    def _load(self):
        try:
            if self.events_file.exists():
                with open(self.events_file, "r", encoding="utf-8") as f:
                    for line in f.readlines()[-self._events.maxlen:]:
                        try: self._events.append(json.loads(line))
                        except Exception: pass
        except Exception: pass
        try:
            if self.loss_file.exists():
                d = json.loads(self.loss_file.read_text())
                self.rounds = list(d.get("rounds", []))[-20000:]
                for p in d.get("pools", [])[-2000:]: self.pools.append(p)
        except Exception: pass

    @staticmethod
    def _atomic_write(path: Path, text: str):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    # ---------- identity ----------
    def who(self, client_key_or_uuid, username: str = None) -> str:
        """Public name for an account. Accepts 'user:<uuid>[:dev:x]' or a bare uuid."""
        s = str(client_key_or_uuid or "")
        if s.startswith("user:"):
            s = s[5:].split(":dev:", 1)[0]
        if not s or s.startswith("ip:"):
            return "anonymous"
        if self.show_usernames:
            try: username = username or (self.resolver(s) if self.resolver else None)
            except Exception: username = None
            if username: return _scrub(username, 24)
        return "user-" + hashlib.sha256((self._salt + s).encode()).hexdigest()[:6]

    # ---------- events ----------
    def event(self, etype: str, **fields):
        try:
            ev = {"t": _now_iso(), "ts": int(time.time()), "type": str(etype)[:32]}
            for k, v in fields.items():
                if any(b in k.lower() for b in _BAD_KEYS): continue
                if isinstance(v, bool) or v is None: ev[k] = v
                elif isinstance(v, int): ev[k] = v
                elif isinstance(v, float): ev[k] = round(v, 5) if v == v and abs(v) != float("inf") else None
                else: ev[k] = _scrub(v)
            line = json.dumps(ev, ensure_ascii=False)
            with self._lock:
                self._events.append(ev)
                try:
                    if self.events_file.exists() and self.events_file.stat().st_size > self.rotate_bytes:
                        os.replace(self.events_file, self.events_file.with_suffix(".jsonl.1"))
                    with open(self.events_file, "a", encoding="utf-8") as f: f.write(line + "\n")
                except Exception: pass
        except Exception:
            pass   # logging must NEVER break training

    def tail(self, limit: int = 200, etype: str = "", since: int = 0) -> list:
        limit = max(1, min(int(limit or 200), 2000))
        with self._lock:
            ev = [e for e in self._events
                  if (not etype or e.get("type") == etype) and e.get("ts", 0) > since]
        return ev[-limit:]

    # ---------- loss history ----------
    def note_pool_losses(self, rnd: int, pool_id: str, samples: list, step: int):
        """samples = [(loss, tokens), ...] of the deltas that SURVIVED the cross-check in one pool."""
        good = [(float(l), max(1, int(t or 1))) for l, t in samples
                if l is not None and l == l and 0 < float(l) < 1e6]
        if not good: return
        tok = sum(t for _, t in good)
        mean = sum(l * t for l, t in good) / tok
        with self._lock:
            for l, t in good: self._round_samples.append((l, t, pool_id))
            self.pools.append({"t": _now_iso(), "ts": int(time.time()), "round": rnd, "step": step,
                               "loss": round(mean, 5), "n": len(good), "tokens": tok})

    def end_round(self, rnd: int, step: int, applied: bool, tokens_total: int = 0,
                  heldout=None, extra: dict = None):
        with self._lock:
            samples = self._round_samples; self._round_samples = []
            row = {"t": _now_iso(), "ts": int(time.time()), "round": rnd, "step": step, "applied": bool(applied)}
            if samples:
                tok = sum(t for _, t, _ in samples)
                row["loss"] = round(sum(l * t for l, t, _ in samples) / tok, 5)
                row["lossMin"] = round(min(l for l, _, _ in samples), 5)
                row["lossMax"] = round(max(l for l, _, _ in samples), 5)
                row["deltas"] = len(samples); row["tokens"] = tok
            if heldout is not None: row["heldoutLoss"] = round(float(heldout), 5)
            if extra: row.update(extra)
            if samples or applied:
                self.rounds.append(row)
                self.rounds = self.rounds[-20000:]
            return row

    # ---------- static export (for the website / HF) ----------
    def export_static(self):
        try:
            with self._lock:
                ev = list(self._events)[-1000:]
                loss = {"updatedAt": _now_iso(), "rounds": list(self.rounds), "pools": list(self.pools)}
            self._atomic_write(self.public_file, json.dumps({"updatedAt": _now_iso(), "events": ev},
                                                            ensure_ascii=False))
            self._atomic_write(self.loss_file, json.dumps(loss, ensure_ascii=False))
        except Exception:
            pass

    def hf_operations(self, temp_dir, CommitOperationAdd) -> list:
        """Copies the static files into temp_dir and returns CommitOperationAdd ops for HF."""
        ops = []
        try:
            for src in (self.public_file, self.loss_file):
                if src.exists():
                    dst = Path(temp_dir) / src.name
                    dst.write_bytes(src.read_bytes())
                    ops.append(CommitOperationAdd(path_in_repo=src.name, path_or_fileobj=str(dst)))
        except Exception:
            pass
        return ops


PUB = PublicLog(Path(__file__).resolve().parent)
