"""Presigned object storage.

The orchestrator issues a short-lived, single-object PUT grant.  Bulk bytes go
to the store, not through the control plane.  ``LocalObjectStore`` is the
development backend and the reference for the token format; an S3/R2 backend
only has to return the same grant shape and verify the same token.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode


@dataclass(frozen=True)
class UploadGrant:
    key: str
    url: str
    method: str
    headers: dict
    expires_at: float
    sha256: str

    def as_dict(self) -> dict:
        return {"key": self.key, "url": self.url, "method": self.method,
                "headers": dict(self.headers), "expiresAt": self.expires_at,
                "sha256": self.sha256}


class LocalObjectStore:
    def __init__(self, root: str | Path, *, secret: bytes,
                 url_base: str = "/fl/object"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.secret = secret
        self.url_base = url_base.rstrip("/")

    @staticmethod
    def _safe_key(key: str) -> str:
        key = key.strip().lstrip("/")
        if not key or ".." in Path(key).parts or key.startswith("/"):
            raise ValueError("invalid object key")
        return key

    def _mac(self, key: str, expires_at: int, sha256: str) -> str:
        msg = f"{key}\n{expires_at}\n{sha256}".encode("utf-8")
        return hmac.new(self.secret, msg, hashlib.sha256).hexdigest()

    def grant_put(self, key: str, sha256: str, *, expires_s: int = 900,
                  base_url: Optional[str] = None) -> UploadGrant:
        key = self._safe_key(key)
        expires_at = int(time.time()) + int(expires_s)
        sha = sha256.lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        query = urlencode({"exp": expires_at, "sha256": sha,
                           "sig": self._mac(key, expires_at, sha)})
        base = (base_url or self.url_base).rstrip("/")
        return UploadGrant(key=key, url=f"{base}/{key}?{query}", method="PUT",
                           headers={"Content-Type": "application/octet-stream"},
                           expires_at=float(expires_at), sha256=sha)

    def verify(self, key: str, expires_at: int, sha256: str, signature: str,
               *, now: Optional[float] = None) -> bool:
        try:
            key = self._safe_key(key)
        except ValueError:
            return False
        now = time.time() if now is None else now
        if int(expires_at) < int(now):
            return False
        return hmac.compare_digest(self._mac(key, int(expires_at), sha256.lower()),
                                   str(signature))

    def path_for(self, key: str) -> Path:
        return self.root / self._safe_key(key)

    def put(self, key: str, data: bytes, *, expected_sha256: Optional[str] = None) -> dict:
        path = self.path_for(key)
        sha = hashlib.sha256(data).hexdigest()
        if expected_sha256 and not hmac.compare_digest(sha, expected_sha256.lower()):
            raise ValueError("object_hash_mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp.{secrets.token_hex(4)}")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try: tmp.unlink()
                except OSError: pass
        return {"key": key, "sha256": sha, "size": len(data)}

    def commit_file(self, key: str, source: str | Path, *,
                    expected_sha256: Optional[str] = None) -> dict:
        source = Path(source)
        sha = _sha256_file(source)
        if expected_sha256 and not hmac.compare_digest(sha, expected_sha256.lower()):
            raise ValueError("object_hash_mismatch")
        dest = self.path_for(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, dest)
        return {"key": key, "sha256": sha, "size": dest.stat().st_size}

    def verify_file(self, key: str, expected_sha256: str) -> bool:
        path = self.path_for(key)
        if not path.exists():
            return False
        return hmac.compare_digest(_sha256_file(path), expected_sha256.lower())

    def delete(self, key: str) -> None:
        try: self.path_for(key).unlink()
        except OSError: pass


def _sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block: break
            h.update(block)
    return h.hexdigest()


__all__ = ["LocalObjectStore", "UploadGrant"]
