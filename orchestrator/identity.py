"""Node identity used to sign tickets and, later, round records."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class NodeIdentity:
    def __init__(self, private_key: Ed25519PrivateKey):
        self._key = private_key
        self.public_key_hex = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

    @classmethod
    def generate(cls) -> "NodeIdentity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: str | Path) -> "NodeIdentity":
        path = Path(path)
        if path.exists():
            data = path.read_bytes()
            key = serialization.load_pem_private_key(data, password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise ValueError(f"{path} is not an Ed25519 private key")
            return cls(key)
        key = Ed25519PrivateKey.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, pem)
        finally:
            os.close(fd)
        return cls(key)

    def sign(self, payload: bytes) -> str:
        return self._key.sign(payload).hex()

    def verify(self, payload: bytes, signature_hex: str, public_key_hex: Optional[str] = None) -> bool:
        try:
            pub = bytes.fromhex(public_key_hex or self.public_key_hex)
            Ed25519PublicKey.from_public_bytes(pub).verify(bytes.fromhex(signature_hex), payload)
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False


__all__ = ["NodeIdentity"]
