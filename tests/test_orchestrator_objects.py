import time
from pathlib import Path

import pytest

from orchestrator.objects import LocalObjectStore


def store(tmp_path):
    return LocalObjectStore(tmp_path / "objects", secret=b"s" * 32,
                            url_base="https://objects.example.test/put")


def test_grant_is_scoped_and_expires(tmp_path):
    s = store(tmp_path)
    sha = "a" * 64
    grant = s.grant_put("deltas/r1/a.bin", sha, expires_s=60)
    assert grant.url.startswith("https://objects.example.test/put/deltas/r1/a.bin?")
    assert s.verify("deltas/r1/a.bin", int(grant.expires_at), sha,
                    grant.url.split("sig=")[1]) is True
    assert s.verify("deltas/r1/other.bin", int(grant.expires_at), sha,
                    grant.url.split("sig=")[1]) is False
    assert s.verify("deltas/r1/a.bin", int(time.time()) - 1, sha,
                    grant.url.split("sig=")[1]) is False


def test_put_verifies_hash_and_rejects_traversal(tmp_path):
    s = store(tmp_path)
    payload = b"delta-bytes"
    import hashlib
    sha = hashlib.sha256(payload).hexdigest()
    assert s.put("deltas/r1/a.bin", payload, expected_sha256=sha)["size"] == len(payload)
    assert s.verify_file("deltas/r1/a.bin", sha) is True
    with pytest.raises(ValueError):
        s.put("deltas/r1/b.bin", payload, expected_sha256="b" * 64)
    with pytest.raises(ValueError):
        s.grant_put("../escape.bin", sha)
    assert s.path_for("deltas/r1/a.bin").read_bytes() == payload
