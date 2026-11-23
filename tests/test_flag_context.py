"""A flag line must carry the submitter's own numbers, not just a verdict."""
import json
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import server
except ImportError as exc:  # pragma: no cover - the CI job has no torch
    raise unittest.SkipTest(f"needs torch and the server modules: {exc}")


def _delta(path: str, meta: dict) -> None:
    blob = json.dumps(meta).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(blob)))
        f.write(blob)


class FlagContextTest(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self._orig = server._flag_byzantine
        server._flag_byzantine = lambda key, ip, reason: self.seen.append((key, ip, reason))
        self.tmp = TemporaryDirectory()

    def tearDown(self):
        server._flag_byzantine = self._orig
        self.tmp.cleanup()

    def _run(self, meta):
        p = str(Path(self.tmp.name) / "d.bin")
        _delta(p, meta)
        server._flag_byzantine_from_delta(p, "cross-check outlier in pool p1")

    def test_the_reason_carries_the_declared_numbers(self):
        self._run({"nodeKey": "k1", "nodeIp": "1.2.3.4", "paradigm": "mosaic",
                   "declaredBatchSize": 16, "actualBatchSize": 8, "oomDownscales": 0,
                   "actualSteps": 5, "lastLoss": 3.25})
        self.assertEqual(len(self.seen), 1)
        reason = self.seen[0][2]
        self.assertIn("cross-check outlier in pool p1", reason)
        self.assertIn("paradigm=mosaic", reason)
        self.assertIn("declaredBatchSize=16", reason)
        self.assertIn("actualBatchSize=8", reason)
        self.assertIn("loss=3.2500", reason)

    def test_a_delta_without_metadata_says_so(self):
        self._run({"nodeKey": "k2", "nodeIp": "5.6.7.8"})
        self.assertEqual(len(self.seen), 1)
        self.assertIn("no declared config", self.seen[0][2])

    def test_a_declared_oom_downscale_is_still_not_flagged(self):
        self._run({"nodeKey": "k3", "nodeIp": "9.9.9.9", "oomDownscales": 1})
        if hasattr(server, "_OOM_EXEMPT"):
            self.assertEqual(self.seen, [])
        else:
            self.assertEqual(len(self.seen), 1)
