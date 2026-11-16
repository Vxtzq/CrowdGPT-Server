"""Integration test: client-written delta -> real upload handler -> aggregation.

This drives `process_decompressed_payload` (the code path a real submission takes) with a
payload built in the *client's* wire format (gzip: json header + bf16 base + engram tail),
then feeds the resulting delta file into `apply_round_aggregation`.  It covers the parts a
unit test of either side alone cannot: that `optimizerSteps`/`accum` survive the upload
handler into the delta metadata, that the server re-frames the engram tail with a count,
and that the applied update is exactly `lr * pool_w * step_scale * delta`.

Run:  python3 -m unittest discover -s tests -v
"""
import gzip
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "CrowdGPT"))
import server  # noqa: E402

MODEL_SIZE = 8
ENG_DIM = 2
ENG_BUCKETS = 4
CLIENT_KEY = "ck-test"


def client_payload(vec, *, tokens=1_000_000, opt_steps=2, accum=250, local_steps=500,
                   eng=None, extra=None):
    """Bytes exactly as client.py's `_write_delta_file` emits them (gzip, no engram count)."""
    base = torch.tensor(vec, dtype=torch.float32).to(torch.bfloat16)
    body = {"taskId": "task-1", "loss": 3.5, "localSteps": local_steps,
            "tokensProcessed": tokens, "loraRank": 0, "isDelta": True,
            "weightFormat": "bf16", "hasEngram": bool(eng),
            "engramSparseCount": 0 if not eng else len(eng[0]),
            "deltaCodec": "off", "crossCheckUsed": True, "actualSeed": 1,
            "actualSteps": local_steps, "actualBatchSize": 2,
            "optimizerSteps": opt_steps, "accum": accum,
            "innerOptimizer": "sgd_nesterov", "innerLr": 8e-4,
            "lastLoss": 3.4, "lossStd": 0.1, "lossSamples": 500}
    if extra:
        body.update(extra)
    hdr = json.dumps(body).encode()
    out = bytearray(struct.pack("<I", len(hdr)) + hdr)
    out += base.view(torch.int16).numpy().tobytes()
    if eng:
        idx = np.asarray(eng[0], dtype=np.uint32)
        vals = torch.tensor(eng[1], dtype=torch.float32).to(torch.bfloat16)
        out += idx.tobytes()
        out += vals.view(torch.int16).numpy().tobytes()
    return gzip.compress(bytes(out))


class _Req:
    """Enough of a Starlette Request for the upload handler."""
    class _C:
        host = "127.0.0.1"
    client = _C()
    headers = {"X-Client-Id": CLIENT_KEY}
    url = type("U", (), {"path": "/fl/submit"})()


class UploadToAggregationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="upagg_")
        self._saved = {k: getattr(server, k) for k in (
            "MODEL_SIZE", "ENG_NUM_BUCKETS", "global_weights", "global_engram_weights",
            "global_step", "get_current_user", "handle_checkpoints", "load_model_weights",
            "load_engram_weights", "round_state", "db", "pools", "task_tickets",
            "client_hardware", "ORCH")}
        self._cfg = dict(server.MODEL_CONFIG)
        self._cc = dict(server.CROSS_CHECK)
        self._fl = dict(server.FL_CONFIG)
        self._hard = dict(server.HARDEN)

        server.MODEL_SIZE = MODEL_SIZE
        server.MODEL_CONFIG["dim"] = ENG_DIM
        server.ENG_NUM_BUCKETS = ENG_BUCKETS
        server.global_weights = torch.zeros(MODEL_SIZE, dtype=torch.float32)
        server.global_engram_weights = torch.zeros(ENG_BUCKETS * ENG_DIM, dtype=torch.float32)
        server.global_step = 0
        server.get_current_user = lambda req: None
        server.handle_checkpoints = lambda: None
        server.load_model_weights = lambda: None
        server.load_engram_weights = lambda: None
        server.ORCH = None
        server.FL_CONFIG["WEIGHT_DECAY"] = 0.0
        server.FL_CONFIG["SERVER_LR"] = 1.0
        server.FL_CONFIG["MIN_LR_MULT"] = 1.0
        server.FL_CONFIG["MAX_DELTA_APPLY_NORM"] = 12000.0
        server.HARDEN["LOSS_CHECK_MODE"] = "on"
        server.HARDEN["REQUIRE_TASK_TICKET"] = False
        server.HARDEN["MIN_TRAIN_TIME_S"] = 0
        server.HARDEN["MIN_SUBMIT_INTERVAL_S"] = 0
        server.CROSS_CHECK["TARGET_OPT_STEPS"] = 2
        server.round_state["current_round"] = 3
        server.round_state["is_aggregating"] = False
        server.db["tasks"]["task-1"] = {
            "poolId": "p1", "shardId": "p1", "chunkIdx": 0, "round": 3,
            "targetLocalSteps": 500, "targetBatchSize": 2, "targetSeed": 1,
            "targetAutocast": "bf16", "targetOptimizer": "any",
            "targetOptimizerSteps": 2,
        }
        server.pools.clear()
        del server.ready_pools[:]
        # two members so a single submission does not complete the pool (which would pop it and
        # start the cross-check thread); we want to inspect the stored delta here.
        server.pools["p1"] = {"pool_id": "p1", "clients": [CLIENT_KEY, "ck-other"],
                              "status": "training", "deltas": {}, "chunk_idx": 0, "round": 3}
        server.task_tickets.pop(CLIENT_KEY, None)
        server.client_hardware[CLIENT_KEY] = {"autocast": "bf16", "optim": "adamw32"}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(server, k, v)
        server.MODEL_CONFIG = self._cfg
        server.CROSS_CHECK = self._cc
        server.FL_CONFIG = self._fl
        server.HARDEN = self._hard

    def _submit(self, vec, eng=None, **kw):
        path = os.path.join(self.tmp, "payload.bin")
        with open(path, "wb") as f:
            f.write(client_payload(vec, eng=eng, **kw))
        resp = server.process_compressed_payload(path, "127.0.0.1", CLIENT_KEY, _Req(), None)
        return resp

    def test_submission_records_optimizer_steps(self):
        vec = [0.05 * (i + 1) for i in range(MODEL_SIZE)]
        resp = self._submit(vec, eng=([1], [[3.0, 4.0]]))
        self.assertNotIn("error", str(resp).lower(), resp)
        deltas = server.pools["p1"]["deltas"]
        self.assertIn(CLIENT_KEY, deltas)
        path = deltas[CLIENT_KEY]
        md = server._delta_metadata(path)
        self.assertEqual(md["optimizerSteps"], 2)
        self.assertEqual(md["accum"], 250)
        self.assertEqual(md["localSteps"], 500)
        self.assertEqual(md["tokensProcessed"], 1_000_000)
        # the server re-frames the engram tail with a count, so the reader can find it again
        df = server.open_delta(path, MODEL_SIZE)
        idx, vals = server.read_engram_tail(df, ENG_DIM)
        self.assertEqual(idx.tolist(), [1])
        self.assertTrue(torch.allclose(vals.float(), torch.tensor([[3.0, 4.0]]), atol=1e-2))
        os.remove(path)

    def test_legacy_client_without_optimizer_steps(self):
        vec = [0.05 * (i + 1) for i in range(MODEL_SIZE)]
        body_extra = {"optimizerSteps": None}
        path = os.path.join(self.tmp, "payload_legacy.bin")
        payload = client_payload(vec)
        # drop the two new keys entirely: a pre-v2 client never sends them
        hdr_len = struct.unpack("<I", gzip.decompress(payload)[:4])[0]
        raw = gzip.decompress(payload)
        hdr = json.loads(raw[4:4 + hdr_len])
        hdr.pop("optimizerSteps", None)
        hdr.pop("accum", None)
        hdr.pop("innerOptimizer", None)
        hdr.pop("innerLr", None)
        nb = json.dumps(hdr).encode()
        new = struct.pack("<I", len(nb)) + nb + raw[4 + hdr_len:]
        with open(path, "wb") as f:
            f.write(gzip.compress(new))
        server.process_compressed_payload(path, "127.0.0.1", CLIENT_KEY, _Req(), None)
        dpath = server.pools["p1"]["deltas"][CLIENT_KEY]
        md = server._delta_metadata(dpath)
        self.assertNotIn("optimizerSteps", md)     # never invented
        os.remove(dpath)

    def test_upload_then_aggregate_applies_the_step_scale(self):
        vec = [0.05 * (i + 1) for i in range(MODEL_SIZE)]
        self._submit(vec, opt_steps=1)             # H=1 with TARGET_OPT_STEPS=2 -> x2
        dpath = server.pools["p1"]["deltas"][CLIENT_KEY]
        stats = server._pool_delta_stats([(dpath, 1.0)])
        self.assertEqual(stats["optimizerSteps"], 1)
        self.assertEqual(stats["tokens"], 1_000_000)
        server.apply_round_aggregation([{"pool_id": "p1", "chunk_idx": 0,
                                         "deltas": [(dpath, 1.0)], "target_steps": 64, **stats}])
        applied = server.global_weights.clone()
        expect = torch.tensor(vec, dtype=torch.float32).to(torch.bfloat16).float() * 2.0
        self.assertTrue(torch.allclose(applied, expect, atol=2e-3), (applied, expect))

    def test_upload_that_needs_normalising_keeps_the_metadata(self):
        """A delta above MAX_DELTA_APPLY_NORM is rewritten on upload; the fields must survive."""
        big = [4000.0] + [0.0] * (MODEL_SIZE - 1)
        self._submit(big, opt_steps=2, accum=250)
        dpath = server.pools["p1"]["deltas"][CLIENT_KEY]
        md = server._delta_metadata(dpath)
        self.assertEqual(md["optimizerSteps"], 2)
        self.assertEqual(md["accum"], 250)
        os.remove(dpath)


if __name__ == "__main__":
    unittest.main()
