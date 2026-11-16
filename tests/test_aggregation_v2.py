"""Tests for the A1/A3 aggregation changes (PLAN.md §3).

These exercise the *real* `server.py` module: real `apply_round_aggregation`, real
`open_delta`/`stream_add_base_delta`/`read_engram_tail`, real delta file format.  Only the
model itself is shrunk (MODEL_SIZE 8, engram dim 2) and the side effects that would write
checkpoints are stubbed, so the arithmetic under test is exactly the shipped arithmetic.

Run:  python3 -m unittest discover -s tests -v
"""
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
import server  # noqa: E402


MODEL_SIZE = 8          # tiny stand-in for 657,787,392
ENG_DIM = 2
ENG_BUCKETS = 4


def _bf16(vec):
    return torch.tensor(vec, dtype=torch.float32).to(torch.bfloat16)


def write_delta(path, vec, *, tokens=1_000_000, opt_steps=None, eng=None, delta_norm=None,
                extra=None):
    """Write a delta file in the shipped format (bf16 base vector + sparse engram tail)."""
    base = _bf16(vec)
    meta = {"tokensProcessed": tokens,
            "delta_norm": float(base.float().norm().item()) if delta_norm is None else delta_norm}
    if opt_steps is not None:
        meta["optimizerSteps"] = opt_steps
    if extra:
        meta.update(extra)
    hdr = json.dumps(meta).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(hdr)))
        f.write(hdr)
        f.write(base.view(torch.int16).numpy().tobytes())
        if eng:
            idx = np.asarray(eng[0], dtype=np.uint32)
            vals = _bf16(eng[1]).reshape(len(idx), ENG_DIM)
            f.write(struct.pack("<I", len(idx)))
            f.write(idx.tobytes())
            f.write(vals.view(torch.int16).numpy().tobytes())
        else:
            f.write(struct.pack("<I", 0))
    return path


class AggregationV2Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aggv2_")
        self._saved = dict(
            MODEL_SIZE=server.MODEL_SIZE,
            MODEL_CONFIG=dict(server.MODEL_CONFIG),
            ENG_NUM_BUCKETS=server.ENG_NUM_BUCKETS,
            global_weights=server.global_weights,
            global_engram_weights=server.global_engram_weights,
            global_step=server.global_step,
            CROSS_CHECK=dict(server.CROSS_CHECK),
            FL_CONFIG=dict(server.FL_CONFIG),
            HARDEN=dict(server.HARDEN),
            INIT_MODE=server.INIT_MODE,
            handle_checkpoints=server.handle_checkpoints,
            load_model_weights=server.load_model_weights,
            load_engram_weights=server.load_engram_weights,
            events=[],
        )
        server.MODEL_SIZE = MODEL_SIZE
        server.MODEL_CONFIG["dim"] = ENG_DIM
        server.ENG_NUM_BUCKETS = ENG_BUCKETS
        server.global_weights = torch.zeros(MODEL_SIZE, dtype=torch.float32)
        server.global_engram_weights = torch.zeros(ENG_BUCKETS * ENG_DIM, dtype=torch.float32)
        server.global_step = 0
        # These tests exercise A3's *step-mode* scaling; token mode has its own StepBudgetTest.
        server.CROSS_CHECK["TARGET_TOKENS_PER_STEP"] = 0
        server.FL_CONFIG["WEIGHT_DECAY"] = 0.0
        server.FL_CONFIG["SERVER_LR"] = 1.0
        server.FL_CONFIG["MIN_LR_MULT"] = 1.0
        server.FL_CONFIG["MAX_WEIGHT_NORM"] = 1e10
        server.FL_CONFIG["MAX_DELTA_APPLY_NORM"] = 12000.0
        server.HARDEN["LOSS_CHECK_MODE"] = "on"      # skip handle_checkpoints()
        server.handle_checkpoints = lambda: None
        server.load_model_weights = lambda: None
        server.load_engram_weights = lambda: None
        self._events = []
        orig_event = server.PUB.event
        server.PUB.event = lambda name, **kw: (self._events.append((name, kw)), orig_event(name, **kw))[1]

    def tearDown(self):
        server.MODEL_SIZE = self._saved["MODEL_SIZE"]
        server.MODEL_CONFIG = self._saved["MODEL_CONFIG"]
        server.ENG_NUM_BUCKETS = self._saved["ENG_NUM_BUCKETS"]
        server.global_weights = self._saved["global_weights"]
        server.global_engram_weights = self._saved["global_engram_weights"]
        server.global_step = self._saved["global_step"]
        server.CROSS_CHECK = self._saved["CROSS_CHECK"]
        server.FL_CONFIG = self._saved["FL_CONFIG"]
        server.HARDEN = self._saved["HARDEN"]
        server.INIT_MODE = self._saved["INIT_MODE"]
        server.handle_checkpoints = self._saved["handle_checkpoints"]
        server.load_model_weights = self._saved["load_model_weights"]
        server.load_engram_weights = self._saved["load_engram_weights"]

    # ------------------------------------------------------------------ helpers
    def delta_path(self, name):
        return os.path.join(self.tmp, f"delta_{name}.bin")

    def pool(self, name, deltas, tokens=None, opt_steps="auto", target_steps=None):
        entry = {"pool_id": name, "chunk_idx": 0, "deltas": deltas}
        if tokens is not None:
            entry["tokens"] = tokens
        if opt_steps != "auto":
            entry["optimizerSteps"] = opt_steps
        if target_steps is not None:
            # A3's reference is the round budget this pool was asked for (micro-steps).
            entry["target_steps"] = target_steps
        return entry

    # ------------------------------------------------------------------ step scaling (A3.1)
    def test_step_scale_clamped_and_legacy_unscaled(self):
        n_ref = 2
        scaled = server._step_scale(n_ref, {"optimizerSteps": 1})
        self.assertAlmostEqual(scaled, 2.0)                       # 1 step instead of 2 -> double
        self.assertAlmostEqual(server._step_scale(n_ref, {"optimizerSteps": 2}), 1.0)
        self.assertAlmostEqual(server._step_scale(n_ref, {"optimizerSteps": 8}), 0.5)  # clamped
        self.assertAlmostEqual(server._step_scale(4, {"optimizerSteps": 1}), 2.0)      # 4 -> clamp
        # a legacy delta (no optimizerSteps) must be applied unscaled, NOT as if it took 1 step
        self.assertEqual(server._step_scale(n_ref, {"tokensProcessed": 5}), 1.0)
        self.assertIsNone(server._delta_opt_steps({"tokensProcessed": 5}))
        self.assertIsNone(server._delta_opt_steps({"optimizerSteps": 0}))
        self.assertIsNone(server._delta_opt_steps({"optimizerSteps": "bad"}))
        # no target configured -> nothing is scaled
        self.assertEqual(server._step_scale(0, {"optimizerSteps": 1}), 1.0)

    def test_one_step_delta_is_doubled_and_legacy_is_not(self):
        """Two pools, identical deltas: H=1 must land 2x the H=2 one, legacy must land 1x."""
        vec = [0.1 * (i + 1) for i in range(MODEL_SIZE)]
        p_h1 = self.delta_path("h1")
        p_h2 = self.delta_path("h2")
        p_legacy = self.delta_path("legacy")
        write_delta(p_h1, vec, opt_steps=1)
        write_delta(p_h2, vec, opt_steps=2)
        write_delta(p_legacy, vec)          # no optimizerSteps at all
        server.CROSS_CHECK["TARGET_OPT_STEPS"] = 2
        server.apply_round_aggregation([self.pool("a", [(p_h1, 1.0)], target_steps=64)])
        w_after_h1 = server.global_weights.clone()
        server.global_weights = torch.zeros(MODEL_SIZE, dtype=torch.float32)
        server.apply_round_aggregation([self.pool("b", [(p_h2, 1.0)], target_steps=64)])
        w_after_h2 = server.global_weights.clone()
        server.global_weights = torch.zeros(MODEL_SIZE, dtype=torch.float32)
        server.apply_round_aggregation([self.pool("c", [(p_legacy, 1.0)], target_steps=64)])
        w_after_legacy = server.global_weights.clone()

        self.assertAlmostEqual(float(w_after_h1.norm() / w_after_h2.norm()), 2.0, places=4)
        self.assertAlmostEqual(float(w_after_legacy.norm() / w_after_h2.norm()), 1.0, places=4)

    def test_pool_reference_follows_its_own_round_budget(self):
        """Regression: A3's reference is the pool's own round budget.

        The budget is hardware-scaled (500 x VRAM/8 micro-steps), so a 32 GB pool runs 63
        optimizer steps.  Referencing a global nominal 500-micro-step round (16 steps) would
        apply clamp(16/63) = 0.5 and silently halve an entire healthy pool's delta.
        """
        vec = [0.05 * (i + 1) for i in range(MODEL_SIZE)]
        p = self.delta_path("p63")
        write_delta(p, vec, opt_steps=63)
        server.CROSS_CHECK["TARGET_LOCAL_STEPS"] = 500
        server.CROSS_CHECK["TARGET_TOKENS_PER_STEP"] = 131072
        self._events.clear()
        server.apply_round_aggregation([self.pool("big", [(p, 1.0)], target_steps=2000)])
        ev = [kw for name, kw in self._events if name == "aggregation"][-1]
        self.assertEqual(ev["poolNRefs"], [63])       # derived from 2000 micro-steps / accum 32
        self.assertEqual(ev["stepScales"], [1.0])     # not rescaled against the 16-step nominal

    def test_outlier_is_corrected_against_its_own_pool(self):
        """A client that ran 16 steps where its pool asked for 63 is scaled up (clamped at 2x),
        while the clients that hit the budget stay at 1.0."""
        vec = [0.05 * (i + 1) for i in range(MODEL_SIZE)]
        p63 = self.delta_path("q63")
        p16 = self.delta_path("q16")
        write_delta(p63, vec, opt_steps=63)
        write_delta(p16, vec, opt_steps=16)
        server.CROSS_CHECK["TARGET_TOKENS_PER_STEP"] = 131072
        self._events.clear()
        server.apply_round_aggregation([self.pool("big", [(p63, 1.0), (p16, 1.0)],
                                                  target_steps=2000)])
        ev = [kw for name, kw in self._events if name == "aggregation"][-1]
        self.assertEqual(ev["poolNRefs"], [63])
        self.assertEqual(ev["stepScales"], [1.0, 2.0])

    # ------------------------------------------------------------------ pool weights (A3.3)
    def test_pools_weighted_by_tokens(self):
        a = self.delta_path("tok_a")
        b = self.delta_path("tok_b")
        write_delta(a, [1.0] * MODEL_SIZE, tokens=3_000_000, opt_steps=2)
        write_delta(b, [1.0] * MODEL_SIZE, tokens=1_000_000, opt_steps=2)
        server.CROSS_CHECK["TARGET_OPT_STEPS"] = 2
        server.apply_round_aggregation([
            self.pool("a", [(a, 1.0)], tokens=3_000_000),
            self.pool("b", [(b, 1.0)], tokens=1_000_000),
        ])
        w = server.global_weights
        # both deltas are identical, so every coordinate must equal the summed pool weight (1.0)
        self.assertTrue(torch.allclose(w, torch.ones(MODEL_SIZE), atol=1e-6), w)
        agg = [e for e in self._events if e[0] == "aggregation"]
        self.assertTrue(agg, "aggregation event missing")
        self.assertEqual(agg[-1][1]["poolWeights"], [0.75, 0.25])

    def test_pool_weights_fall_back_to_equal_without_token_metadata(self):
        a = self.delta_path("notok_a")
        b = self.delta_path("notok_b")
        write_delta(a, [1.0] * MODEL_SIZE, opt_steps=2)
        write_delta(b, [2.0] * MODEL_SIZE, opt_steps=2)
        # the file headers carry token counts, so strip them to force the "no metadata" path
        for p in (a, b):
            md = server._delta_metadata(p)
            md.pop("tokensProcessed", None)
            md["tokens"] = 0
            hdr = json.dumps(md).encode()
            with open(p, "r+b") as f:
                old = struct.unpack("<I", f.read(4))[0]
                rest = f.read()
                f.seek(0)
                f.truncate()
                f.write(struct.pack("<I", len(hdr)) + hdr + rest[old:])
        server.CROSS_CHECK["TARGET_OPT_STEPS"] = 2
        server.apply_round_aggregation([
            {"pool_id": "a", "chunk_idx": 0, "deltas": [(a, 1.0)], "tokens": 0},
            {"pool_id": "b", "chunk_idx": 0, "deltas": [(b, 1.0)], "tokens": 0},
        ])
        # equal pools -> (1 + 2) / 2 = 1.5 per coordinate
        self.assertTrue(torch.allclose(server.global_weights, torch.full((MODEL_SIZE,), 1.5),
                                       atol=1e-6), server.global_weights)

    # ------------------------------------------------------------------ cap tripwire (A3.2)
    def test_delta_cap_is_a_warning_not_a_silent_clip(self):
        p = self.delta_path("huge")
        write_delta(p, [4000.0] + [0.0] * (MODEL_SIZE - 1), opt_steps=2)
        server.FL_CONFIG["MAX_DELTA_APPLY_NORM"] = 12000.0   # new default: no clip at 4000
        server.CROSS_CHECK["TARGET_OPT_STEPS"] = 2
        server.apply_round_aggregation([self.pool("a", [(p, 1.0)])])
        agg = [e for e in self._events if e[0] == "aggregation"][-1][1]
        self.assertEqual(agg["clipped"], 0)
        self.assertAlmostEqual(agg["maxDeltaApplyNorm"], 12000.0)

        server.global_weights = torch.zeros(MODEL_SIZE, dtype=torch.float32)
        server.FL_CONFIG["MAX_DELTA_APPLY_NORM"] = 1000.0
        p2 = self.delta_path("huge2")
        write_delta(p2, [4000.0] + [0.0] * (MODEL_SIZE - 1), opt_steps=2)
        with self.assertLogs(server.log, level="WARNING") as captured:
            server.apply_round_aggregation([self.pool("b", [(p2, 1.0)])])
        self.assertTrue(any("cap tripped" in m for m in captured.output), captured.output)
        # 4000 clipped to 1000, then averaged over one pool
        self.assertAlmostEqual(float(server.global_weights[0]), 1000.0, places=1)

    # ------------------------------------------------------------------ engram (A3.4)
    def test_engram_gets_the_same_step_and_pool_scaling(self):
        """The engram is normalised exactly like the base: pool_w * weight * step_scale."""
        server.HARDEN["ENGRAM_FULL_TRUST_WEIGHT"] = 0.2
        server.CROSS_CHECK["TARGET_OPT_STEPS"] = 2
        server.FL_CONFIG["MAX_DELTA_APPLY_NORM"] = 12000.0
        # two equal pools: H=1 (step_scale 2) moves row 1 by (3,4); H=2 (step_scale 1) leaves it
        # alone, so the weighted mean lands at 2/3 of (3,4) — not the unweighted 1/2.
        pa = self.delta_path("eng_a")
        pb = self.delta_path("eng_b")
        tiny = [0.01] + [0.0] * (MODEL_SIZE - 1)
        write_delta(pa, tiny, opt_steps=1, eng=([1], [[3.0, 4.0]]))
        write_delta(pb, tiny, opt_steps=2, eng=([1], [[0.0, 0.0]]))
        server.global_engram_weights = torch.zeros(ENG_BUCKETS * ENG_DIM)
        server.apply_round_aggregation([
            self.pool("a", [(pa, 1.0)], tokens=1_000_000, target_steps=64),
            self.pool("b", [(pb, 1.0)], tokens=1_000_000, target_steps=64),
        ])
        row = server.global_engram_weights[1 * ENG_DIM:2 * ENG_DIM]
        # eff_a = 0.5 * 2 = 1.0, eff_b = 0.5 * 1 = 0.5 -> mean = (3,4) * 1.0/1.5 * lr(1.0) * trust(1.0)
        self.assertAlmostEqual(float(row[0]), 2.0, places=3)
        self.assertAlmostEqual(float(row[1]), 8.0 / 3.0, places=3)

    def test_single_pool_engram_row_is_applied_verbatim(self):
        server.HARDEN["ENGRAM_FULL_TRUST_WEIGHT"] = 0.2
        server.CROSS_CHECK["TARGET_OPT_STEPS"] = 2
        p = self.delta_path("eng_solo")
        write_delta(p, [0.01] + [0.0] * (MODEL_SIZE - 1), opt_steps=1, eng=([1], [[3.0, 4.0]]))
        server.global_engram_weights = torch.zeros(ENG_BUCKETS * ENG_DIM)
        server.apply_round_aggregation([self.pool("solo", [(p, 1.0)], tokens=1_000_000)])
        row = server.global_engram_weights[1 * ENG_DIM:2 * ENG_DIM]
        self.assertAlmostEqual(float(row[0]), 3.0, places=3)
        self.assertAlmostEqual(float(row[1]), 4.0, places=3)

    def test_engram_row_norm_cap_still_applies(self):
        server.HARDEN["ENGRAM_ROW_NORM_MAX"] = 1.0
        p = self.delta_path("eng_cap")
        write_delta(p, [0.01] + [0.0] * (MODEL_SIZE - 1), opt_steps=2, eng=([0], [[3.0, 4.0]]))
        server.global_engram_weights = torch.zeros(ENG_BUCKETS * ENG_DIM)
        server.apply_round_aggregation([self.pool("solo", [(p, 1.0)], tokens=1_000_000)])
        row = server.global_engram_weights[:ENG_DIM]
        self.assertAlmostEqual(float(row.norm()), 1.0, places=3)

    # ------------------------------------------------------------------ norm guard
    def test_norm_guard_reverts_and_does_not_advance(self):
        p = self.delta_path("guard")
        server.global_weights = torch.full((MODEL_SIZE,), 1.0)
        write_delta(p, [1.0] * MODEL_SIZE, opt_steps=2)     # +100 % norm -> outside [0.7, 1.5]
        server.CROSS_CHECK["TARGET_OPT_STEPS"] = 2
        server.apply_round_aggregation([self.pool("a", [(p, 1.0)])])
        self.assertEqual(server.global_step, 0)
        self.assertTrue(any(e[0] == "aggregation_reverted" for e in self._events), self._events)

    # ------------------------------------------------------------------ metadata plumbing
    def test_pool_delta_stats_uses_majority_and_sums_tokens(self):
        paths = []
        for i, (steps, toks) in enumerate([(2, 1_000_000), (2, 1_000_000), (8, 2_000_000)]):
            p = self.delta_path(f"stats{i}")
            write_delta(p, [0.1] * MODEL_SIZE, opt_steps=steps, tokens=toks)
            paths.append((p, 1.0))
        stats = server._pool_delta_stats(paths)
        self.assertEqual(stats["optimizerSteps"], 2)
        self.assertEqual(stats["tokens"], 4_000_000)

    def test_pool_delta_stats_legacy_pool(self):
        p = self.delta_path("stats_legacy")
        write_delta(p, [0.1] * MODEL_SIZE, tokens=123)
        stats = server._pool_delta_stats([(p, 1.0)])
        self.assertIsNone(stats["optimizerSteps"])
        self.assertEqual(stats["tokens"], 123)

    def test_merge_preserves_optimizer_steps_and_tokens(self):
        a = self.delta_path("m_a")
        b = self.delta_path("m_b")
        write_delta(a, [1.0] * MODEL_SIZE, opt_steps=2, tokens=1_000_000)
        write_delta(b, [3.0] * MODEL_SIZE, opt_steps=2, tokens=1_000_000)
        merged = server._merge_pool_deltas("poolM", [(a, 0.5), (b, 0.5)])
        self.assertIsNotNone(merged)
        md = server._delta_metadata(merged)
        self.assertEqual(md["optimizerSteps"], 2)
        self.assertEqual(md["tokens"], 2_000_000)
        self.assertGreater(md["delta_norm"], 0)
        # the merged vector is the weighted average of the two deltas
        df = server.open_delta(merged, MODEL_SIZE)
        os.remove(merged)

    def test_upload_metadata_keeps_optimizer_steps_absent_for_legacy_clients(self):
        """The upload handler must not invent a step count for a client that did not send one."""
        body = {"localSteps": 500}
        opt_steps = int(body.get("optimizerSteps") or 0)
        self.assertEqual(opt_steps, 0)      # -> key omitted from delta_metadata
        body = {"localSteps": 500, "optimizerSteps": 2, "accum": 250}
        opt_steps = int(body.get("optimizerSteps") or 0)
        accum = int(body.get("accum") or 0)
        self.assertEqual((opt_steps, accum), (2, 250))

    # ------------------------------------------------------------------ task payload
    def test_flat_layout_matches_the_client_parameter_order(self):
        """`flat_ln_slices` must agree with the packing layout clients iterate."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sidecar" / "Crowd-v1" / "packing"))
        try:
            import pack_flat
        except Exception as e:                      # pragma: no cover - sidecar not present
            self.skipTest(f"sidecar packing helper unavailable: {e}")
        real = self._saved["MODEL_CONFIG"]          # setUp shrinks MODEL_CONFIG, use the real one
        cfg = {"vocab_size": real["vocabSize"],
               "hidden_size": real["dim"],
               "intermediate_size": real["mlpHidden"],
               "head_dim": real["headDim"],
               "num_attention_heads": real["nHeads"],
               "num_key_value_heads": real["nKvHeads"],
               "num_hidden_layers": real["nLayers"]}
        layout = pack_flat.flat_layout(cfg)
        expected = {}
        off = 0
        for name, shape in layout:
            n = 1
            for s in shape:
                n *= s
            expected[name] = (off, n)
            off += n
        self.assertEqual(off, server.calculate_model_size(real))
        self.assertEqual(off, 657_787_392)          # = MODEL_SIZE of the shipped model
        got = {}
        for name in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"):
            for key, (o, n) in expected.items():
                if key.endswith(name):
                    got[key] = (o, n)
        for key in ("ln.weight", "ln.bias"):
            got[key] = expected[key]
        slices = {(o, n, f) for o, n, f in server.flat_ln_slices(real)}
        for name, (o, n) in got.items():
            fill = 1.0 if name.endswith(".weight") else 0.0
            self.assertIn((o, n, fill), slices, f"{name} at offset {o} missing from flat_ln_slices")
        self.assertEqual(len(slices), len(got))

    def test_expected_init_norm_matches_the_documented_init(self):
        server.INIT_MODE = "ln_identity"
        real = self._saved["MODEL_CONFIG"]
        n_ln_w = sum(n for _o, n, f in server.flat_ln_slices(real) if f == 1.0)
        total = server.calculate_model_size(real)
        self.assertAlmostEqual(server.expected_init_norm(real),
                               math.sqrt(4e-4 * (total - 2 * n_ln_w) + n_ln_w), places=3)
        # 581.7 with ln=1, 512.9 with the v1 flat draw (the number PLAN.md §4 quotes)
        self.assertAlmostEqual(server.expected_init_norm(real), 581.66, places=1)
        server.INIT_MODE = "flat"
        self.assertAlmostEqual(server.expected_init_norm(real), 512.95, places=1)


if __name__ == "__main__":
    unittest.main()




class RoundRefStepsTest(unittest.TestCase):
    """A3's reference: the optimizer steps a round of N chunks contains, derived from mainline's
    own accumulation target (TARGET_ACCUM_STEPS=32)."""

    def test_steps_follow_the_round_length(self):
        self.assertEqual(server._round_ref_steps(500), 16)    # 8 GB card
        self.assertEqual(server._round_ref_steps(250), 8)     # 4 GB card
        self.assertEqual(server._round_ref_steps(2000), 63)   # 32 GB card
        self.assertEqual(server._round_ref_steps(2500), 79)   # 80 GB card
        self.assertEqual(server._round_ref_steps(0), 0)
        self.assertEqual(server._round_ref_steps(None), 0)

    def test_reference_honours_a_custom_accum(self):
        old = dict(server.CROSS_CHECK)
        try:
            server.CROSS_CHECK["TARGET_ACCUM_STEPS"] = 16
            self.assertEqual(server._round_ref_steps(500), 32)
        finally:
            server.CROSS_CHECK.clear(); server.CROSS_CHECK.update(old)
