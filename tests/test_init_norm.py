"""G0 init gate: the norm of the initialisation must be measured accurately.

Regression this pins down: `Tensor.norm()` accumulates a 657 M-parameter buffer in
float32 and under-reports it by ~12 % (519.39 against a true 581.66), so the startup
gate compared a wrong number with the right expectation and logged
`G0 init check failed` on every launch with INIT_MODE=ln_identity.
"""
import gc
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402


class FlatNormTest(unittest.TestCase):
    def test_matches_float64_reference(self):
        torch.manual_seed(0)
        t = (torch.randn(30_000_000) * 0.02).float()
        want = float(t.double().norm())
        got = server.flat_norm(t, chunk=1_000_000)
        self.assertAlmostEqual(got / want, 1.0, places=9)

    def test_empty_buffer_is_zero(self):
        self.assertEqual(server.flat_norm(torch.zeros(0)), 0.0)


class InitGateTest(unittest.TestCase):
    """The gate the server applies at startup, on the buffer the server builds."""

    def test_gate_passes_for_both_init_modes(self):
        cfg = server.MODEL_CONFIG
        for mode in ("ln_identity", "flat"):
            with self.subTest(mode=mode):
                w = server.build_init_weights(cfg, mode=mode)
                got = server.flat_norm(w)
                want = server.expected_init_norm(cfg, mode=mode)
                self.assertLessEqual(abs(got - want), 0.5,
                                     f"G0 gate: ‖W‖={got:.2f} vs expected ≈{want:.2f} ({mode})")
                del w
                gc.collect()

    def test_ln_identity_sets_layer_norm_slices_only(self):
        cfg = server.MODEL_CONFIG
        w = server.build_init_weights(cfg, mode="ln_identity")
        slices = server.flat_ln_slices(cfg)
        self.assertEqual(slices[0][1], cfg["dim"])
        self.assertGreater(len(slices), 2 * cfg["nLayers"])
        for off, n, fill in slices:
            block = w[off:off + n]
            self.assertTrue(torch.equal(block, torch.full_like(block, fill)),
                            f"slice at {off} (+{n}) is not uniformly {fill}")
        # something outside the LayerNorm slices must still be the random draw
        embed = w[:cfg["vocabSize"] * cfg["dim"]]
        self.assertFalse(bool((embed == 1.0).any()))
        del w, slices
        gc.collect()


if __name__ == "__main__":
    unittest.main()
