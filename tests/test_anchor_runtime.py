"""Tests for the deployment glue: a real anchor step, and the env wiring.

The stdlib-only CI job runs this file too, so every third-party import happens
inside a function or behind the skip guard at the top.
"""
import os
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anchor_runtime  # noqa: E402

try:                                   # the repo's CI job installs nothing
    import torch
    import torch.nn as nn
except ImportError as _missing:         # pragma: no cover - environment dependent
    torch = None
    nn = None
    _torch_missing = _missing
else:
    _torch_missing = None


class BatchCyclerTest(unittest.TestCase):
    def test_it_cycles_forever_and_counts_what_it_served(self):
        cycler = anchor_runtime.BatchCycler(["a", "b"])
        self.assertEqual([cycler.next() for _ in range(5)], ["a", "b", "a", "b", "a"])
        self.assertEqual(cycler.served, 5)
        self.assertEqual(len(cycler), 2)

    def test_no_batches_is_an_error_not_an_empty_loop(self):
        with self.assertRaises(ValueError):
            anchor_runtime.BatchCycler([])


class WireFromEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = {}
        self.registered = []
        module = types.ModuleType("fake_anchor_deployment")

        def make_step():
            return lambda: 1.25

        module.make_step = make_step
        sys.modules["fake_anchor_deployment"] = module

    def tearDown(self):
        sys.modules.pop("fake_anchor_deployment", None)

    def _register(self, step, *, every=0):
        self.registered.append((step, every))
        return {"enabled": step is not None, "every": every, "level": "ok"}

    def test_nothing_configured_registers_nothing(self):
        self.assertIsNone(anchor_runtime.wire_from_env(self._register, env=self.env))
        self.assertEqual(self.registered, [])

    def test_a_factory_is_imported_and_registered_with_the_cadence(self):
        self.env.update(CROWDGPT_ANCHOR_FACTORY="fake_anchor_deployment:make_step",
                        CROWDGPT_ANCHOR_EVERY="3")
        state = anchor_runtime.wire_from_env(self._register, env=self.env)
        self.assertEqual(state["every"], 3)
        (step, every), = self.registered
        self.assertEqual(every, 3)
        self.assertEqual(step(), 1.25)

    def test_a_malformed_spec_fails_loudly(self):
        self.env["CROWDGPT_ANCHOR_FACTORY"] = "no_colon_here"
        with self.assertRaises(ValueError):
            anchor_runtime.wire_from_env(self._register, env=self.env)

    def test_a_factory_returning_a_non_callable_fails_loudly(self):
        self.env["CROWDGPT_ANCHOR_FACTORY"] = "fake_anchor_deployment:make_step"
        sys.modules["fake_anchor_deployment"].make_step = lambda: 17
        with self.assertRaises(TypeError):
            anchor_runtime.wire_from_env(self._register, env=self.env)


@unittest.skipIf(torch is None, f"needs torch: {_torch_missing}")
class MakeAnchorStepTest(unittest.TestCase):
    """The reference step, exercised on a real (tiny) tied-readout model."""

    def _fixture(self):
        torch.manual_seed(0)
        vocab, dim, ctx, batch = 32, 8, 4, 2
        flat = torch.zeros(vocab * dim + dim * vocab, dtype=torch.float32)

        class Tiny(nn.Module):
            """Shaped like the shipped model: an embedding feeding a readout."""

            def __init__(self):
                super().__init__()
                self.wte = nn.Embedding(vocab, dim)
                self.head = nn.Linear(dim, vocab, bias=False)

            def forward(self, idx):
                return self.head(self.wte(idx))

        model = Tiny()
        names = [n for n, _ in model.named_parameters()]

        def sync_out(m):
            off = 0
            for n, p in m.named_parameters():
                flat[off:off + p.numel()] = p.detach().reshape(-1)
                off += p.numel()
            assert off == flat.numel()

        def sync_in(m):
            off = 0
            with torch.no_grad():
                for n, p in m.named_parameters():
                    p.copy_(flat[off:off + p.numel()].view_as(p))
                    off += p.numel()

        sync_out(model)                     # seed the buffer from the init
        gen = torch.Generator().manual_seed(1)
        batches = [(torch.randint(0, vocab, (batch, ctx), generator=gen),
                    torch.randint(0, vocab, (batch, ctx), generator=gen))
                   for _ in range(2)]
        return model, flat, names, sync_in, sync_out, batches

    def test_the_step_descends_and_writes_the_weights_back(self):
        model, flat, _, sync_in, sync_out, batches = self._fixture()
        step = anchor_runtime.make_anchor_step(model=model, batches=batches,
                                               sync_in=sync_in, sync_out=sync_out,
                                               lr=1e-2)
        before = flat.clone()
        losses = [step() for _ in range(5)]
        self.assertTrue(all(loss == loss for loss in losses), "loss must be finite")
        self.assertLess(losses[-1], losses[0], "five steps on one batch must descend")
        self.assertFalse(torch.equal(before, flat), "the step must write back")
        self.assertEqual(step.describe()["batches"], 2)

    def test_every_step_reloads_the_aggregators_weights(self):
        """A stale model would train something the round already replaced."""
        model, flat, _, sync_in, sync_out, batches = self._fixture()
        pulls = []
        counted_in = lambda m: (pulls.append(1), sync_in(m))[1]
        step = anchor_runtime.make_anchor_step(model=model, batches=batches,
                                               sync_in=counted_in, sync_out=sync_out,
                                               lr=1e-2)
        for _ in range(3):
            step()
            flat += 0.001                  # the aggregator moved between anchors
        self.assertEqual(len(pulls), 3)

    def test_a_model_with_nothing_to_train_is_rejected(self):
        model, flat, _, sync_in, sync_out, batches = self._fixture()
        for p in model.parameters():
            p.requires_grad_(False)
        with self.assertRaises(ValueError):
            anchor_runtime.make_anchor_step(model=model, batches=batches,
                                            sync_in=sync_in, sync_out=sync_out)


if __name__ == "__main__":
    unittest.main()
