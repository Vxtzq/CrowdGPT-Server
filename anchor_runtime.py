"""The deployment glue for aggregator anchoring: one real torch step.

``anchor_step`` owns the *arithmetic* of when to anchor; ``server.set_anchor_step``
owns the *wiring*. This module supplies the piece in between, so a deployment can
turn anchoring on with an environment variable instead of writing the same
optimiser boilerplate:

    CROWDGPT_ANCHOR_EVERY=2
    CROWDGPT_ANCHOR_FACTORY=my_deployment.anchor:make_step

The factory is called with no arguments and must return a callable taking no
arguments, doing one gradient step of the whole model on the true terminal loss,
and returning the loss it saw. ``make_anchor_step`` below is the reference
implementation; anything with the same contract works.

Why the split matters (measured at 65 M, 488 rounds, tuned MOSAIC base): giving
the aggregator a full-model step every 2 rounds takes the held-out loss from
5.8346 to 5.6732, and every round to 5.5854, because a MOSAIC slice never sees the
terminal loss cross a cut. Every candidate that tried to fix that *inside* a slice
(head lr, input-skip head, recalibration, deeper head, overlap at the cut) was
measured and refuted.

The contract deliberately does not know about the wire format. ``sync_in`` and
``sync_out`` are the deployment's own two functions: ``sync_in(model)`` loads the
server's current global weights into ``model`` and ``sync_out(model)`` writes them
back after the step. On the client side of this repo those are
``SotaGPT.load_base_weights`` and the flat-vector writer; both walk
``named_parameters()`` in the same order, which is what makes the pair correct.
Keeping them as callables also keeps this module importable without torch, so the
standard-library CI job stays green.
"""
from __future__ import annotations

import itertools
from typing import Callable, Iterable, Sequence

DEFAULT_LR = 3e-4
DEFAULT_CLIP = 1.0


class BatchCycler:
    """Cycle a fixed list of batches, forever.

    Anchoring happens every N rounds for the whole run, so the step needs a
    deterministic, endless supply of data without re-reading a shard list each
    time. Cycling is intentional: a single anchor batch every 2 rounds would be
    over-fitted within a few thousand rounds, exactly like the head-recalibration
    arm that measured worse the harder it was applied.
    """

    def __init__(self, batches: Sequence):
        if not batches:
            raise ValueError("anchoring needs at least one batch")
        self._batches = list(batches)
        self._cycle = itertools.cycle(self._batches)
        self.served = 0

    def next(self):
        self.served += 1
        return next(self._cycle)

    def __len__(self) -> int:
        return len(self._batches)


def _default_loss(model, x, y):
    """Cross entropy over the last axis — the terminal objective, not a probe."""
    import torch.nn.functional as F

    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def make_anchor_step(*, model, batches, sync_in: Callable, sync_out: Callable,
                     lr: float = DEFAULT_LR, clip: float = DEFAULT_CLIP,
                     loss_fn: Callable | None = None,
                     betas: tuple[float, float] = (0.9, 0.95),
                     weight_decay: float = 0.01):
    """Build the step function ``server.set_anchor_step`` expects.

    ``sync_in``/``sync_out`` are called on every step, not once: the aggregator's
    weights change between anchors, and a step taken on stale weights would train
    a model that no longer exists. The optimiser is built once and lives for the
    whole run (a fresh AdamW per call is sign-SGD at one step — the same defect
    that was found and fixed inside MOSAIC's slices).
    """
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no parameters that require grad")
    opt = torch.optim.AdamW(params, lr=float(lr), betas=betas, weight_decay=weight_decay)
    cycler = BatchCycler(batches)
    loss_function = loss_fn or _default_loss

    def step() -> float:
        sync_in(model)
        x, y = cycler.next()
        opt.zero_grad(set_to_none=True)
        loss = loss_function(model, x, y)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(params, float(clip))
        opt.step()
        sync_out(model)
        return float(loss.detach())

    step.describe = lambda: {          # for the startup log / /fl/round_status
        "lr": float(lr), "clip": float(clip), "batches": len(cycler),
        "params": len(params),
    }
    return step


def wire_from_env(register: Callable, *, env=None) -> dict | None:
    """Import ``CROWDGPT_ANCHOR_FACTORY`` and register the step it returns.

    ``register`` is ``server.set_anchor_step``. Nothing is imported and nothing is
    registered when the variable is unset, so an existing deployment is unchanged.
    Returns the state dict from ``register`` (useful in logs), or ``None``.
    """
    import importlib
    import os

    env = os.environ if env is None else env
    spec = (env.get("CROWDGPT_ANCHOR_FACTORY") or "").strip()
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError(
            "CROWDGPT_ANCHOR_FACTORY must be 'module:callable', got %r" % (spec,))
    module_name, attr = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attr)
    step = factory()
    if not callable(step):
        raise TypeError("anchor factory %s returned %r, not a callable" % (spec, step))
    every = int(env.get("CROWDGPT_ANCHOR_EVERY") or 0)
    return register(step, every=every)


def describe_batches(batches: Iterable) -> dict:
    """Small helper for deployments that want to log what they wired."""
    batches = list(batches)
    return {"batches": len(batches), "firstShape": [tuple(getattr(t, "shape", ())) for t in
                                                    (batches[0] if batches else ())]}
