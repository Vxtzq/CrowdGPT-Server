"""DiLoCo's outer step, as the aggregator needs it.

DiLoCo differs from plain federated averaging by one thing: the *outer* optimizer
has momentum across rounds. Each round a peer trains locally from the current
global weights and ships a delta; the aggregator moves the global weights by
`lr * (delta + momentum * (delta - previous_delta))` instead of by the raw mean.
Without that term the whole method degenerates into FedAvg, which is why the
server had to grow this module: `client.py` can only declare inner steps, the
momentum belongs to whoever merges.

The complication is that the server never materialises the mean. It streams each
delta straight into `global_weights` with a scale (`stream_add_base_delta`), so a
second full vector would be needed to compute the mean, and a third to keep the
previous gradient. This module supplies the coefficient algebra that makes the
same update expressible as *two* streaming adds plus one carried vector.

Convention warning: this follows `crowdgpt_ng.diloco.OuterOptimizer`, which is
what the measured 306M run used. Its Nesterov form uses the previous *gradient*
(not an EMA of gradients)::

    step = grad + momentum * (grad - prev_grad)

Textbook heavy-ball (`momentum * prev + grad`) is available via `nesterov=False`
and gives a different trajectory. Keeping the two switchable is deliberate: the
measured numbers belong to one of them.
"""

from __future__ import annotations

# Above this the update is not "momentum", it is divergence: 0.9-0.99 is the range
# every DiLoCo variant uses, and the outer step is applied once per round (~10-30
# inner steps), so a value near 1 compounds across very little real work.
MAX_MOMENTUM = 0.99
DEFAULT_MOMENTUM = 0.9


def resolve_momentum(raw, *, default: float = 0.0) -> float:
    """Parse a configured/environment momentum value, clamped to a sane range.

    ``default=0.0`` means "off", which is the behaviour the server shipped with:
    a deployment that does not ask for DiLoCo semantics must not get them
    implicitly, because it changes every round's magnitude.
    """
    if raw is None or raw == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if value != value:                      # NaN
        return float(default)
    return max(0.0, min(MAX_MOMENTUM, value))


def nesterov_update(theta, grad, prev_grad=None, *, lr: float,
                    momentum: float = DEFAULT_MOMENTUM, nesterov: bool = True):
    """One outer step in vector form: the definition everything else must match.

    ``theta`` is the pre-round weights, ``grad = theta - mean_client_weights``.
    Returns the new weights; the caller keeps ``grad`` as the next ``prev_grad``.
    """
    if prev_grad is None:
        prev_grad = grad * 0.0
    if nesterov:
        step = grad + momentum * (grad - prev_grad)
    else:
        step = grad + momentum * prev_grad
    return theta - lr * step


def streaming_scales(*, lr: float, momentum: float, nesterov: bool = True) -> dict:
    """Scales for the streaming form, which is what the aggregator can afford.

    Per round, with ``g`` the carried vector and ``d_i`` a delta with effective
    weight ``eff_i``::

        w += carry * g                        # the momentum term, once
        g  = 0
        for each delta: w += weight * eff * d ; g += buffer * eff * d

    The buffer holds the **gradient**, not the delta sum: with
    ``s = sum(eff_i * d_i)`` the gradient is ``grad = theta - mean = -s``, so
    ``buffer = -1`` in both variants. It is stored lr-free, so changing the
    learning-rate schedule does not silently rescale history.

    Nesterov: ``theta - lr*(grad + b*(grad - grad_prev))``
    ``= theta + lr*(1+b)*s + lr*b*grad_prev``, hence ``weight = lr*(1+b)`` and
    ``carry = +lr*b``.
    Heavy-ball: ``theta - lr*(grad + b*grad_prev) = theta + lr*s - lr*b*grad_prev``,
    so only the carry sign flips.
    """
    beta = max(0.0, min(MAX_MOMENTUM, float(momentum)))
    return {
        "carry": float(lr) * beta if nesterov else -float(lr) * beta,
        "weight": float(lr) * (1.0 + beta) if nesterov else float(lr),
        "buffer": -1.0,
        "factor": (1.0 + beta) if nesterov else 1.0,
    }


def buffer_bytes(model_size: int, dtype_size: int = 4) -> float:
    """Size of the carried gradient in GB, for the memory guard."""
    return model_size * dtype_size / (1024 ** 3)
