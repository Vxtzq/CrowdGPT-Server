"""Small, explicit adversarial checks used by the coordinator."""
from __future__ import annotations

import math

# A cross-entropy above the uniform baseline (ln 151669 ~= 11.93) is noise;
# a value below ~1 nat for this corpus/model is not a plausible step mean.
LOSS_FLOOR = 1.0
LOSS_CEILING = 20.0


def plausible_loss(value) -> bool:
    try: v = float(value)
    except (TypeError, ValueError): return False
    return math.isfinite(v) and LOSS_FLOOR <= v <= LOSS_CEILING


def same_origin(a: dict, b: dict) -> bool:
    """True when two submissions share a device or /24 subnet."""
    if a.get("device") and a.get("device") == b.get("device"):
        return True
    sa, sb = a.get("subnet"), b.get("subnet")
    return bool(sa and sb and sa == sb)


__all__ = ["LOSS_CEILING", "LOSS_FLOOR", "plausible_loss", "same_origin"]
