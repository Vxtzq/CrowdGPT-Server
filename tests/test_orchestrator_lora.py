"""LoRA wire format and train-mode enforcement.

The coordinator's HTTP layer is not imported here: the rules that keep dense
and LoRA pools apart, and the payload decoder the server calls, both live in
`orchestrator.lora` and are exercised directly.
"""
import numpy as np
import pytest

from orchestrator import lora


def _adapter(name, offset, rows, cols, rank=4, seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((rank, cols)) * 0.05).astype(np.float32)
    b = (rng.standard_normal((rows, rank)) * 0.25).astype(np.float32)
    return {"name": name, "offset": offset, "a": a, "b": b}


# ============ TRAIN-MODE IDENTITY ============

def _sig(mode=None, rank=None, backend="CUDA"):
    base = (backend, 8, "bf16", "adamw", "ampere", "rtx30", "8.0")
    return base if mode is None else base + (mode, 0 if mode == "dense" else rank)


def test_legacy_signatures_read_as_dense():
    assert lora.signature_mode(_sig()) == ("dense", 0)
    assert lora.signatures_compatible(_sig(), _sig())


def test_dense_and_lora_signatures_never_match():
    dense, adapter = _sig("dense", 0), _sig("lora", 16)
    assert not lora.signatures_compatible(dense, adapter)
    assert not lora.signatures_compatible(adapter, dense)
    # A legacy signature is a dense signature.
    assert not lora.signatures_compatible(_sig(), adapter)


def test_lora_rank_is_part_of_pool_identity():
    assert lora.signatures_compatible(_sig("lora", 16), _sig("lora", 16))
    assert not lora.signatures_compatible(_sig("lora", 16), _sig("lora", 32))
    # A missing rank only equals another missing rank.
    assert lora.signatures_compatible(_sig("lora", 0), _sig("lora", 0))


def test_mode_and_rank_parsing():
    assert lora.train_mode_of("LoRA") == "lora"
    assert lora.train_mode_of("adapters") == "lora"
    assert lora.train_mode_of(None) == "dense"
    assert lora.train_mode_of("whatever") == "dense"
    assert lora.rank_of("16") == 16
    assert lora.rank_of(-3) == 0
    assert lora.rank_of("nope") == 0
    assert lora.rank_of(None) == 0


def test_trained_params_follow_the_mode():
    assert lora.trained_params("dense", has_engram=False) == ["base"]
    assert lora.trained_params("dense", has_engram=True) == ["base", "engram"]
    assert lora.trained_params("lora", has_engram=False) == ["adapters"]
    assert lora.trained_params("lora", has_engram=True) == ["adapters", "engram"]


# ============ SUBMISSION CONSISTENCY ============

def _task(mode="dense", rank=0, trained=("base",)):
    return {"trainMode": mode, "loraRank": rank, "trainedParams": list(trained)}


def test_lora_body_in_dense_task_is_rejected():
    body = {"isLoRA": True, "loraRank": 16, "trainedParams": ["adapters"]}
    with pytest.raises(ValueError, match="train_mode_mismatch"):
        lora.check_submission(_task(), body)


def test_dense_body_in_lora_task_is_rejected():
    body = {"isLoRA": False, "loraRank": 0, "trainedParams": ["base"]}
    with pytest.raises(ValueError, match="train_mode_mismatch"):
        lora.check_submission(_task("lora", 16, ["adapters"]), body)


def test_rank_mismatch_is_rejected():
    body = {"isLoRA": True, "loraRank": 32, "trainedParams": ["adapters"]}
    with pytest.raises(ValueError, match="lora_rank_mismatch"):
        lora.check_submission(_task("lora", 16, ["adapters"]), body)


def test_trained_params_beyond_the_task_are_rejected():
    body = {"isLoRA": True, "loraRank": 16, "trainedParams": ["adapters", "base"]}
    with pytest.raises(ValueError, match="trained_params_mismatch"):
        lora.check_submission(_task("lora", 16, ["adapters"]), body)


def test_matching_lora_submission_is_accepted():
    body = {"isLoRA": True, "loraRank": "16", "trainedParams": ["adapters", "engram"]}
    assert lora.check_submission(_task("lora", 16, ["adapters", "engram"]), body) == (True, 16)


def test_dense_clients_and_legacy_tasks_keep_working():
    # A pre-LoRA client sends none of the new fields.
    assert lora.check_submission(_task(), {}) == (False, 0)
    assert lora.check_submission({"trainMode": None, "trainedParams": None}, {}) == (False, 0)
    # Declaring dense explicitly is equivalent.
    assert lora.check_submission(_task(), {"trainMode": "dense", "isLoRA": False}) == (False, 0)


def test_rank_only_body_is_treated_as_lora():
    with pytest.raises(ValueError, match="train_mode_mismatch"):
        lora.check_submission(_task(), {"loraRank": 16})


# ============ WIRE FORMAT ============

def test_payload_round_trips_and_expands_to_dense():
    item = _adapter("blocks.0.attn.wq.weight", 4, rows=8, cols=4, rank=2, seed=7)
    blob = lora.encode_payload([item])
    parsed = lora.decode_payload(blob, model_size=64)
    assert [p["name"] for p in parsed] == ["blocks.0.attn.wq.weight"]
    assert parsed[0]["offset"] == 4
    dense = lora.expand_to_dense(parsed, 64)
    expected = (item["b"] @ item["a"] * (lora.ALPHA / 2)).reshape(-1)
    assert np.allclose(dense[4:4 + 32], expected, atol=2e-3)
    assert np.allclose(dense[:4], 0.0)
    assert np.allclose(dense[4 + 32:], 0.0)


def test_streamed_chunks_match_the_dense_expansion():
    items = [_adapter("blocks.0.attn.wq.weight", 3, rows=2, cols=3, rank=1, seed=1),
             _adapter("blocks.0.mlp.w3.weight", 12, rows=1, cols=4, rank=1, seed=2)]
    blob = lora.encode_payload(items)
    parsed = lora.decode_payload(blob, model_size=20)
    chunks = list(lora.iter_dense_chunks(parsed, 20, chunk=8))
    assert [len(c) for c in chunks] == [8, 8, 4]          # adapter 1 spans a chunk boundary
    assert np.allclose(np.concatenate(chunks), lora.expand_to_dense(parsed, 20), atol=1e-6)


def test_payload_rejects_bad_magic_truncation_and_trailing_bytes():
    blob = lora.encode_payload([_adapter("x", 0, rows=4, cols=2, rank=1)])
    with pytest.raises(ValueError, match="magic"):
        lora.decode_payload(b"XXXX" + blob[4:], model_size=16)
    with pytest.raises(ValueError):
        lora.decode_payload(blob[:-3], model_size=16)
    with pytest.raises(ValueError, match="trailing"):
        lora.decode_payload(blob + b"\x00", model_size=16)


def test_payload_rejects_out_of_bounds_and_overlapping_spans():
    item = _adapter("x", 0, rows=4, cols=4, rank=1)
    with pytest.raises(ValueError, match="exceeds model size"):
        lora.decode_payload(lora.encode_payload([item]), model_size=8)
    overlapping = [item, {"name": "y", "offset": 4, "a": item["a"], "b": item["b"]}]
    with pytest.raises(ValueError, match="overlapping"):
        lora.decode_payload(lora.encode_payload(overlapping), model_size=64)


def test_payload_rejects_empty_counts_and_non_finite_factors():
    with pytest.raises(ValueError, match="count"):
        lora.decode_payload(lora.WIRE_MAGIC + (0).to_bytes(4, "little"), model_size=16)
    item = _adapter("x", 0, rows=4, cols=2, rank=1)
    item["a"] = item["a"].copy()
    item["a"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        lora.decode_payload(lora.encode_payload([item]), model_size=16)


def test_expansion_rejects_spans_outside_the_model():
    parsed = [{"name": "x", "offset": 8, "a": np.zeros((1, 4)), "b": np.zeros((2, 1))}]
    with pytest.raises(ValueError, match="outside the model"):
        lora.expand_to_dense(parsed, 10)
    with pytest.raises(ValueError, match="outside the model"):
        list(lora.iter_dense_chunks(parsed, 10))


def test_full_model_rank16_payload_stays_far_below_a_dense_delta():
    # 24 blocks x (4 attention + 3 MLP) rank-16 adapters on the 658M model.
    shapes = [(1536, 1536)] * 4 + [(2560, 1536), (2560, 1536), (1536, 2560)]
    block = [_adapter(f"blocks.0.w{i}.weight", 0, rows=r, cols=c, rank=16)
             for i, (r, c) in enumerate(shapes)]
    per_block = len(lora.encode_payload(block))
    total = per_block * 24
    assert 15_000_000 < total < 50_000_000            # ~17 MB, vs 1.3 GB dense
