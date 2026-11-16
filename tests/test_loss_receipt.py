"""Tests for the signed loss-ratchet receipt (P1-S3).

The ratchet is the only thing standing between a client and the shared weights,
yet the old `_query_evaluator` trusted whatever JSON the configured URL
returned and even shipped `previousAcceptedLoss` so the caller could talk the
ratchet down. These tests pin the replacement: the evaluator must sign a
receipt that binds the measurement to the round, the step, the request and the
exact staged candidate bytes, and a receipt that does not verify is a rejection
in every mode.

Like ``test_reload_checkpoint.py`` this does not import ``server.py`` (which
pulls in torch, FastAPI and background state). It extracts the real functions
with ``ast`` and runs them against a namespace that plays the module globals.
Signing needs ``cryptography``; those tests skip when it is absent.

Run:  python3 -m unittest discover -s tests -v
"""
import ast
import hashlib
import json
import math
import time
import unittest
from pathlib import Path

SERVER_PY = Path(__file__).resolve().parent.parent / "server.py"

EXTRACTED = (
    "_parse_loss_evaluator_keys",
    "_loss_canonical",
    "_verify_ed25519",
    "_loss_receipt_problem",
)

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    HAVE_CRYPTO = True
except Exception:  # pragma: no cover - CI installs cryptography for the orchestrator
    Ed25519PrivateKey = None
    HAVE_CRYPTO = False


def _extract_source(names):
    tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"), filename=str(SERVER_PY))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = node
    missing = set(names) - set(found)
    if missing:
        raise AssertionError(f"functions not found in server.py: {sorted(missing)}")
    return compile(ast.Module(body=[found[n] for n in names], type_ignores=[]),
                   str(SERVER_PY), "exec")


_CODE = _extract_source(EXTRACTED)


def _keypair():
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key().public_bytes_raw().hex()


def _sign(priv, receipt):
    return priv.sign(json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8")).hex()


def _candidate(model="a" * 64, engram="b" * 64):
    return {"modelSha256": model, "engramSha256": engram, "format": "bf16"}


class _Base(unittest.TestCase):
    def setUp(self):
        self.prv, self.pub = _keypair() if HAVE_CRYPTO else (None, "0" * 64)
        self.ns = {
            "__name__": "server_extract",
            "hashlib": hashlib,
            "json": json,
            "math": math,
            "time": time,
            "Dict": dict,
            "Optional": __import__("typing").Optional,
            "HARDEN": {"LOSS_RECEIPT_MAX_AGE_S": 3600},
            "LOSS_EVALUATOR_KEYS": {"primary": self.pub},
            "loss_ratchet_state": {"receipts": []},
        }
        exec(_CODE, self.ns)

    def receipt(self, **overrides):
        body = {
            "step": 500,
            "round": 7,
            "requestId": "req-1",
            "modelSha256": "a" * 64,
            "engramSha256": "b" * 64,
            "loss": 4.25,
            "issuedAt": time.time(),
            "evaluator": "primary",
        }
        body.update(overrides)
        return body

    def problem(self, receipt, *, signature=None, pubkey=None, step=500, rnd=7,
                request_id="req-1", cand=None):
        sig = _sign(self.prv, receipt) if signature is None and HAVE_CRYPTO else (signature or "")
        return self.ns["_loss_receipt_problem"](
            receipt, step=step, rnd=rnd, request_id=request_id,
            cand=cand if cand is not None else _candidate(),
            signature=sig, pubkey=self.pub if pubkey is None else pubkey)


class ParseKeyTests(unittest.TestCase):
    def setUp(self):
        self.parse = None
        ns = {"Dict": dict}
        exec(_extract_source(("_parse_loss_evaluator_keys",)), ns)
        self.parse = ns["_parse_loss_evaluator_keys"]

    def test_bare_pubkey_gets_a_derived_name(self):
        pub = "c" * 64
        self.assertEqual(self.parse(pub), {pub[:16]: pub})

    def test_named_key_and_separators(self):
        a, b = "d" * 64, "e" * 64
        parsed = self.parse(f" primary:{a} ; backup:{b} , {b} ")
        self.assertEqual(parsed["primary"], a)
        self.assertEqual(parsed["backup"], b)

    def test_garbage_is_dropped_rather_than_widening_trust(self):
        self.assertEqual(self.parse(""), {})
        self.assertEqual(self.parse("primary:nothex"), {})
        self.assertEqual(self.parse(":" + "f" * 63), {})
        self.assertEqual(self.parse("x" * 64), {})   # not hex
        self.assertEqual(self.parse("g" * 64), {})   # not hex


@unittest.skipUnless(HAVE_CRYPTO, "cryptography not installed")
class ReceiptVerdictTests(_Base):
    def test_valid_receipt_is_accepted(self):
        self.assertIsNone(self.problem(self.receipt()))

    def test_missing_receipt_is_rejected(self):
        self.assertEqual(self.problem(None), "receipt_missing")
        self.assertEqual(self.problem({}), "receipt_missing")

    def test_untrusted_key_is_rejected(self):
        _, other = _keypair()
        self.assertEqual(self.problem(self.receipt(), pubkey=other), "receipt_untrusted_key")

    def test_tampered_loss_breaks_the_signature(self):
        receipt = self.receipt()
        signature = _sign(self.prv, receipt)
        receipt["loss"] = 1.0
        self.assertEqual(self.problem(receipt, signature=signature), "receipt_signature_invalid")

    def test_receipt_is_bound_to_the_step_and_round(self):
        self.assertEqual(self.problem(self.receipt(), step=501), "receipt_step_mismatch")
        self.assertEqual(self.problem(self.receipt(), rnd=8), "receipt_round_mismatch")

    def test_receipt_is_bound_to_the_request_nonce(self):
        self.assertEqual(self.problem(self.receipt(), request_id="req-2"), "receipt_request_mismatch")

    def test_receipt_is_bound_to_the_candidate_bytes(self):
        other = _candidate(model="c" * 64)
        self.assertEqual(self.problem(self.receipt(), cand=other), "receipt_candidate_mismatch")
        self.assertEqual(self.problem(self.receipt(modelSha256="c" * 64)), "receipt_candidate_mismatch")

    def test_stale_receipt_is_rejected(self):
        old = self.receipt(issuedAt=time.time() - 7200)
        self.assertEqual(self.problem(old), "receipt_stale")

    def test_age_check_can_be_disabled(self):
        self.ns["HARDEN"]["LOSS_RECEIPT_MAX_AGE_S"] = 0
        old = self.receipt(issuedAt=time.time() - 10 * 24 * 3600)
        self.assertIsNone(self.problem(old))

    def test_non_finite_or_missing_loss_is_rejected(self):
        self.assertEqual(self.problem(self.receipt(loss=float("nan"))), "receipt_loss_invalid")
        self.assertEqual(self.problem(self.receipt(loss="4.25")), "receipt_loss_invalid")

    def test_replayed_receipt_is_rejected(self):
        receipt = self.receipt()
        digest = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        self.ns["loss_ratchet_state"]["receipts"].append(digest)
        self.assertEqual(self.problem(receipt), "receipt_replayed")

    def test_installed_evaluator_key_makes_a_second_key_untrusted(self):
        _, second = _keypair()
        self.ns["LOSS_EVALUATOR_KEYS"]["second"] = second
        self.assertIsNone(self.problem(self.receipt()))
        self.assertEqual(self.problem(self.receipt(), pubkey="9" * 64), "receipt_untrusted_key")


class DefaultPostureTests(unittest.TestCase):
    """The insecure defaults are the actual bug, so pin them in the source."""

    @classmethod
    def setUpClass(cls):
        cls.source = SERVER_PY.read_text(encoding="utf-8")

    def test_an_unreachable_evaluator_fails_closed_by_default(self):
        self.assertIn('os.getenv("CROWDGPT_LOSS_FAIL_OPEN", "0")', self.source)
        self.assertNotIn('os.getenv("CROWDGPT_LOSS_FAIL_OPEN", "1")', self.source)

    def test_signed_receipts_are_required_by_default(self):
        self.assertIn('os.getenv("CROWDGPT_LOSS_RECEIPT_REQUIRED", "1")', self.source)

    def test_the_evaluator_request_no_longer_carries_the_previous_loss(self):
        start = self.source.index("def _query_evaluator")
        end = self.source.index("def ", start + 10)
        body = self.source[start:end]
        self.assertNotIn("previousAcceptedLoss\":", body)
        self.assertIn('"requestId": request_id', body)

    def test_a_receipt_problem_rejects_the_round_even_when_fail_open(self):
        start = self.source.index("def _finish_with_loss_check")
        end = self.source.index("def _valid_upload_id")
        body = self.source[start:end]
        reject = body.index("if problem:")
        fail_open = body.index("if HARDEN[\"LOSS_FAIL_OPEN\"]")
        self.assertLess(reject, fail_open, "a bad receipt must be checked before fail-open")


if __name__ == "__main__":
    unittest.main()
