"""Regression tests for POST /api/admin/reload (reload_checkpoint in server.py).

The tests do not import server.py (that would pull in torch, FastAPI, Hugging Face
and start background state).  Instead they extract the real function definitions
from server.py with ``ast`` and execute them in an isolated namespace that plays
the role of the module globals.  Only numpy is required; torch is replaced by a
tiny stub that supports the operations the extracted code uses.

Run:  python3 -m unittest discover -s tests -v
"""
import ast
import asyncio
import logging
import os
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

SERVER_PY = Path(__file__).resolve().parent.parent / "server.py"

# Real functions taken verbatim from server.py.
EXTRACTED = (
    "reload_checkpoint",
    "load_model_weights",
    "load_engram_weights",
    "save_model_weights",
    "save_engram_weights",
    "save_latest_checkpoint",
)

MODEL_SIZE = 12
ENG_PARAMS = 8


class _FakeTensor:
    """Minimal stand-in for the torch.Tensor API used by the extracted code."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def __getitem__(self, key):
        return _FakeTensor(self.arr[key])

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.arr


class _FakeTorch:
    Tensor = _FakeTensor

    @staticmethod
    def from_numpy(arr):
        return _FakeTensor(arr)


class _HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(status_code, detail)
        self.status_code = status_code
        self.detail = detail


def _extract_source(names):
    tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"), filename=str(SERVER_PY))
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []  # drop @app.post(...)
            found[node.name] = node
    missing = set(names) - set(found)
    if missing:
        raise AssertionError(f"functions not found in server.py: {sorted(missing)}")
    module = ast.Module(body=[found[n] for n in names], type_ignores=[])
    return compile(module, str(SERVER_PY), "exec")


_CODE = _extract_source(EXTRACTED)


def _write_f32(path, values):
    np.asarray(values, dtype=np.float32).tofile(path)


def _read_f32(path):
    return np.fromfile(path, dtype=np.float32)


class ReloadCheckpointTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.calls = []
        self.old_model = np.full(MODEL_SIZE, 9.0, dtype=np.float32)
        self.old_engram = np.full(ENG_PARAMS, -9.0, dtype=np.float32)
        ns = {
            "__name__": "server_extract",
            "np": np, "os": os, "gc": __import__("gc"),
            "torch": _FakeTorch, "HTTPException": _HTTPException,
            "log": logging.getLogger("test_reload_checkpoint"),
            "CHECKPOINT_DIR": self.dir,
            "LATEST_CHECKPOINT_FILE": self.dir / "model_latest.bin",
            "ENGRAM_CHECKPOINT_FILE": self.dir / "engram_latest.bin",
            "MODEL_SIZE": MODEL_SIZE, "ENG_PARAMS": ENG_PARAMS,
            # in-memory module state that is "already loaded" before the reload
            "global_weights": _FakeTensor(self.old_model.copy()),
            "global_engram_weights": _FakeTensor(self.old_engram.copy()),
            "global_step": 500,
            "loss_ratchet_state": {"lastAcceptedLoss": 1.0, "lastAcceptedStep": 500},
            "_admin_ok": lambda s: s == "ok",
            # side effects not under test: recorded, never touch network/disk
            "load_global_state": lambda: self.calls.append("load_global_state"),
            "save_global_state": lambda: self.calls.append("save_global_state"),
            "_save_loss_state": lambda: self.calls.append("_save_loss_state"),
            "sync_huggingface_state": lambda: self.calls.append("sync_huggingface_state"),
            "update_and_save_status": lambda push_hf=True: self.calls.append(("update_and_save_status", push_hf)),
        }
        exec(_CODE, ns)
        self.ns = ns

    def tearDown(self):
        self._tmp.cleanup()

    def _reload(self, step, secret="ok"):
        return asyncio.run(self.ns["reload_checkpoint"](step=step, secret=secret))

    def _step_files(self, step, model, engram=None):
        _write_f32(self.dir / f"model_step_{step}.bin", model)
        if engram is not None:
            _write_f32(self.dir / f"engram_step_{step}.bin", engram)

    # --- the bug ------------------------------------------------------------

    def test_reload_with_engram_updates_module_engram(self):
        model = np.arange(MODEL_SIZE, dtype=np.float32)
        engram = np.arange(ENG_PARAMS, dtype=np.float32) + 100
        self._step_files(100, model, engram)

        res = self._reload(100)

        self.assertEqual(res, {"status": "reloaded", "step": 100})
        self.assertEqual(self.ns["global_step"], 100)
        np.testing.assert_array_equal(self.ns["global_weights"].arr, model)
        np.testing.assert_array_equal(self.ns["global_engram_weights"].arr, engram)

    def test_reload_with_engram_saves_reloaded_engram_to_latest(self):
        model = np.arange(MODEL_SIZE, dtype=np.float32)
        engram = np.arange(ENG_PARAMS, dtype=np.float32) + 100
        self._step_files(100, model, engram)

        self._reload(100)

        np.testing.assert_array_equal(_read_f32(self.dir / "model_latest.bin"), model)
        np.testing.assert_array_equal(_read_f32(self.dir / "engram_latest.bin"), engram)
        # the source checkpoint must be left untouched
        np.testing.assert_array_equal(_read_f32(self.dir / "engram_step_100.bin"), engram)
        np.testing.assert_array_equal(_read_f32(self.dir / "model_step_100.bin"), model)

    def test_save_after_reload_does_not_write_through_hardlinked_latest(self):
        # handle_checkpoints() makes *_latest.bin hard links to the newest step files.
        newest_model = np.full(MODEL_SIZE, 7.0, dtype=np.float32)
        newest_engram = np.full(ENG_PARAMS, 7.0, dtype=np.float32)
        self._step_files(500, newest_model, newest_engram)
        os.link(self.dir / "model_step_500.bin", self.dir / "model_latest.bin")
        os.link(self.dir / "engram_step_500.bin", self.dir / "engram_latest.bin")
        model = np.arange(MODEL_SIZE, dtype=np.float32)
        engram = np.arange(ENG_PARAMS, dtype=np.float32) + 100
        self._step_files(100, model, engram)

        self._reload(100)

        np.testing.assert_array_equal(_read_f32(self.dir / "engram_latest.bin"), engram)
        np.testing.assert_array_equal(_read_f32(self.dir / "engram_step_500.bin"), newest_engram)
        np.testing.assert_array_equal(_read_f32(self.dir / "model_step_500.bin"), newest_model)

    def test_second_reload_after_save_round_trips(self):
        model = np.arange(MODEL_SIZE, dtype=np.float32)
        engram = np.arange(ENG_PARAMS, dtype=np.float32) + 100
        self._step_files(100, model, engram)
        self._reload(100)
        # simulate a restart-style reload of the latest files written above
        self.ns["global_weights"] = _FakeTensor(self.old_model.copy())
        self.ns["global_engram_weights"] = _FakeTensor(self.old_engram.copy())

        self._reload(-1)

        np.testing.assert_array_equal(self.ns["global_weights"].arr, model)
        np.testing.assert_array_equal(self.ns["global_engram_weights"].arr, engram)

    # --- unchanged behaviour on the other paths --------------------------------

    def test_reload_without_engram_keeps_current_engram(self):
        model = np.arange(MODEL_SIZE, dtype=np.float32)
        self._step_files(100, model)

        self._reload(100)

        self.assertEqual(self.ns["global_step"], 100)
        np.testing.assert_array_equal(self.ns["global_weights"].arr, model)
        np.testing.assert_array_equal(self.ns["global_engram_weights"].arr, self.old_engram)
        np.testing.assert_array_equal(_read_f32(self.dir / "engram_latest.bin"), self.old_engram)

    def test_reload_with_wrong_size_engram_keeps_current_engram(self):
        model = np.arange(MODEL_SIZE, dtype=np.float32)
        bad_engram = np.arange(ENG_PARAMS + 3, dtype=np.float32)
        self._step_files(100, model, bad_engram)

        res = self._reload(100)

        self.assertEqual(res["step"], 100)
        np.testing.assert_array_equal(self.ns["global_weights"].arr, model)
        np.testing.assert_array_equal(self.ns["global_engram_weights"].arr, self.old_engram)
        np.testing.assert_array_equal(_read_f32(self.dir / "engram_latest.bin"), self.old_engram)
        np.testing.assert_array_equal(_read_f32(self.dir / "engram_step_100.bin"), bad_engram)

    def test_reload_with_wrong_size_model_is_rejected_without_side_effects(self):
        self._step_files(100, np.zeros(MODEL_SIZE + 1), np.zeros(ENG_PARAMS))

        with self.assertRaises(_HTTPException) as cm:
            self._reload(100)

        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(self.ns["global_step"], 500)
        np.testing.assert_array_equal(self.ns["global_weights"].arr, self.old_model)
        np.testing.assert_array_equal(self.ns["global_engram_weights"].arr, self.old_engram)
        self.assertFalse((self.dir / "engram_latest.bin").exists())
        self.assertEqual(self.calls, [])

    def test_reload_missing_step_returns_404(self):
        with self.assertRaises(_HTTPException) as cm:
            self._reload(123)
        self.assertEqual(cm.exception.status_code, 404)
        self.assertEqual(self.calls, [])

    def test_reload_requires_admin_secret(self):
        with self.assertRaises(_HTTPException) as cm:
            self._reload(100, secret="wrong")
        self.assertEqual(cm.exception.status_code, 403)

    def test_reload_latest_loads_engram_from_latest_file(self):
        model = np.arange(MODEL_SIZE, dtype=np.float32)
        engram = np.arange(ENG_PARAMS, dtype=np.float32) + 100
        _write_f32(self.dir / "model_latest.bin", model)
        _write_f32(self.dir / "engram_latest.bin", engram)

        self._reload(-1)

        np.testing.assert_array_equal(self.ns["global_weights"].arr, model)
        np.testing.assert_array_equal(self.ns["global_engram_weights"].arr, engram)

    def test_reload_resets_ratchet_and_runs_post_reload_hooks(self):
        self._step_files(100, np.zeros(MODEL_SIZE), np.zeros(ENG_PARAMS))

        self._reload(100)

        self.assertEqual(self.ns["loss_ratchet_state"],
                         {"lastAcceptedLoss": None, "lastAcceptedStep": None})
        self.assertEqual(self.calls, ["_save_loss_state", "save_global_state",
                                      "sync_huggingface_state", ("update_and_save_status", True)])


_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_NESTED_SCOPES = _FUNCTION_SCOPES + (ast.ClassDef,)


def _own_scope_nodes(fn):
    """Yield the nodes evaluated in fn's own scope, not in scopes nested inside it.

    A nested def/lambda/class body is a separate scope, so its `global` statements
    and assignments must not be attributed to fn.  Only the parts of a nested
    definition that Python evaluates in the enclosing scope (decorators, default
    values, class bases/keywords) are kept.  Comprehensions are treated as part of
    fn: that is conservative (it can only add offenders, never hide one).
    """
    stack = [fn.body] if isinstance(fn, ast.Lambda) else list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.ClassDef):
            stack.extend(node.decorator_list + node.bases + node.keywords)
        elif isinstance(node, _FUNCTION_SCOPES):
            if not isinstance(node, ast.Lambda):
                stack.extend(node.decorator_list)
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d is not None)
        else:
            stack.extend(ast.iter_child_nodes(node))


def _undeclared_global_assignments(tree, names):
    """Return "func:line:name" for every function that assigns a name from `names`
    in its own scope without declaring it `global` in that same scope."""
    offenders = []
    for fn in ast.walk(tree):
        if not isinstance(fn, _FUNCTION_SCOPES):
            continue
        own = list(_own_scope_nodes(fn))
        declared = {n for g in own if isinstance(g, ast.Global) for n in g.names}
        hits = sorted(
            (node.lineno, node.id) for node in own
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            and node.id in names and node.id not in declared
        )
        fn_name = getattr(fn, "name", "<lambda>")
        offenders.extend(f"{fn_name}:{line}:{name}" for line, name in hits)
    return offenders


class GlobalDeclarationTest(unittest.TestCase):
    """Static guard: no function may rebind the global model state without `global`."""

    STATE = {"global_weights", "global_engram_weights", "global_step"}

    def _scan(self, source):
        source = textwrap.dedent(source)
        compile(source, "<snippet>", "exec")  # the snippet must be valid Python
        return _undeclared_global_assignments(ast.parse(source), self.STATE)

    def test_state_assignments_are_declared_global(self):
        tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"))
        self.assertEqual(_undeclared_global_assignments(tree, self.STATE), [])

    def test_guard_detects_original_reload_bug(self):
        # server.py as it was before the fix, rebuilt in memory only.
        head = ('@app.post("/api/admin/reload")\n'
                'async def reload_checkpoint(step: int = -1, secret: str = ""):\n')
        fixed = head + "    global global_weights, global_engram_weights, global_step\n"
        source = SERVER_PY.read_text(encoding="utf-8")
        self.assertEqual(source.count(fixed), 1)
        old = source.replace(fixed, head + "    global global_weights, global_step\n")

        offenders = _undeclared_global_assignments(ast.parse(old), self.STATE)

        self.assertEqual(len(offenders), 1)
        self.assertRegex(offenders[0], r"^reload_checkpoint:\d+:global_engram_weights$")

    def test_global_in_nested_function_does_not_cover_outer(self):
        # Reviewer's false green: only `inner` declares the global, so `outer`
        # still binds a local name.
        offenders = self._scan("""
            def outer():
                global_engram_weights = 1
                def inner():
                    global global_engram_weights
                    global_engram_weights = 2
                return inner
        """)
        self.assertEqual(offenders, ["outer:3:global_engram_weights"])

    def test_global_declared_in_own_scope_passes(self):
        offenders = self._scan("""
            def outer():
                global global_engram_weights, global_step
                global_engram_weights = 1
                global_step += 1
                def inner():
                    global global_engram_weights
                    global_engram_weights = 2
                class Helper:
                    def method(self):
                        global global_step
                        global_step = 3
                return inner, Helper
        """)
        self.assertEqual(offenders, [])

    def test_global_in_outer_does_not_cover_nested_function(self):
        offenders = self._scan("""
            def outer():
                global global_step
                global_step = 1
                def inner():
                    global_step = 2
                f = lambda: (global_step := 3)
                return inner, f
        """)
        self.assertEqual(offenders, ["inner:6:global_step", "<lambda>:7:global_step"])

    def test_nested_default_values_belong_to_enclosing_scope(self):
        offenders = self._scan("""
            def outer():
                def inner(x=(global_step := 1)):
                    global global_step
                return inner
        """)
        self.assertEqual(offenders, ["outer:3:global_step"])


if __name__ == "__main__":
    unittest.main()
