"""State-safety regressions: importing server.py must not overwrite state, users.json
writes are atomic, the cleanup thread survives races, and missing weights are reported.

Like test_reload_checkpoint.py, these tests do not import server.py (importing it
used to overwrite users.json and global_state.json). They extract the real
functions with ``ast`` and run them in a namespace that stands in for the module.

Run:  python3 -m unittest discover -s tests -v
"""
import ast
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

SERVER_PY = Path(__file__).resolve().parent.parent / "server.py"
TREE = ast.parse(SERVER_PY.read_text(encoding="utf-8"))


def _function(name):
    for node in TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in server.py")


def _load(names, ns):
    funcs = []
    for name in names:
        fn = _function(name)
        fn.decorator_list = []
        funcs.append(fn)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SERVER_PY), "exec"), ns)
    return ns


def _calls_atexit_register(node):
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "register"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "atexit"
        for n in ast.walk(node)
    )


class _FakeTorch:
    @staticmethod
    def from_numpy(arr):
        return arr


class ExitHookTest(unittest.TestCase):
    def test_import_does_not_register_exit_save(self):
        top_level = [n for n in TREE.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        self.assertFalse(any(_calls_atexit_register(n) for n in top_level))

    def test_startup_registers_exit_save(self):
        self.assertTrue(_calls_atexit_register(_function("startup")))

    def test_shutdown_saves_once(self):
        # uvicorn re-raises SIGTERM after shutdown, so atexit hooks never run; shutdown() must save.
        calls = []
        atexit_stub = type("A", (), {"unregister": staticmethod(lambda fn: calls.append(("unregister", fn)))})
        save = lambda: calls.append("save")
        ns = _load(("shutdown",), {"atexit": atexit_stub, "emergency_save": save})
        ns["shutdown"]()
        self.assertEqual(calls, [("unregister", save), "save"])


class UsersDbTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "users.json"
        self.ns = _load(
            ("_write_json_atomic", "save_users_db", "load_users_db"),
            {"json": json, "os": os, "Path": Path, "log": logging.getLogger("test"),
             "USERS_DB_FILE": self.path, "users_db": {"users": {}, "tokens": {}}},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_then_load_round_trips_without_temp_file(self):
        self.ns["users_db"] = {"users": {"u1": {"name": "a"}}, "tokens": {}}
        self.ns["save_users_db"]()
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())
        self.ns["users_db"] = {}
        self.ns["load_users_db"]()
        self.assertEqual(self.ns["users_db"]["users"], {"u1": {"name": "a"}})

    def test_unreadable_file_raises_instead_of_starting_empty(self):
        self.path.write_text('{"users": {"u1"', encoding="utf-8")
        self.ns["users_db"] = {"users": {"keep": {}}, "tokens": {}}
        with self.assertRaises(json.JSONDecodeError):
            self.ns["load_users_db"]()
        self.assertEqual(self.ns["users_db"]["users"], {"keep": {}})
        self.assertEqual(self.path.read_text(encoding="utf-8"), '{"users": {"u1"')


class _VanishingFile:
    def stat(self):
        raise FileNotFoundError("removed by aggregation")

    def unlink(self):
        raise AssertionError("must not unlink a file it could not stat")


class _DeltaDir:
    def glob(self, _pattern):
        return [_VanishingFile()]


class CleanupTest(unittest.TestCase):
    def _ns(self):
        noop = lambda *a, **k: None  # noqa: E731
        return _load(("cleanup_loop", "_cleanup_tick"), {
            "time": time, "gc": type("G", (), {"collect": staticmethod(noop)}),
            "log": logging.getLogger("test"), "_auth_rate_lock": threading.Lock(),
            "_register_attempts": {}, "_login_attempts": {}, "active_clients": {},
            "client_hardware": {}, "client_last_submit": {}, "blacklisted_subnets": {},
            "banned_client_keys": {}, "banned_users": {"old": {"until": 1.0}, "live": {"until": None}},
            "db": {"tasks": {}, "shardMeta": {}, "pendingSubmissions": {}},
            "FL_CONFIG": {"IP_TIMEOUT_S": 60, "SHARD_TIMEOUT_S": 60, "CLEANUP_INTERVAL_S": 0},
            "DELTA_TEMP_DIR": _DeltaDir(), "_cleanup_expired_uploads": noop,
            "_sweep_orphans": noop, "_pool_cleanup_tick": noop,
        })

    def test_tick_survives_delta_file_vanishing_and_expires_bans(self):
        ns = self._ns()
        ns["_cleanup_tick"]()
        self.assertEqual(set(ns["banned_users"]), {"live"})

    def test_loop_keeps_running_after_a_failed_tick(self):
        ns = self._ns()
        calls = []

        def tick():
            calls.append(1)
            if len(calls) == 1:
                raise KeyError("raced")
            raise SystemExit  # stop the infinite loop after the second pass

        ns["_cleanup_tick"] = tick
        with self.assertLogs("test", level="ERROR"):
            with self.assertRaises(SystemExit):
                ns["cleanup_loop"]()
        self.assertEqual(len(calls), 2)


class LoadWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model_latest.bin"
        self.ns = _load(("load_model_weights",), {
            "np": np, "torch": _FakeTorch, "log": logging.getLogger("test"),
            "LATEST_CHECKPOINT_FILE": self.path, "MODEL_SIZE": 4, "global_weights": None,
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_returns_false(self):
        self.assertFalse(self.ns["load_model_weights"]())

    def test_short_file_returns_false(self):
        np.zeros(3, dtype=np.float32).tofile(self.path)
        self.assertFalse(self.ns["load_model_weights"]())
        self.assertIsNone(self.ns["global_weights"])

    def test_good_file_returns_true(self):
        np.arange(4, dtype=np.float32).tofile(self.path)
        self.assertTrue(self.ns["load_model_weights"]())
        self.assertEqual(list(self.ns["global_weights"]), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
