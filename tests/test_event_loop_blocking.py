"""Slow work and threading locks must stay off the event loop, or one upload stalls every client."""

import ast
import hashlib
import secrets
import threading
import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SERVER.read_text(encoding="utf-8"))
THREAD_LOCKS = {"pool_lock", "uploads_lock", "users_lock", "shard_lock", "stats_lock"}
# submit_chunk only updates byte counters under uploads_lock, and no holder of that lock does large IO.
# startup runs once before the server accepts requests.
ALLOWED = {"submit_chunk": {"uploads_lock"}, "startup": THREAD_LOCKS}


def _function(name):
    for node in TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in server.py")


def _locks_taken(node):
    return {item.context_expr.id for w in ast.walk(node) if isinstance(w, ast.With)
            for item in w.items if isinstance(item.context_expr, ast.Name)} & THREAD_LOCKS


def _calls(node, name):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call)
            and getattr(c.func, "id", getattr(c.func, "attr", None)) == name]


def _load(names, ns):
    funcs = []
    for n in names:
        f = _function(n)
        f.decorator_list = []
        funcs.append(f)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SERVER), "exec"), ns)
    return ns


class _HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(status_code, detail)
        self.status_code = status_code


class AsyncHandlerTest(unittest.TestCase):
    def test_no_threading_lock_inside_async_handlers(self):
        for node in TREE.body:
            if isinstance(node, ast.AsyncFunctionDef):
                self.assertEqual(_locks_taken(node) - ALLOWED.get(node.name, set()), set(), node.name)

    def test_heavy_handlers_hand_off_to_a_worker_thread(self):
        for name in ("submit_init", "submit_finalize", "object_put", "register", "auth_login"):
            self.assertTrue(_calls(_function(name), "run_in_threadpool"), name)

    def test_client_key_lookup_takes_no_lock(self):
        # get_client_key runs in the country middleware on the event loop for every request.
        self.assertEqual(_locks_taken(_function("get_client_key")), set())


class PoolMergeTest(unittest.TestCase):
    def test_merge_runs_outside_pool_lock(self):
        sync = _function("_complete_pool_sync")
        self.assertTrue(_calls(sync, "_merge_pool_deltas"))
        for w in ast.walk(sync):
            if isinstance(w, ast.With) and any(getattr(i.context_expr, "id", None) == "pool_lock" for i in w.items):
                self.assertFalse(_calls(w, "_merge_pool_deltas"))


class LoginTest(unittest.TestCase):
    def setUp(self):
        salt = "ab" * 8
        self.saves = []
        self.ns = _load(("hash_password", "_login_user"), {
            "hashlib": hashlib, "secrets": secrets, "HTTPException": _HTTPException,
            "users_lock": threading.Lock(), "_is_user_banned": lambda uid: False,
            "generate_token": lambda: "tok", "save_users_db": lambda: self.saves.append(1),
            "log": type("L", (), {"info": staticmethod(lambda *a: None)})})
        self.ns["users_db"] = {"tokens": {}, "users": {"u1": {
            "uuid": "u1", "username": "Ann", "username_lower": "ann", "salt": salt,
            "password_hash": self.ns["hash_password"]("pw123456", salt)}}}

    def test_good_password_issues_token(self):
        out = self.ns["_login_user"]({"username": "ann", "password": "pw123456"})
        self.assertEqual(out["token"], "tok")
        self.assertEqual(self.ns["users_db"]["tokens"], {"tok": "u1"})
        self.assertEqual(self.saves, [1])

    def test_bad_password_rejected_without_save(self):
        with self.assertRaises(_HTTPException) as cm:
            self.ns["_login_user"]({"username": "ann", "password": "wrong"})
        self.assertEqual(cm.exception.status_code, 401)
        self.assertEqual(self.saves, [])


class RoundStatusTest(unittest.TestCase):
    def test_known_client_does_not_wait_for_shard_lock(self):
        node = _function("get_round_status")
        guarded = [w for w in ast.walk(node) if isinstance(w, ast.If)
                   and any(getattr(i.context_expr, "id", None) == "shard_lock"
                           for b in w.body if isinstance(b, ast.With) for i in b.items)]
        self.assertTrue(guarded, "shard_lock should only be taken inside a 'client not in round' check")


if __name__ == "__main__":
    unittest.main()
