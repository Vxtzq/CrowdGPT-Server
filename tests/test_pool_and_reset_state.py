"""Pool matching and reset state: who may share a pool, and what a reset must clear."""

import ast
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

SERVER = Path(__file__).resolve().parents[1] / "server.py"
TREE = ast.parse(SERVER.read_text(encoding="utf-8"))


def _function(name):
    for node in TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            node.decorator_list = []
            return node
    raise AssertionError(f"{name} not found in server.py")


def _load(names, ns):
    module = ast.Module(body=[_function(n) for n in names], type_ignores=[])
    exec(compile(module, str(SERVER), "exec"), ns)
    return ns


def _sig(engram):
    return ("CUDA", 16, "fp32", "adamw", "ada", "4090", "8.9", engram)


class EngramPoolTest(unittest.TestCase):
    def setUp(self):
        self.ns = _load(("_pools_compatible", "_hw_signature"), {
            "POOL_BF16": "off", "client_hardware": {},
            "_gpu_arch_class": lambda g: "ada", "_gpu_card_class": lambda g: "4090",
        })

    def test_engram_and_no_engram_clients_do_not_share_a_pool(self):
        ok = self.ns["_pools_compatible"]
        self.assertTrue(ok(_sig(True), _sig(True)))
        self.assertTrue(ok(_sig(False), _sig(False)))
        self.assertFalse(ok(_sig(False), _sig(True)))
        self.assertFalse(ok(_sig(True), _sig(False)))

    def test_pools_saved_without_the_flag_count_as_engram_pools(self):
        ok = self.ns["_pools_compatible"]
        old = _sig(True)[:7]
        self.assertTrue(ok(_sig(True), old))
        self.assertFalse(ok(_sig(False), old))

    def test_signature_carries_the_engram_header(self):
        self.ns["client_hardware"].update({
            "ip:a": {"backend": "cuda", "bs": 16, "gpu": "RTX 4090", "engram": False},
            "ip:b": {"backend": "cuda", "bs": 16, "gpu": "RTX 4090"},
        })
        self.assertFalse(self.ns["_hw_signature"]("ip:a")[7])
        self.assertTrue(self.ns["_hw_signature"]("ip:b")[7])


class StarvedWaiterTest(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        self.ns = _load(("_starved_peer_waiting",), {
            "time": time, "MATCHMAKING": {"STARVED_ADMIT_S": 10}, "FL_CONFIG": {"IP_TIMEOUT_S": 900},
            "_WAITERS": {"ip:w": {"since": self.now - 100, "hw": _sig(True)}},
            "active_clients": {},
            "_pools_compatible": lambda a, b: True, "_pool_contains_same_actor": lambda p, k: False,
        })
        self.pool = {"clients": ["ip:a"], "hardware_sig": _sig(True)}

    def test_waiter_that_stopped_polling_does_not_hold_the_pool(self):
        self.ns["active_clients"]["ip:w"] = self.now - 2000
        self.assertFalse(self.ns["_starved_peer_waiting"](self.pool))

    def test_live_waiter_still_holds_the_pool(self):
        self.ns["active_clients"]["ip:w"] = self.now
        self.assertTrue(self.ns["_starved_peer_waiting"](self.pool))


class ResetClearsMomentumTest(unittest.TestCase):
    def test_reset_drops_outer_momentum(self):
        outer = {"grad": object(), "momentum": None, "rounds": 3}
        ns = _load(("_reset_weights_to_init",), {
            "build_init_weights": lambda: "init",
            "torch": SimpleNamespace(randn=lambda n: _Scalable()),
            "ENG_PARAMS": 4, "loss_ratchet_state": {}, "_save_loss_state": lambda: None,
            "_WN_CACHE": {}, "_OUTER": outer,
        })
        ns["_reset_weights_to_init"]()
        self.assertIsNone(outer["grad"])
        self.assertEqual(outer["rounds"], 0)

    def test_reload_drops_outer_momentum(self):
        source = ast.unparse(_function("reload_checkpoint"))
        self.assertIn("_OUTER.update(grad=None, rounds=0)", source)


class _Scalable:
    def __mul__(self, other):
        return self

    def float(self):
        return self


class BanFreesPoolSeatTest(unittest.TestCase):
    def test_banned_user_leaves_forming_pools(self):
        removed, joined = [], []
        pools = {
            "solo": {"status": "forming", "clients": ["user:u1"], "deltas": {}},
            "pair": {"status": "forming", "clients": ["user:u2", "user:u1"], "deltas": {}},
            "busy": {"status": "training", "clients": ["user:u3"], "deltas": {}},
        }
        ns = {
            "time": time, "datetime": __import__("datetime").datetime,
            "timezone": __import__("datetime").timezone,
            "HTTPException": Exception, "Request": object,
            "_admin_ok": lambda s: True, "_admin_secret": lambda r, s: s,
            "users_lock": threading.Lock(), "user_countries_lock": threading.Lock(),
            "shard_lock": threading.Lock(), "pool_lock": threading.Lock(),
            "users_db": {"users": {"u1": {"username": "mallory"}}, "tokens": {}},
            "user_countries": {}, "banned_users": {}, "banned_client_keys": {},
            "save_users_db": lambda: None, "ban_subnet": lambda *a: None,
            "round_state": {"pending_deltas": []}, "_delta_metadata": lambda p: None,
            "pools": pools, "client_pool_map": {"user:u1": "pair", "user:u2": "pair"},
            "_user_id_from_key": lambda k: k.split(":", 1)[1],
            "_persist_pool_removed": removed.append,
            "_persist_pool_join": lambda pool, c: joined.append(c),
            "PUB": SimpleNamespace(event=lambda *a, **k: None, who=lambda u: u),
            "log": SimpleNamespace(warning=lambda *a, **k: None),
        }
        _load(("admin_ban_user",), ns)
        ns["admin_ban_user"](secret="ok", user_id="u1")
        self.assertNotIn("solo", pools)
        self.assertEqual(removed, ["solo"])
        self.assertEqual(pools["pair"]["clients"], ["user:u2"])
        self.assertEqual(joined, ["user:u2"])
        self.assertEqual(ns["client_pool_map"], {"user:u2": "pair"})
        self.assertEqual(pools["busy"]["clients"], ["user:u3"])


if __name__ == "__main__":
    unittest.main()
