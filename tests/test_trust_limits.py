"""Client identity and upload limits that a client must not be able to talk its way around."""

import ast
import ipaddress
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

SERVER = Path(__file__).resolve().parents[1] / "server.py"
SOURCE = SERVER.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


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


class _HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(status_code, detail)
        self.status_code = status_code


def _request(peer, headers):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


class ClientIpTest(unittest.TestCase):
    def setUp(self):
        nets = [ipaddress.ip_network("127.0.0.1/32")]
        self.ns = _load(("_peer_in_trusted", "get_client_ip"),
                        {"ipaddress": ipaddress, "HARDEN": {"TRUSTED_PROXY_NETS": nets}, "Request": object})

    def test_forged_headers_from_untrusted_peer_are_ignored(self):
        req = _request("203.0.113.5", {"cf-connecting-ip": "10.0.0.1", "x-forwarded-for": "10.0.0.2"})
        self.assertEqual(self.ns["get_client_ip"](req), "203.0.113.5")

    def test_headers_from_trusted_proxy_are_used(self):
        req = _request("127.0.0.1", {"x-forwarded-for": "198.51.100.7, 127.0.0.1"})
        self.assertEqual(self.ns["get_client_ip"](req), "198.51.100.7")

    def test_no_trusted_proxy_means_socket_address(self):
        self.ns["HARDEN"]["TRUSTED_PROXY_NETS"] = []
        req = _request("203.0.113.5", {"cf-connecting-ip": "10.0.0.1"})
        self.assertEqual(self.ns["get_client_ip"](req), "203.0.113.5")


class SourceChecksTest(unittest.TestCase):
    def test_local_steps_clamped_to_assigned_steps(self):
        self.assertIn('task_meta.get("targetLocalSteps")', SOURCE)

    def test_chunks_checked_against_declared_sizes(self):
        body = ast.unparse(_function("submit_chunk"))
        self.assertIn("Chunk larger than declared chunkSize", body)
        self.assertIn("Upload larger than declared totalBytes", body)

    def test_finalize_requires_granted_object_key(self):
        self.assertIn("object_key_mismatch", ast.unparse(_function("submit_finalize")))


class BanUserTest(unittest.TestCase):
    def setUp(self):
        self.ns = {
            "time": time, "datetime": __import__("datetime").datetime,
            "timezone": __import__("datetime").timezone,
            "HTTPException": _HTTPException, "Request": object,
            "_admin_ok": lambda s: s == "ok", "_admin_secret": lambda r, s: s,
            "users_lock": threading.Lock(), "user_countries_lock": threading.Lock(),
            "users_db": {"users": {"u1": {"username": "mallory"}, "u2": {"username": "alice"}},
                         "tokens": {"t1": "u1", "t2": "u1", "t3": "u2"}},
            "user_countries": {"user:u1": {"last_ip": "198.51.100.9"}},
            "banned_users": {}, "banned_client_keys": {}, "subnets": [],
            "save_users_db": lambda: None,
            "PUB": SimpleNamespace(event=lambda *a, **k: None, who=lambda u: u),
            "log": SimpleNamespace(warning=lambda *a, **k: None),
        }
        self.ns["ban_subnet"] = lambda ip, why, hours: self.ns["subnets"].append(ip)
        _load(("admin_ban_user",), self.ns)

    def test_ban_revokes_tokens_and_bans_last_ip(self):
        out = self.ns["admin_ban_user"](secret="ok", user_id="u1", ban_ip=True, purge_pending=False)
        self.assertEqual(self.ns["users_db"]["tokens"], {"t3": "u2"})
        self.assertEqual(self.ns["subnets"], ["198.51.100.9"])
        self.assertEqual(out["ips_banned"], ["198.51.100.9"])
        self.assertIn("u1", self.ns["banned_users"])


if __name__ == "__main__":
    unittest.main()
