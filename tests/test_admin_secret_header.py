"""Regression tests for the admin secret being readable from a header.

`?secret=` puts the coordinator's admin credential into every proxy access log,
the operator's browser history, any `Referer` the admin page emits, and shell
history. `_admin_secret(request, secret)` makes a header the supported form
while keeping the query parameter as a deprecated fallback:

    X-Admin-Secret: <secret>
    Authorization: Bearer <secret>

The functions under test are lifted out of ``server.py`` with ``ast`` and run in
an isolated namespace, because importing server.py pulls in torch, FastAPI
start-up, Hugging Face and background state. Only the standard library is
needed for the extraction tests; the FastAPI injection test skips itself when
FastAPI is not installed.

Run:  python3 -m unittest discover -s tests -v
"""
import ast
import secrets
import unittest
from pathlib import Path
from typing import Optional

SERVER_PY = Path(__file__).resolve().parent.parent / "server.py"
SECRET = "unit-test-admin-secret"


def _extract(*names):
    """Compile the named top-level functions out of server.py verbatim."""
    tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            module = ast.Module(body=[node], type_ignores=[])
            code = compile(ast.fix_missing_locations(module), str(SERVER_PY), "exec")
            namespace = {"Optional": Optional, "Request": object, "secrets": secrets,
                         "ADMIN_SECRET": SECRET}
            exec(code, namespace)
            found[node.name] = namespace[node.name]
    missing = set(names) - set(found)
    if missing:
        raise AssertionError(f"server.py no longer defines: {sorted(missing)}")
    return found


class _Headers(dict):
    """Case-insensitive mapping, like Starlette's Headers."""

    def __init__(self, **items):
        super().__init__({k.lower(): v for k, v in items.items()})

    def get(self, key, default=None):
        return super().get(key.lower(), default)


class _Request:
    def __init__(self, **headers):
        self.headers = _Headers(**headers)


class AdminSecretResolutionTests(unittest.TestCase):
    def setUp(self):
        # Instance attributes, not class attributes: a function stored on the
        # class would be bound as a method and receive ``self`` as ``request``.
        fn = _extract("_admin_secret", "_admin_ok")
        self.admin_secret = fn["_admin_secret"]
        self.admin_ok = fn["_admin_ok"]

    def test_header_secret_is_used(self):
        request = _Request(**{"X-Admin-Secret": SECRET})
        self.assertEqual(self.admin_secret(request), SECRET)
        self.assertTrue(self.admin_ok(self.admin_secret(request)))

    def test_header_secret_is_case_insensitive(self):
        request = _Request(**{"x-adMIN-sEcret": SECRET})
        self.assertEqual(self.admin_secret(request), SECRET)

    def test_header_wins_over_a_stale_query_parameter(self):
        request = _Request(**{"X-Admin-Secret": SECRET})
        self.assertEqual(self.admin_secret(request, "stale-bookmark"), SECRET)

    def test_bearer_token_is_accepted(self):
        request = _Request(Authorization=f"Bearer {SECRET}")
        self.assertEqual(self.admin_secret(request), SECRET)
        request = _Request(Authorization=f"bearer   {SECRET}  ")
        self.assertEqual(self.admin_secret(request), SECRET)

    def test_non_bearer_authorization_is_ignored(self):
        request = _Request(Authorization=f"Basic {SECRET}")
        self.assertEqual(self.admin_secret(request, ""), "")

    def test_query_parameter_still_works(self):
        self.assertEqual(self.admin_secret(_Request(), SECRET), SECRET)
        self.assertTrue(self.admin_ok(self.admin_secret(_Request(), SECRET)))

    def test_empty_header_falls_back_to_the_query_parameter(self):
        request = _Request(**{"X-Admin-Secret": "   "})
        self.assertEqual(self.admin_secret(request, SECRET), SECRET)

    def test_no_credential_anywhere_is_falsy(self):
        self.assertEqual(self.admin_secret(_Request()), "")
        self.assertFalse(self.admin_ok(self.admin_secret(_Request())))
        self.assertFalse(self.admin_ok(self.admin_secret(None)))

    def test_direct_call_without_a_request_still_checks_the_query_parameter(self):
        # Tests and scripts call the endpoint functions directly, with no ASGI
        # request in hand; that path must keep authenticating on `secret=`.
        self.assertEqual(self.admin_secret(None, SECRET), SECRET)
        self.assertTrue(self.admin_ok(self.admin_secret(None, SECRET)))

    def test_wrong_secret_is_rejected(self):
        request = _Request(**{"X-Admin-Secret": "not-the-secret"})
        self.assertFalse(self.admin_ok(self.admin_secret(request)))


class AdminEndpointCoverageTests(unittest.TestCase):
    """Every endpoint that takes a secret must resolve it through the header."""

    def setUp(self):
        self.tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"))

    @staticmethod
    def _params(node):
        return [a.arg for a in node.args.args]

    def test_no_endpoint_checks_the_bare_query_parameter(self):
        offenders = []
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "_admin_ok" and len(call.args) == 1
                        and isinstance(call.args[0], ast.Name)
                        and call.args[0].id == "secret"):
                    offenders.append(node.name)
        self.assertEqual(offenders, [], "these endpoints ignore the admin secret header")

    def test_every_secret_endpoint_accepts_a_request(self):
        missing = []
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = self._params(node)
            defaults = node.args.defaults
            named = list(zip(params[len(params) - len(defaults):], defaults)) if defaults else []
            if not any(name == "secret" and isinstance(default, ast.Constant)
                       and default.value == "" for name, default in named):
                continue  # a helper, not a route handler taking `?secret=`
            if "request" not in params:
                missing.append(node.name)
        self.assertEqual(missing, [], "these endpoints cannot read the admin secret header")

    def test_admin_secret_is_resolved_on_every_guarded_endpoint(self):
        users = 0
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "_admin_secret"):
                    users += 1
                    break
        self.assertGreaterEqual(users, 20, f"only {users} endpoints resolve the admin secret")


class FastApiInjectionTests(unittest.TestCase):
    """`request: Request = None` must still be injected by FastAPI.

    The default exists only so tests and scripts can call the endpoint functions
    directly; at runtime the request must always be provided, otherwise the
    header path would silently never run.
    """

    def test_request_is_injected_despite_the_default(self):
        try:
            from fastapi import FastAPI, Request
            from fastapi.testclient import TestClient
        except Exception:  # pragma: no cover - fastapi is a server dependency
            self.skipTest("fastapi[testclient] not installed")

        app = FastAPI()

        @app.get("/probe")
        async def probe(request: Request = None, secret: str = ""):
            seen = request is not None
            return {"seen": seen, "header": request.headers.get("x-admin-secret") if seen else None}

        client = TestClient(app)
        body = client.get("/probe", headers={"X-Admin-Secret": "abc"}).json()
        self.assertTrue(body["seen"], "FastAPI did not inject Request when it had a default")
        self.assertEqual(body["header"], "abc")


if __name__ == "__main__":
    unittest.main()
