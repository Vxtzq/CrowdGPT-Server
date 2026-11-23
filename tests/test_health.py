"""The liveness route, so a monitor cannot mistake a slow round for a dead box."""

import asyncio
import unittest

try:  # the CI job is standard-library only
    import torch  # noqa: F401
    import server
except ImportError as _missing:  # pragma: no cover
    raise unittest.SkipTest(f"needs torch and server: {_missing}")


class HealthTest(unittest.TestCase):
    def test_health_is_registered(self):
        paths = {getattr(r, "path", "") for r in server.app.routes}
        self.assertIn("/health", paths)

    def test_health_answers_without_touching_the_model(self):
        body = asyncio.run(server.health())
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["modelSize"], server.MODEL_SIZE)
        self.assertGreaterEqual(body["uptimeSeconds"], 0.0)
        self.assertIsInstance(body["aggregating"], bool)
        self.assertIsInstance(body["round"], int)

    def test_health_does_not_report_a_round_as_zero_when_one_is_running(self):
        saved = dict(server.round_state)
        try:
            server.round_state["current_round"] = 7
            server.round_state["is_aggregating"] = True
            body = asyncio.run(server.health())
            self.assertEqual(body["round"], 7)
            self.assertTrue(body["aggregating"])
        finally:
            server.round_state.clear()
            server.round_state.update(saved)


if __name__ == "__main__":
    unittest.main()
