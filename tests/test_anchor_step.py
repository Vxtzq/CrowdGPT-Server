"""Pure tests for the anchor cadence -- no torch, so the stdlib-only CI job runs them.

The anchor *step* is the deployment's, but "every N rounds" is arithmetic we own,
and it is the part that fails silently: an anchor configured but never fired looks
exactly like an anchor that is turned off. See anchor_step.py for the measurements
this feature exists for.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchor_step import MAX_EVERY, anchor_state, resolve_every, should_anchor  # noqa: E402


class ResolveEveryTest(unittest.TestCase):
    def test_absent_or_empty_falls_back_to_the_default(self):
        for raw in (None, "", "   ", "garbage", float("nan")):
            self.assertEqual(resolve_every(raw, default=7), 7)
        self.assertEqual(resolve_every(None), 0)

    def test_numbers_parse_and_negative_means_off(self):
        self.assertEqual(resolve_every("1"), 1)
        self.assertEqual(resolve_every(" 4 "), 4)
        self.assertEqual(resolve_every(-3), 0)
        self.assertEqual(resolve_every("2.9"), 2)      # truncates, never rounds up

    def test_a_typo_cannot_silently_mean_never(self):
        self.assertEqual(resolve_every(10**9), MAX_EVERY)


class ShouldAnchorTest(unittest.TestCase):
    def test_zero_is_off(self):
        for step in (0, 1, 2, 100):
            self.assertFalse(should_anchor(step, 0))

    def test_fires_on_multiples_of_the_cadence(self):
        fired = [s for s in range(1, 10) if should_anchor(s, 3)]
        self.assertEqual(fired, [3, 6, 9])

    def test_step_zero_is_never_an_anchor_round(self):
        self.assertFalse(should_anchor(0, 1))

    def test_every_one_fires_every_round(self):
        self.assertEqual([s for s in range(1, 5) if should_anchor(s, 1)], [1, 2, 3, 4])


class AnchorStateTest(unittest.TestCase):
    def test_no_cadence_is_disabled_and_not_a_warning(self):
        st = anchor_state(every=0, has_step=True)
        self.assertEqual(st["level"], "disabled")
        self.assertFalse(st["enabled"])
        self.assertIsNone(st["warning"])

    def test_cadence_without_a_step_function_is_misconfigured(self):
        st = anchor_state(every=2, has_step=False)
        self.assertEqual(st["level"], "misconfigured")
        self.assertFalse(st["enabled"])
        self.assertIn("greedy gap stays open", st["warning"])

    def test_cadence_and_step_together_report_what_they_did(self):
        st = anchor_state(every=2, has_step=True, rounds=3, last_loss=5.5)
        self.assertEqual(st["level"], "ok")
        self.assertTrue(st["enabled"])
        self.assertEqual((st["every"], st["rounds"], st["lastLoss"]), (2, 3, 5.5))
        self.assertIsNone(st["warning"])


if __name__ == "__main__":
    unittest.main()
