"""Unit tests for the cohort policy (stdlib only, so CI runs them everywhere)."""

import unittest

import pool_policy as P


class ResolveOverflowMaxTest(unittest.TestCase):
    def test_default_is_one_rescue_seat(self):
        self.assertEqual(P.resolve_overflow_max(None), 1)
        self.assertEqual(P.resolve_overflow_max(""), 1)
        self.assertEqual(P.resolve_overflow_max("   "), 1)

    def test_zero_disables_rescue_seats(self):
        self.assertEqual(P.resolve_overflow_max("0"), 0)
        self.assertEqual(P.resolve_overflow_max(0), 0)

    def test_negative_and_junk_fall_back_safely(self):
        self.assertEqual(P.resolve_overflow_max("-3"), 0)
        self.assertEqual(P.resolve_overflow_max("many"), 1)
        self.assertEqual(P.resolve_overflow_max(object()), 1)

    def test_it_is_clamped(self):
        self.assertEqual(P.resolve_overflow_max("2"), 2)
        self.assertEqual(P.resolve_overflow_max("999"), P.MAX_OVERFLOW)


class ResolveStarvedTest(unittest.TestCase):
    def test_default_is_ten_minutes(self):
        self.assertEqual(P.resolve_starved_s(None), 600)

    def test_zero_admits_immediately(self):
        self.assertEqual(P.resolve_starved_s("0"), 0)

    def test_junk_and_negatives(self):
        self.assertEqual(P.resolve_starved_s("soon"), 600)
        self.assertEqual(P.resolve_starved_s("-1"), 0)

    def test_clamped(self):
        self.assertEqual(P.resolve_starved_s("1e9"), P.MAX_STARVED_S)


class AdmitsTest(unittest.TestCase):
    def setUp(self):
        self.kw = dict(pool_size=2, overflow_max=1, starved_s=600)

    def test_below_target_always_admits(self):
        self.assertTrue(P.admits(1, 0.0, **self.kw))

    def test_at_target_requires_starvation(self):
        self.assertFalse(P.admits(2, 599.0, **self.kw))
        self.assertTrue(P.admits(2, 600.0, **self.kw))

    def test_hard_cap_refuses_even_a_starved_peer(self):
        self.assertFalse(P.admits(3, 10_000.0, **self.kw))

    def test_zero_overflow_means_the_cap_is_the_target(self):
        kw = dict(pool_size=2, overflow_max=0, starved_s=0)
        self.assertTrue(P.admits(1, 0.0, **kw))
        self.assertFalse(P.admits(2, 10_000.0, **kw))

    def test_starved_zero_admits_a_rescue_seat_at_once(self):
        kw = dict(pool_size=2, overflow_max=2, starved_s=0)
        self.assertTrue(P.admits(2, 0.0, **kw))
        self.assertTrue(P.admits(3, 0.0, **kw))
        self.assertFalse(P.admits(4, 0.0, **kw))


class ActivationTargetTest(unittest.TestCase):
    def _kw(self, **over):
        kw = dict(pool_size=2, overflow_max=1, starved_waiting=False,
                  forming_age_s=0.0, forming_timeout_s=1800.0)
        kw.update(over)
        return kw

    def test_target_is_the_pool_size(self):
        self.assertEqual(P.activation_target(**self._kw()), 2)

    def test_a_waiting_starved_peer_raises_the_bar_by_one(self):
        self.assertEqual(P.activation_target(**self._kw(starved_waiting=True)), 3)

    def test_no_overflow_leaves_the_bar_alone(self):
        self.assertEqual(
            P.activation_target(**self._kw(starved_waiting=True, overflow_max=0)), 2)

    def test_a_pool_that_has_waited_too_long_starts_anyway(self):
        self.assertEqual(
            P.activation_target(**self._kw(starved_waiting=True, forming_age_s=1800.0)), 2)

    def test_the_bar_never_exceeds_target_plus_one(self):
        self.assertEqual(
            P.activation_target(**self._kw(starved_waiting=True, overflow_max=5)), 3)


class DescribeTest(unittest.TestCase):
    def test_it_reports_target_and_hard_cap_separately(self):
        d = P.describe(pool_size=2, overflow_max=1, starved_s=600)
        self.assertEqual(d, {"poolSize": 2, "overflowMax": 1, "hardCap": 3,
                             "starvedAdmitSeconds": 600})

    def test_zero_overflow_reports_a_cap_equal_to_the_target(self):
        d = P.describe(pool_size=2, overflow_max=0, starved_s=0)
        self.assertEqual(d["hardCap"], 2)


if __name__ == "__main__":
    unittest.main()
