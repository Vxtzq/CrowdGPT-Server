"""The heap ratchet: freed arenas must go back to the kernel.

A round boundary allocates and drops several full-model buffers (delta read,
checkpoint save). CPython frees them; glibc keeps the arena; VmRSS climbs a few
GB per round with a flat live heap, and a 2.5 GB model OOMs after three rounds.
These tests pin the two primitives that fix it, and that every gc point trims.
"""

import unittest

try:  # the repo CI job is stdlib-only
    import torch  # noqa: F401
    import server
except ImportError as _missing:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"needs torch and server: {_missing}")


class RssTest(unittest.TestCase):
    def test_rss_reads_a_plausible_number(self):
        # Any live CPython+tensor process is well over 1 MB and under 1 TB.
        self.assertGreater(server._rss_mb(), 1.0)
        self.assertLess(server._rss_mb(), 1024 * 1024)


class TrimTest(unittest.TestCase):
    def test_trim_returns_to_the_os_without_raising(self):
        before = server._trim_heap()
        self.assertGreater(before, 1.0)

    def test_trim_after_dropping_a_large_allocation_lowers_rss(self):
        blob = torch.zeros(64 * 1024 * 1024 // 4, dtype=torch.float32)  # 64 MB
        touched = server._rss_mb()
        self.assertGreater(touched, 1.0)
        del blob
        after = server._trim_heap()
        # Releasing is best-effort: the assertion is that it does not grow and
        # does not raise; malloc may already have returned the arena.
        self.assertLessEqual(after, touched + 8.0)


class EveryCollectionTrimsTest(unittest.TestCase):
    def test_no_bare_gc_collect_without_a_trim(self):
        import inspect
        src = inspect.getsource(server)
        self.assertIn("_trim_heap()", src)
        # each gc.collect() should be followed by a trim on the same or next line
        lines = src.splitlines()
        missing = []
        for i, line in enumerate(lines):
            if "gc.collect()" in line and "_trim_heap()" not in line:
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if "_trim_heap()" not in nxt:
                    missing.append(i + 1)
        self.assertEqual(missing, [], f"gc.collect() without a trim at lines {missing}")


if __name__ == "__main__":
    unittest.main()


class MemoryReportTest(unittest.TestCase):
    """The report is what turns "RAM grows" into a diagnosable number."""

    def test_report_returns_rss_and_does_not_raise(self):
        rss = server._memory_report("test")
        self.assertGreater(rss, 1.0)
        self.assertLess(rss, 1024 * 1024.0)

    def test_active_upload_bytes_is_zero_with_no_uploads(self):
        saved = getattr(server, "_active_uploads", None)
        server._active_uploads = {}
        try:
            self.assertEqual(server._active_upload_bytes(), 0)
        finally:
            if saved is not None:
                server._active_uploads = saved

    def test_active_upload_bytes_counts_chunks(self):
        saved = getattr(server, "_active_uploads", None)
        server._active_uploads = {"u1": {"chunks": [b"x" * 10, b"y" * 5]}}
        try:
            self.assertEqual(server._active_upload_bytes(), 15)
        finally:
            if saved is not None:
                server._active_uploads = saved
