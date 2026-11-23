"""Pool activation must follow the operator's CROWDGPT_TARGET_LOCAL_STEPS setting."""

import ast
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "server.py").read_text()
TREE = ast.parse(SOURCE)


class PoolStepsTest(unittest.TestCase):
    def _activation_source(self):
        for node in TREE.body:
            if isinstance(node, ast.FunctionDef):
                text = ast.get_source_segment(SOURCE, node)
                if "BASE_STEPS" in text and "hw_multiplier" in text:
                    return text
        raise AssertionError("pool activation step sizing not found")

    def test_base_steps_come_from_config(self):
        text = self._activation_source()
        self.assertIn('BASE_STEPS = CROSS_CHECK["TARGET_LOCAL_STEPS"]', text)
        self.assertNotIn("BASE_STEPS = 500", text)

    def test_clamp_scales_with_base(self):
        text = self._activation_source()
        line = next(ln.strip() for ln in text.splitlines() if ln.strip().startswith("target_steps ="))
        ns = {}
        for base, vram, expected in ((500, 8.0, 500), (500, 1.0, 125), (500, 100.0, 2500), (4, 12.0, 6), (4, 1.0, 1)):
            ns.update(BASE_STEPS=base, scaled_steps=int(base * max(0.25, vram / 8.0)))
            exec(line, {}, ns)
            self.assertEqual(ns["target_steps"], expected, (base, vram))


if __name__ == "__main__":
    unittest.main()
