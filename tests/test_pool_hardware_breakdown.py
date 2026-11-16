import ast
from pathlib import Path
import unittest


SERVER_PY = Path(__file__).resolve().parents[1] / "server.py"
FUNCTION_NAMES = ("_gpu_arch_class", "_pool_hardware_breakdown")


def _load_functions():
    tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"), filename=str(SERVER_PY))
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
    }
    missing = set(FUNCTION_NAMES) - set(functions)
    if missing:
        raise AssertionError(f"functions not found in server.py: {sorted(missing)}")
    module = ast.Module(body=[functions[name] for name in FUNCTION_NAMES], type_ignores=[])
    namespace = {}
    exec(compile(module, str(SERVER_PY), "exec"), namespace)
    return namespace


class PoolHardwareBreakdownTest(unittest.TestCase):
    def setUp(self):
        self.ns = _load_functions()

    def test_counts_supported_statuses_and_ignores_unknown_statuses(self):
        sig = ("CUDA", 8, "fp32", "adamw", "ada")
        pools = [
            {"hardware_sig": sig, "status": "forming", "clients": ["a"]},
            {"hardware_sig": sig, "status": "training", "clients": ["b", "c"]},
            {"hardware_sig": sig, "status": "idle", "clients": ["d"]},
            {"hardware_sig": sig, "status": "completing", "clients": ["e"]},
            {"hardware_sig": sig, "status": "unexpected", "clients": ["f"]},
            {"hardware_sig": ("CUDA",), "status": "training", "clients": ["ignored"]},
        ]
        self.assertEqual(self.ns["_pool_hardware_breakdown"](pools), {
            "CUDA-fp32-ada": {
                "forming": 1, "training": 1, "idle": 1, "completing": 1, "clients": 6,
            },
        })

    def test_classifies_intel_arc_for_hardware_grouping(self):
        self.assertEqual(
            self.ns["_gpu_arch_class"]("Intel(R) Arc(TM) A750 Graphics"),
            "intel_arc",
        )


if __name__ == "__main__":
    unittest.main()
