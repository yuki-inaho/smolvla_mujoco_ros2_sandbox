"""A-8 contact probe: runs and reports retention metrics.

This is feasibility evidence, not a quality gate: the assertions only fix the
contract of the probe output (it must run and report the metrics), so a low
retention result is still a passing test.
"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "contact_poc.py"


def _module():
    spec = importlib.util.spec_from_file_location("contact_poc_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contact_poc_reports_retention_metrics():
    module = _module()
    result = module.run(module.DEFAULT_SCENE, module.DEFAULT_MODEL, 0.2, 0.3)
    assert {
        "retained",
        "slip_m",
        "offset_start_m",
        "offset_end_m",
        "fruit_z_start_m",
        "fruit_z_end_m",
        "max_penetration_m",
    } <= set(result)
    assert isinstance(result["retained"], bool)
