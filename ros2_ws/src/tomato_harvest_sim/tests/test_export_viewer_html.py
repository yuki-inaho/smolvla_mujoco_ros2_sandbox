"""Standalone HTML viewer export contract (self-contained, no CDN)."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "export_viewer_html.py"

SCENE = {
    "scene_id": "unit-scene",
    "stems": [[0.0, 0.0, 1.0], [0.1, 0.1, 1.1], [0.2, 0.2, 1.2], [0.3, 0.3, 1.3]],
    "fruits": [{"position": [0.1, 0.2, 1.2], "ripeness": 1, "cluster": 0}],
    "target": {"position": [0.2, 0.3, 1.25]},
    "trajectory": [
        {
            "time_from_start_s": 0.0,
            "joints": {
                "lift": 0.30,
                "shoulder_yaw": 0.10,
                "elbow_yaw": 0.00,
                "wrist_yaw": -0.10,
            },
            "eef": [0.30, 0.20, 1.10],
        },
        {
            "time_from_start_s": 1.0,
            "joints": {
                "lift": 0.20,
                "shoulder_yaw": 0.30,
                "elbow_yaw": 0.20,
                "wrist_yaw": -0.30,
            },
            "eef": [0.35, 0.25, 1.30],
        },
    ],
}


def _module():
    spec = importlib.util.spec_from_file_location(
        "export_viewer_html_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_payload_contract_and_downsampling():
    module = _module()
    payload = module.build_payload(SCENE, dunk_every=3, stems_stride=2, trajectory_stride=1)
    assert payload["scene_id"] == "unit-scene"
    assert len(payload["stems"]) == 2
    assert payload["fruits"][0][3] == 1
    assert payload["trajectory"]["duration"] == 1.0
    assert len(payload["trajectory"]["times"]) == 2
    assert payload["joints"]["home"][0] == 0.30
    assert payload["constants"]["gripper_closed"] == 0.0843


def test_render_html_is_self_contained_and_parses():
    module = _module()
    payload = module.build_payload(SCENE, dunk_every=3, stems_stride=1, trajectory_stride=1)
    html = module.render_html(payload, None, "unit viewer")
    assert "<canvas" in html
    assert html.rstrip().endswith("</html>")
    assert "__HARVEST_DATA__" not in html
    assert "__VIDEO_SRC__" not in html
    assert "requestAnimationFrame" in html
    assert "https://" not in html  # no CDN / external dependency
    marker = 'type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    data = json.loads(html[start:end])
    assert data["has_video"] is False
    assert data["trajectory"]["joints"][0][0] == 0.30
