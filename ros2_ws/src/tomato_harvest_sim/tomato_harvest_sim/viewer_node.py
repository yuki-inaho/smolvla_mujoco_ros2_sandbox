"""Browser viewer for the tomato harvest simulation.

Serves a canvas 2D projection page plus two JSON endpoints:
  GET /api/state -> latest /sim/status merged with scene-derived counts
  GET /api/scene -> point cloud / fruits / target / eef path for projection

The page exposes ``window.__harvestState`` (updated on every poll) so that
``playwright-cli eval`` can inspect it during automated checks.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

INDEX_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>tomato harvest viewer</title>
<style>
  body { margin: 0; background: #101418; color: #cfd8dc; font: 13px/1.5 monospace; }
  header { padding: 8px 12px; background: #182028; }
  canvas { display: block; margin: 0 auto; background: #0b0f13; }
</style>
</head>
<body>
<header id="hud">loading...</header>
<canvas id="view" width="900" height="640"></canvas>
<script>
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const hud = document.getElementById('hud');
let scene = null;
let state = null;
window.__harvestState = null;

function project(p, cx, cy, s) {
  const [x, y, z] = p;
  const sx = cx + (x - y) * s;
  const sy = cy + (x + y) * s * 0.45 - z * s;
  return [sx, sy];
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!scene) { requestAnimationFrame(draw); return; }
  const cx = canvas.width / 2, cy = canvas.height * 0.62, s = 320;
  ctx.fillStyle = '#3b5568';
  for (const stem of scene.stems) {
    const [sx, sy] = project(stem, cx, cy, s);
    ctx.fillRect(sx, sy, 1.4, 1.4);
  }
  for (const fruit of scene.fruits) {
    const [sx, sy] = project(fruit.position, cx, cy, s);
    ctx.fillStyle = (fruit.ripeness >= 2) ? '#ff5252' : '#7cb342';
    ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.fill();
  }
  if (scene.target && scene.target.position) {
    const [sx, sy] = project(scene.target.position, cx, cy, s);
    ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(sx, sy, 14, 0, Math.PI * 2); ctx.stroke();
  }
  if (scene.eef_path && scene.eef_path.length) {
    ctx.strokeStyle = '#4fc3f7'; ctx.lineWidth = 1.5; ctx.beginPath();
    scene.eef_path.forEach((p, i) => {
      const [sx, sy] = project(p, cx, cy, s);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  }
  if (state && state.eef) {
    const [sx, sy] = project(state.eef, cx, cy, s);
    ctx.fillStyle = '#00e676';
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, Math.PI * 2); ctx.fill();
  }
  requestAnimationFrame(draw);
}

async function poll() {
  try {
    if (!scene) scene = await (await fetch('/api/scene')).json();
    state = await (await fetch('/api/state')).json();
    window.__harvestState = state;
    hud.textContent = JSON.stringify(state);
  } catch (err) { hud.textContent = 'error: ' + err; }
}
setInterval(poll, 250);
poll();
draw();
</script>
</body>
</html>
"""


def load_scene(path: str) -> dict:
    scene_path = Path(path).expanduser()
    if not scene_path.exists():
        raise FileNotFoundError(f"scene.json not found: {scene_path}")
    return json.loads(scene_path.read_text())


def build_state(status: dict, scene: dict) -> dict:
    stems = scene.get("stems", []) or []
    fruits = scene.get("fruits", []) or []
    target = scene.get("target") or {}
    eef = status.get("eef")
    return {
        "phase": status.get("phase", "IDLE"),
        "picked": bool(status.get("picked", False)),
        "frame_index": int(status.get("frame_index", 0) or 0),
        "eef": list(eef) if eef is not None else None,
        "target": status.get("target") or target.get("position"),
        "points_count": len(stems) + len(fruits) + (1 if target else 0),
        "updated_at": time.time(),
    }


def make_scene_payload(scene: dict, max_points: int = 20000) -> dict:
    stems = scene.get("stems", []) or []
    if max_points and len(stems) > max_points:
        stems = stems[:max_points]
    return {
        "scene_id": scene.get("scene_id"),
        "stems": stems,
        "fruits": scene.get("fruits", []),
        "target": scene.get("target", {}),
        "eef_path": scene.get("eef_path", []),
    }


def make_handler(node: ViewerNode):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/state"):
                self._json(node.state)
            elif self.path.startswith("/api/scene"):
                self._json(make_scene_payload(node.scene))
            elif self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def _json(self, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence access logs
            pass

    return Handler


class ViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("viewer_node")
        self.declare_parameter("scene_path", "")
        self.declare_parameter("port", 8765)
        scene_path = self.get_parameter("scene_path").get_parameter_value().string_value
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        if not scene_path:
            raise RuntimeError("scene_path parameter is required")
        self.scene = load_scene(scene_path)
        self.state = build_state({}, self.scene)
        self.create_subscription(String, "/sim/status", self._on_status, 10)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), make_handler(self))
        self.get_logger().info(f"viewer ready on http://127.0.0.1:{self.port}/")

    def _on_status(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warning("ignoring malformed /sim/status payload")
            return
        self.state = build_state(status, self.scene)

    def serve_forever(self) -> None:
        self.httpd.serve_forever()


def main(argv=None) -> int:
    rclpy.init(args=argv)
    node = None
    try:
        node = ViewerNode()
        thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        thread.start()
        node.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.httpd.shutdown()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
