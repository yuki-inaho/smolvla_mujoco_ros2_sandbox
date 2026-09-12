#!/usr/bin/env python3
"""Export a standalone, dependency-free HTML viewer for the harvest simulation.

Layout
  scripts/export_viewer_html.py   generator (this file, pure Python, no ROS)
  artifacts/viewer/harvest_sim.html   single-file artifact (data + video embedded)

The HTML file replays the same phase machine as ``harvest_sim_node`` in
vanilla JavaScript (no CDN, no build step) so it can be opened directly from
disk. Fruits/stems/trajectory come from ``scene.json``; the 30 s render can be
embedded as a base64 video for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2_ws" / "src" / "tomato_harvest_sim"
DEFAULT_SCENE = (
    PKG / "assets" / "scenes" / "7a7e56ff__camera_l__2026-08-04_08-19-43-646" / "scene.json"
)
DEFAULT_VIDEO = ROOT / "artifacts" / "video" / "harvest_realtime_30s_faithful.mp4"
DEFAULT_OUT = ROOT / "artifacts" / "viewer" / "harvest_sim.html"
HOME_JOINTS = [0.30, 0.4363323129985824, 0.0, -0.4363323129985824]
DUNK_JOINTS = [0.003, 0.5934119456780721, -0.6108652381980153, 0.017453292519943295]

TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1117; color: #cfd8dc; font: 13px/1.5 ui-monospace, monospace; }
  header { padding: 8px 14px; background: #161d26; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; color: #9fd3ff; }
  #hud { white-space: pre; }
  #controls { padding: 8px 14px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; background: #121821; }
  #controls label { display: flex; gap: 4px; align-items: center; }
  button { background: #22303f; color: #cfd8dc; border: 1px solid #35485c; border-radius: 4px; padding: 4px 10px; cursor: pointer; }
  button:hover { background: #2c3d50; }
  #layout { display: flex; gap: 10px; padding: 10px 14px; align-items: flex-start; flex-wrap: wrap; }
  canvas { background: #080c10; border: 1px solid #1d2833; border-radius: 6px; }
  #side { display: flex; flex-direction: column; gap: 8px; }
  video { width: 480px; max-width: 100%; background: #000; border: 1px solid #1d2833; border-radius: 6px; }
  input[type=range] { width: 320px; }
  .note { color: #7d8b99; padding: 0 14px 14px; }
</style>
</head>
<body>
<header>
  <h1>tomato harvest simulation</h1>
  <div id="hud">loading...</div>
</header>
<div id="controls">
  <button id="play">pause</button>
  <label>speed <select id="speed">
    <option value="0.25">0.25x</option>
    <option value="0.5">0.5x</option>
    <option value="1" selected>1x</option>
    <option value="2">2x</option>
  </select></label>
  <label>t <input id="scrub" type="range" min="0" max="1000" value="0"></label>
  <label>yaw <input id="yaw" type="range" min="0" max="360" value="125"></label>
  <label><input id="loop" type="checkbox" checked> loop</label>
  <label><input id="sync" type="checkbox"> video sync</label>
</div>
<div id="layout">
  <canvas id="view" width="960" height="640"></canvas>
  <div id="side"><video id="video" controls loop src="__VIDEO_SRC__"></video></div>
</div>
<div class="note">__NOTE__</div>
<script id="harvest-data" type="application/json">__HARVEST_DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("harvest-data").textContent);
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const hud = document.getElementById("hud");
const video = document.getElementById("video");
const scrub = document.getElementById("scrub");
const playButton = document.getElementById("play");
const speedSelect = document.getElementById("speed");
const loopBox = document.getElementById("loop");
const syncBox = document.getElementById("sync");
if (!D.has_video) { video.style.display = "none"; syncBox.parentElement.style.display = "none"; }
const HOME = D.joints.home, DUNK = D.joints.dunk;
const VMAX = D.joints.vmax, AMAX = D.joints.amax;
const T = D.timing, C = D.constants;

function trapezoidTime(delta, vmax, amax) {
  const s = Math.abs(delta);
  if (s === 0) return 0;
  const ramp = vmax / amax, rampDist = 0.5 * amax * ramp * ramp;
  if (s <= 2 * rampDist) return 2 * Math.sqrt(s / amax);
  return 2 * ramp + (s - 2 * rampDist) / vmax;
}
function minDuration(start, target) {
  let m = 0;
  for (let i = 0; i < 4; i++) m = Math.max(m, trapezoidTime(target[i] - start[i], VMAX[i], AMAX[i]));
  return m;
}
function trapPos(delta, t, duration, vmax, amax) {
  const sign = delta >= 0 ? 1 : -1, dist = Math.abs(delta);
  if (dist === 0) return delta;
  if (duration <= 0) throw new Error("infeasible duration");
  const minimum = trapezoidTime(delta, vmax, amax);
  if (duration < minimum - 1e-9 * (1 + minimum)) throw new Error("infeasible duration");
  t = Math.min(Math.max(t, 0), duration);
  const disc = Math.pow(duration * amax, 2) - 4 * dist * amax;
  let peak = (duration * amax - Math.sqrt(Math.max(disc, 0))) / 2;
  if (peak <= 0) return delta;
  peak = Math.min(peak, vmax);
  const ramp = peak / amax, rampDist = 0.5 * amax * ramp * ramp;
  if (t < ramp) return sign * 0.5 * amax * t * t;
  if (t > duration - ramp) return sign * (dist - 0.5 * amax * Math.pow(duration - t, 2));
  return sign * (rampDist + peak * (t - ramp));
}
function jointPos(start, target, t, duration) {
  if (duration === undefined) duration = minDuration(start, target);
  return start.map((s, i) => s + trapPos(target[i] - s[i], t, duration, VMAX[i], AMAX[i]));
}
const shakeStep = trapezoidTime(T.shake_amp, T.shake_v, T.shake_a);
const dunkMove = minDuration(HOME, DUNK);
const dunkShake = T.shake_count * 2 * shakeStep;
const dunkDuration = 2 * dunkMove + T.dunk_release + dunkShake + T.dunk_settle;
function shakeOffset(t) {
  if (shakeStep <= 0) return 0;
  const index = Math.floor(t / shakeStep);
  if (index >= T.shake_count * 2) return 0;
  const position = trapPos(T.shake_amp, t - index * shakeStep, shakeStep, T.shake_v, T.shake_a);
  return index % 2 === 0 ? position : T.shake_amp - position;
}
function buildSchedule() {
  const cycles = Math.max(8, D.dunk_every);
  const schedule = [];
  for (let i = 0; i < cycles; i++) {
    const isDunk = D.dunk_every > 0 && (i + 1) % D.dunk_every === 0;
    const phases = [
      ["APPROACH", D.trajectory.duration, false],
      ["CLOSE", C.close_duration, false],
      ["LIFT", D.trajectory.duration, false],
      ["UNLOAD", isDunk ? dunkDuration : C.unload_hold, isDunk],
      ["DONE", C.done_hold, false],
    ];
    schedule.push({ index: i, is_dunk: isDunk, duration: phases.reduce((a, p) => a + p[1], 0), phases });
  }
  return schedule;
}
const schedule = buildSchedule();
const block = schedule.reduce((a, c) => a + c.duration, 0);
function sampleTrajectory(t) {
  const times = D.trajectory.times, values = D.trajectory.joints;
  const last = times.length - 1;
  if (t <= times[0]) return values[0].slice();
  if (t >= times[last]) return values[last].slice();
  let lo = 0, hi = last;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (times[mid] <= t) lo = mid; else hi = mid; }
  const f = (t - times[lo]) / (times[hi] - times[lo]);
  return values[lo].map((v, i) => v + (values[hi][i] - v) * f);
}
function motionState(elapsedTotal) {
  let t = ((elapsedTotal % block) + block) % block;
  let cycle = schedule[schedule.length - 1];
  for (const candidate of schedule) {
    if (t < candidate.duration) { cycle = candidate; break; }
    t -= candidate.duration;
  }
  for (const [phase, duration, isDunk] of cycle.phases) {
    if (t >= duration) { t -= duration; continue; }
    if (phase === "APPROACH") return { phase, joints: sampleTrajectory(t), gripper: 0 };
    if (phase === "CLOSE") {
      const last = sampleTrajectory(D.trajectory.duration);
      const ratio = duration > 0 ? Math.min(Math.max(t / duration, 0), 1) : 1;
      return { phase, joints: last, gripper: C.gripper_closed * ratio };
    }
    if (phase === "LIFT") {
      const playback = Math.max(D.trajectory.duration - t, 0);
      return { phase, joints: sampleTrajectory(playback), gripper: C.gripper_closed };
    }
    if (phase === "UNLOAD") {
      if (!isDunk) return { phase, joints: HOME.slice(), gripper: C.gripper_closed };
      if (t < dunkMove) return { phase, joints: jointPos(HOME, DUNK, t, dunkMove), gripper: C.gripper_closed };
      t -= dunkMove;
      if (t < T.dunk_release) {
        const ratio = T.dunk_release > 0 ? Math.min(Math.max(t / T.dunk_release, 0), 1) : 1;
        return { phase, joints: DUNK.slice(), gripper: C.gripper_closed * (1 - ratio) };
      }
      t -= T.dunk_release;
      if (t < dunkShake) {
        const joints = DUNK.slice();
        joints[0] = Math.max(DUNK[0] + shakeOffset(t), 0);
        return { phase, joints, gripper: 0 };
      }
      t -= dunkShake;
      if (t < T.dunk_settle) return { phase, joints: DUNK.slice(), gripper: 0 };
      t -= T.dunk_settle;
      return { phase, joints: jointPos(DUNK, HOME, t, dunkMove), gripper: 0 };
    }
    if (cycle.is_dunk) return { phase: "DONE", joints: HOME.slice(), gripper: 0 };
    const ratio = duration > 0 ? Math.min(Math.max(t / duration, 0), 1) : 1;
    return { phase: "DONE", joints: HOME.slice(), gripper: C.gripper_closed * (1 - ratio) };
  }
  return { phase: "DONE", joints: HOME.slice(), gripper: 0 };
}
function rot2(angle, length) { return [length * Math.cos(angle), -length * Math.sin(angle)]; }
function armPoints(j) {
  const shoulder = [0.195, 0, 0.96 + j[0]];
  const e1 = rot2(j[1], 0.285), e2 = rot2(j[1] + j[2], -0.345);
  const elbow = [shoulder[0] + e1[0], shoulder[1] + e1[1], shoulder[2]];
  const wrist = [elbow[0] + e2[0], elbow[1] + e2[1], shoulder[2]];
  const sum = j[1] + j[2] + j[3];
  const handOffset = rot2(sum, 0.05), graspOffset = rot2(sum, 0.16);
  const hand = [wrist[0] + handOffset[0], wrist[1] + handOffset[1], wrist[2] - 0.07];
  const grasp = [hand[0] + graspOffset[0], hand[1] + graspOffset[1], hand[2]];
  return { shoulder, elbow, wrist, hand, grasp, yaw: sum };
}
let elapsed = 0, playing = true, lastFrame = null, yawDeg = 125;
let ready = false;
window.__harvestSim = { ready: false, phase: "", elapsed: 0, block };
function project(p) {
  const a = yawDeg * Math.PI / 180;
  const x = p[0] * Math.cos(a) - p[1] * Math.sin(a);
  const y = p[0] * Math.sin(a) + p[1] * Math.cos(a);
  return [canvas.width / 2 + x * 300, canvas.height * 0.66 + y * 150 - (p[2] - 1.0) * 260];
}
function segment(a, b, color, width) {
  const pa = project(a), pb = project(b);
  ctx.strokeStyle = color; ctx.lineWidth = width || 2;
  ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
}
function draw(state) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#7b8a99";
  for (const stem of D.stems) { const p = project(stem); ctx.fillRect(p[0], p[1], 1.5, 1.5); }
  for (const fruit of D.fruits) {
    const p = project(fruit);
    ctx.fillStyle = fruit[3] >= 1 ? "#ff5252" : "#7cb342";
    ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill();
  }
  const target = project(D.target);
  ctx.strokeStyle = "#ffd54f"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(target[0], target[1], 12, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = "rgba(79,195,247,0.55)"; ctx.lineWidth = 1.2; ctx.beginPath();
  D.trajectory.eef.forEach((p, i) => { const q = project(p); if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]); });
  ctx.stroke();
  const arm = armPoints(state.joints);
  segment(arm.shoulder, arm.elbow, "#4f9cf7", 5);
  segment(arm.elbow, arm.wrist, "#4f9cf7", 4);
  segment(arm.wrist, arm.hand, "#8ab4f8", 4);
  segment(arm.hand, arm.grasp, "#c8d6e5", 3);
  const grip = 0.09 - state.gripper * (0.09 - 0.0057) / 0.0843;
  const perp = [Math.sin(arm.yaw), Math.cos(arm.yaw)];
  for (const s of [1, -1]) {
    const a = [arm.grasp[0] + perp[0] * grip / 2 * s, arm.grasp[1] + perp[1] * grip / 2 * s, arm.grasp[2]];
    const b = [a[0] + (arm.grasp[0] - arm.wrist[0]) * 0.25, a[1] + (arm.grasp[1] - arm.wrist[1]) * 0.25, a[2]];
    segment(a, b, "#ff8a65", 3);
  }
  hud.textContent = `${state.phase}  picked=${["LIFT", "UNLOAD", "DONE"].includes(state.phase)}  t=${(elapsed % block).toFixed(2)}s / ${block.toFixed(2)}s  block#${Math.floor(elapsed / block)}`;
  window.__harvestSim = { ready: true, phase: state.phase, elapsed: elapsed % block, block };
}
function frame(now) {
  if (lastFrame === null) lastFrame = now;
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  const speed = parseFloat(speedSelect.value);
  if (playing) elapsed += dt * speed;
  const state = motionState(elapsed);
  if (syncBox.checked && D.video && video.duration) {
    video.currentTime = (elapsed % block) % video.duration;
    if (playing && video.paused) video.play().catch(() => {});
    if (!playing && !video.paused) video.pause();
  }
  scrub.value = String(Math.round(((elapsed % block) / block) * 1000));
  draw(state);
  requestAnimationFrame(frame);
}
playButton.addEventListener("click", () => { playing = !playing; playButton.textContent = playing ? "pause" : "play"; });
scrub.addEventListener("input", () => { elapsed = (Number(scrub.value) / 1000) * block; lastFrame = null; });
document.getElementById("yaw").addEventListener("input", (event) => { yawDeg = Number(event.target.value); });
loopBox.addEventListener("change", () => { if (!loopBox.checked) playing = false, playButton.textContent = "play"; });
ready = true;
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def build_payload(scene: dict, dunk_every: int, stems_stride: int, trajectory_stride: int) -> dict:
    stems = scene.get("stems", [])
    trajectory = scene["trajectory"]
    times = [round(float(entry["time_from_start_s"]), 5) for entry in trajectory[::trajectory_stride]]
    if times[-1] != float(trajectory[-1]["time_from_start_s"]):
        times.append(round(float(trajectory[-1]["time_from_start_s"]), 5))
    joints = [
        [round(float(entry["joints"][key]), 6) for key in ("lift", "shoulder_yaw", "elbow_yaw", "wrist_yaw")]
        for entry in trajectory[::trajectory_stride]
    ]
    if len(joints) != len(times):
        last_joints = trajectory[-1]["joints"]
        joints.append(
            [round(float(last_joints[key]), 6) for key in ("lift", "shoulder_yaw", "elbow_yaw", "wrist_yaw")]
        )
    eef = [
        [round(float(value), 5) for value in entry["eef"]]
        for entry in trajectory[::trajectory_stride]
    ]
    if len(eef) != len(times):
        eef.append([round(float(value), 5) for value in trajectory[-1]["eef"]])
    return {
        "scene_id": scene.get("scene_id"),
        "dunk_every": int(dunk_every),
        "fruits": [
            [round(float(value), 4) for value in fruit["position"]] + [int(fruit["ripeness"])]
            for fruit in scene.get("fruits", [])
        ],
        "stems": [
            [round(float(value), 4) for value in point]
            for point in stems[::stems_stride]
        ],
        "target": [round(float(value), 5) for value in scene["target"]["position"]],
        "trajectory": {"times": times, "joints": joints, "eef": eef,
                       "duration": round(times[-1], 5)},
        "joints": {
            "home": [round(value, 6) for value in HOME_JOINTS],
            "dunk": [round(value, 6) for value in DUNK_JOINTS],
            "vmax": [0.5, 6.0, 6.0, 6.0],
            "amax": [8.0, 40.0, 40.0, 40.0],
        },
        "timing": {
            "dunk_release": 0.4, "dunk_settle": 0.3,
            "shake_count": 2, "shake_amp": 0.024, "shake_v": 0.42, "shake_a": 8.0,
        },
        "constants": {
            "close_duration": 0.4, "unload_hold": 0.3, "done_hold": 0.2,
            "gripper_closed": 0.0843,
        },
    }


def render_html(payload: dict, video_path: Path | None, title: str) -> str:
    video_uri = ""
    if video_path is not None and video_path.exists():
        encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
        video_uri = f"data:video/mp4;base64,{encoded}"
    payload = dict(payload, has_video=bool(video_uri))
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    note = (
        f"generated {datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M')} JST / "
        f"scene={payload.get('scene_id')} / dunk_every={payload['dunk_every']} / "
        f"{'video embedded' if video_uri else 'no video'}"
    )
    return (
        TEMPLATE.replace("__TITLE__", title)
        .replace("__NOTE__", note)
        .replace("__HARVEST_DATA__", data)
        .replace("__VIDEO_SRC__", video_uri)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dunk-every", type=int, default=3)
    parser.add_argument("--stems-stride", type=int, default=4)
    parser.add_argument("--trajectory-stride", type=int, default=2)
    args = parser.parse_args()

    scene = json.loads(args.scene.read_text())
    payload = build_payload(scene, args.dunk_every, args.stems_stride, args.trajectory_stride)
    video_path = None if args.no_video else args.video
    html = render_html(payload, video_path, "tomato harvest sim viewer")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    size = args.out.stat().st_size
    print(
        f"viewer written: {args.out} ({size:,} B, "
        f"fruits={len(payload['fruits'])}, stems={len(payload['stems'])}, "
        f"trajectory={len(payload['trajectory']['times'])} samples)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
