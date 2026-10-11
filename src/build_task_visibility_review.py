"""Build a compact page that explains the doorway occlusion-and-reach task."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


def _segment_box_interval(start: np.ndarray, end: np.ndarray,
                          lower: np.ndarray, upper: np.ndarray):
    """Return the segment parameter interval inside an axis-aligned box."""
    t_min, t_max = 0.0, 1.0
    delta = end - start
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return None
            continue
        a = (lower[axis] - start[axis]) / delta[axis]
        b = (upper[axis] - start[axis]) / delta[axis]
        t_min = max(t_min, min(a, b))
        t_max = min(t_max, max(a, b))
        if t_min > t_max:
            return None
    return t_min, t_max


def _map(x: float, z: float) -> tuple[float, float]:
    # SVG coordinates: 900 x 500, with x in [-0.9, 1.35], z in [0, 2.1].
    return 85 + (x + 0.9) * 350, 435 - z * 190


def _rect(x0: float, z0: float, x1: float, z1: float, **attrs) -> str:
    px0, py1 = _map(x0, z0)
    px1, py0 = _map(x1, z1)
    extra = " ".join(f'{key.replace("_", "-")}="{html.escape(str(value))}"'
                     for key, value in attrs.items())
    return f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{px1-px0:.1f}" height="{py1-py0:.1f}" {extra}/>'


def build(output: Path) -> dict:
    config = json.loads((output / "scene_config_snapshot.json").read_text())
    result = json.loads((output / "delivery_result.json").read_text())
    trace = json.loads((output / "delivery_trace.json").read_text())
    walls = {wall["name"]: wall for wall in config["world"]["walls"]}
    header = walls[config["visibility_model"]["expected_occluder"]]
    base = np.asarray(result["candidate_attempts"]["attempts"][0]["base_q"], dtype=float)
    camera_local = np.asarray(config["visibility_model"]["camera_translation_from_base"], dtype=float)
    c, s = np.cos(base[2]), np.sin(base[2])
    camera = np.array([base[0] + c * camera_local[0] - s * camera_local[1],
                       base[1] + s * camera_local[0] + c * camera_local[1],
                       camera_local[2]])
    target = np.asarray(config["objects"]["cup"]["center"], dtype=float)
    half = np.asarray(header["size"], dtype=float) / 2
    center = np.asarray(header["center"], dtype=float)
    interval = _segment_box_interval(camera, target, center - half, center + half)
    audit = {
        "camera_world": camera.tolist(),
        "cup_center_world": target.tolist(),
        "occluder": header["name"],
        "occluded": interval is not None,
        "ray_entry_t": interval[0] if interval else None,
        "ray_exit_t": interval[1] if interval else None,
        "base_in_room1_during_pick": all(
            row["base_in_room1"] for row in trace
            if row["phase"] in {"navigate_to_pickup", "arm_to_pregrasp", "approach_cup", "close_gripper", "lift_cup"}
        ),
        "ee_enters_room2_during_pick": any(
            row["ee_in_room2"] for row in trace
            if row["phase"] in {"arm_to_pregrasp", "approach_cup", "close_gripper", "lift_cup"}
        ),
        "trajectory_collision_free": result["collision_free"],
    }
    if not audit["occluded"]:
        raise ValueError("Configured base-camera ray does not intersect the expected wall header")
    if not audit["base_in_room1_during_pick"] or not audit["ee_enters_room2_during_pick"]:
        raise ValueError("The recorded trajectory does not demonstrate base-outside / end-effector-inside semantics")

    (output / "visibility_audit.json").write_text(json.dumps(audit, indent=2))

    x0, x1, z0, z1 = -0.9, 1.35, 0.0, 2.1
    room1 = _rect(x0, z0, -0.06, z1, fill="#e8f2fb")
    room2 = _rect(0.06, z0, x1, z1, fill="#eef7ee")
    wall_low = _rect(-0.06, 0.0, 0.06, header["center"][2] - half[2], fill="#536477")
    wall_header = _rect(-0.06, header["center"][2] - half[2], 0.06,
                        header["center"][2] + half[2], fill="#29394c",
                        stroke="#172536", stroke_width=2)
    wall_upper = _rect(-0.06, header["center"][2] + half[2], 0.06, z1, fill="#e3e8ed")
    table = config["objects"]["table"]
    table_left = table["center"][0] - table["size"][0] / 2
    table_right = table["center"][0] + table["size"][0] / 2
    table_top = table["center"][2] + table["size"][2] / 2
    table_top_thickness = table["top_thickness"]
    table_rect = _rect(table_left, table_top - table_top_thickness, table_right, table_top,
                       fill="#8a4d2c", stroke="#61351f", stroke_width=2)
    base_width = config["robot"]["size"][0]
    base_height = config["robot"]["size"][2]
    base_rect = _rect(base[0] - base_width / 2, 0.02,
                      base[0] + base_width / 2, base_height,
                      rx=10, fill="#285db4", stroke="#173c7a", stroke_width=2)
    camera_xy = _map(camera[0], camera[2])
    cup_xy = _map(target[0], target[2])
    ray_wall_t = float(interval[0])
    ray_stop = camera + ray_wall_t * (target - camera)
    ray_stop_xy = _map(ray_stop[0], ray_stop[2])
    end_effector_phases = {"arm_to_pregrasp", "approach_cup", "close_gripper", "lift_cup"}
    ee_points = []
    for row in trace:
        if row["phase"] not in end_effector_phases:
            continue
        pose = np.asarray(row["gripper_pose"], dtype=float)
        ee_points.append(_map(float(pose[0, 3]), float(pose[2, 3])))
    ee_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in ee_points)
    cup_radius = config["objects"]["cup"]["radius"]
    cup_height = config["objects"]["cup"]["height"]
    cup_left, cup_top = _map(target[0] - cup_radius, target[2] + cup_height / 2)
    cup_right, cup_bottom = _map(target[0] + cup_radius, target[2] - cup_height / 2)

    svg = f'''<svg viewBox="0 0 900 500" role="img" aria-label="Side section showing the robot camera ray blocked by the doorway header while the arm reaches below it">
      {room1}{room2}{wall_low}{wall_header}{wall_upper}{table_rect}{base_rect}
      <line x1="40" y1="435" x2="650" y2="435" stroke="#536477" stroke-width="3"/>
      <line x1="{camera_xy[0]:.1f}" y1="{camera_xy[1]:.1f}" x2="{cup_xy[0]:.1f}" y2="{cup_xy[1]:.1f}" stroke="#d74343" stroke-width="3" stroke-dasharray="10 7"/>
      <line x1="{camera_xy[0]:.1f}" y1="{camera_xy[1]:.1f}" x2="{ray_stop_xy[0]:.1f}" y2="{ray_stop_xy[1]:.1f}" stroke="#d74343" stroke-width="5"/>
      <circle cx="{camera_xy[0]:.1f}" cy="{camera_xy[1]:.1f}" r="8" fill="#e74242" stroke="white" stroke-width="3"/>
      <text x="{camera_xy[0]-85:.1f}" y="{camera_xy[1]-16:.1f}" class="small red">底盘相机</text>
      <line x1="{cup_left:.1f}" y1="{cup_top:.1f}" x2="{cup_right:.1f}" y2="{cup_bottom:.1f}" stroke="#1478d4" stroke-width="12" stroke-linecap="round"/>
      <polyline points="{ee_polyline}" fill="none" stroke="#eda51e" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="{ray_stop_xy[0]:.1f}" cy="{ray_stop_xy[1]:.1f}" r="10" fill="#d74343" stroke="white" stroke-width="3"/>
      <text x="{ray_stop_xy[0]+16:.1f}" y="{ray_stop_xy[1]-8:.1f}" class="small red">视线被门楣挡住</text>
      <text x="70" y="55" class="room-label">Room 1 · 底盘停靠</text>
      <text x="460" y="55" class="room-label">Room 2 · 杯子在墙后</text>
      <text x="{_map(-.06,1.96)[0]:.1f}" y="{_map(-.06,1.96)[1]:.1f}" class="small">实体门楣</text>
      <text x="{_map(-.50,.18)[0]:.1f}" y="{_map(-.50,.18)[1]:.1f}" class="small blue">移动底盘</text>
      <text x="{_map(.58,.66)[0]:.1f}" y="{_map(.58,.66)[1]:.1f}" class="small">工作台</text>
      <text x="{_map(.47,.99)[0]:.1f}" y="{_map(.47,.99)[1]:.1f}" class="small blue">杯子</text>
      <text x="{_map(.17,1.30)[0]:.1f}" y="{_map(.17,1.30)[1]:.1f}" class="small amber">末端执行器路径（门楣下方）</text>
      <text x="68" y="478" class="small">x 方向截面：墙体厚 12 cm；蓝色末端路径通过门洞并到达杯子</text>
      <text x="680" y="478" class="small">z 高度（米）</text>
      <g class="grid">'''
    for z in (0.5, 1.0, 1.5, 2.0):
        px, py = _map(1.25, z)
        svg += f'<line x1="70" y1="{py:.1f}" x2="620" y2="{py:.1f}"/><text x="630" y="{py+5:.1f}" class="axis">{z:.1f}</text>'
    svg += '''</g></svg>'''

    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>看不见，但够得到：门口取物任务</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#172b40;font:16px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
header{{padding:18px 24px;background:#fff;border-bottom:1px solid #d8e0e8}}h1{{margin:0 0 5px;font-size:24px}}header p{{margin:0;color:#52677b}}
main{{display:grid;grid-template-columns:minmax(420px,43%) 1fr;gap:14px;padding:14px;height:calc(100vh - 94px)}}section{{background:#fff;border:1px solid #d8e0e8;border-radius:10px;overflow:hidden;min-height:0}}
.left{{padding:14px;display:flex;flex-direction:column}}.left h2{{margin:0 0 6px;font-size:17px}}.diagram{{width:100%;max-height:56vh}}svg text{{font-family:system-ui,"Microsoft YaHei",sans-serif;fill:#34485c}}.room-label{{font-weight:700;font-size:19px}}.small{{font-size:12px}}.axis{{font-size:11px}}.red{{fill:#c62f36}}.blue{{fill:#1e5bb3}}.amber{{fill:#9c6610}}.grid line{{stroke:#8998a8;stroke-dasharray:3 5;opacity:.35}}.notes{{margin-top:8px;padding:10px 12px;background:#f5f8fb;border-radius:7px;font-size:14px}}.notes strong{{color:#153a63}}
.right iframe{{width:100%;height:100%;border:0}}.right{{position:relative}}.tag{{position:absolute;top:10px;left:10px;padding:5px 9px;background:#ffffffdf;border-radius:6px;font-weight:650;z-index:2}}
@media(max-width:950px){{main{{height:auto;grid-template-columns:1fr}}.right{{height:72vh}}.diagram{{max-height:none}}}}
</style><header><h1>看不见，但够得到：门口取物任务</h1><p>底盘停在 Room 1；高位相机的视线被实体门楣挡住；GCS 规划机械臂从门楣下方伸入 Room 2 并抓取杯子。</p></header>
<main><section class="left"><h2>侧面剖面：遮挡与可达同时成立</h2><div class="diagram">{svg}</div><div class="notes"><strong>逐帧验证：</strong>抓取阶段底盘始终属于 Room 1；末端执行器进入 Room 2。虚线视线与门楣碰撞，黄色为规划的末端路径。整条离散轨迹碰撞检查：{"通过" if result["collision_free"] else "未通过"}。</div><div class="notes">相机位姿和视线用于任务可视化审计；Drake 碰撞模型实际包含门楣、墙体、机械臂、夹爪和杯子。</div></section>
<section class="right"><div class="tag">Drake 3D 完整执行动画</div><iframe title="Drake 3D pick and place animation" src="review_scene.html"></iframe></section></main></html>'''
    (output / "task_visibility.html").write_text(page)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
