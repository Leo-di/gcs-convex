"""Export measured Inside/IRIS/GCS results, affine slices, and Drake replays."""

import argparse
import itertools
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from pydrake.geometry import Box, Rgba
from pydrake.math import RigidTransform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.build_scene import build_scene, set_robot_configuration
from src.scene_spec import load_scene_config
from src.semantic_compiler import compile_predicates
from src.green_room_iris import make_checker
from src.inside_iris_comparison import interpolate, save

COLORS = dict(conservative="#238477", nonlinear="#2865B0", control="#BD4545")
LABELS = dict(conservative="Conservative domain", nonlinear="Direct nonlinear")


def affine_slice(record, fixed):
    A, b = np.array(record["A"]), np.array(record["b"])
    C = A[:, :2]
    e = b - A[:, 2:] @ fixed[2:]
    vertices = []
    for i, j in itertools.combinations(range(len(e)), 2):
        M = C[[i, j]]
        if abs(np.linalg.det(M)) < 1e-14:
            continue
        x = np.linalg.solve(M, e[[i, j]])
        if np.max(C @ x - e) <= 1e-8:
            vertices.append(x)
    if len(vertices) < 3:
        return None
    v = np.unique(np.round(vertices, 10), axis=0)
    return v[ConvexHull(v).vertices] if len(v) >= 3 else None


def export_replay(problem, path, output):
    scene = build_scene(
        config=load_scene_config(ROOT / problem["scene"]), visualize=True
    )
    meshcat = scene.meshcat
    lo = np.array(problem["dsl"]["regions"]["work_area"]["lower"])
    hi = np.array(problem["dsl"]["regions"]["work_area"]["upper"])
    meshcat.SetObject("/inside/work_area", Box(*(hi - lo)), Rgba(0.15, 0.55, 0.9, 0.07))
    meshcat.SetTransform("/inside/work_area", RigidTransform((lo + hi) / 2))
    meshcat.SetObject(
        "/inside/x_upper_face",
        Box(0.004, hi[1] - lo[1], hi[2] - lo[2]),
        Rgba(0.2, 0.55, 0.95, 0.22),
    )
    meshcat.SetTransform(
        "/inside/x_upper_face",
        RigidTransform([hi[0], (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]),
    )
    meshcat.SetCameraPose([2.8, -1.25, 2.35], [1.5, -0.6, 1.0])
    meshcat.StartRecording(frames_per_second=30)
    points = np.array(path["waypoints"])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    for i, u in enumerate(np.linspace(0, arc[-1], 121)):
        q = np.array([np.interp(u, arc, points[:, j]) for j in range(7)])
        scene.root_context.SetTime(i / 30)
        set_robot_configuration(
            scene, problem["base_q"], q, scene.gripper_open_positions
        )
        scene.diagram.ForcedPublish(scene.root_context)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    html = meshcat.StaticHtml()
    # External scripts keep the static replay small enough for browser tooling.
    index = 0

    def externalize(match):
        nonlocal index
        content = match.group(1)
        if len(content) < 100000:
            return match.group(0)
        name = f"{output.stem}_{index}.js"
        index += 1
        (output.parent / name).write_text(content)
        return f'<script src="{name}"></script>'

    html = re.sub(r"<script>([\s\S]*?)</script>", externalize, html)
    output.write_text(html)


def report(run, output, replay=True):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run / "protocol.json").read_text())
    problem = protocol["problem"]
    results = json.loads((run / "results.json").read_text())
    if len(results) != 2 or not all(r["satisfied"] for r in results):
        raise ValueError("Both completed methods are required")
    scene, checker, mapping = make_checker(
        load_scene_config(ROOT / problem["scene"]),
        problem["base_q"],
        problem["start_arm_q"],
        [-0.055, 0.055],
    )
    c = compile_predicates(
        problem["dsl"],
        scene,
        base=problem["base_q"],
        gripper=scene.gripper_open_positions,
    )
    guide = np.array(protocol["guide"])
    fixed = guide[len(guide) // 2].copy()
    paths = {m: json.loads((run / m / "path.json").read_text()) for m in LABELS}
    records = {m: json.loads((run / m / "regions.json").read_text()) for m in LABELS}
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax = axes[0, 0]
    x, y = np.linspace(-0.62, 0.62, 81), np.linspace(-0.6, -0.2, 61)
    grid = np.empty((len(y), len(x)))
    for i, b in enumerate(y):
        for j, a in enumerate(x):
            q = fixed.copy()
            q[:2] = [a, b]
            grid[i, j] = c.g(q).max() * 1000
    ax.contourf(x, y, grid, levels=[-1000, 0, 1000], colors=["#EDF3F5", "#F8DADA"])
    ax.contour(x, y, grid, levels=[0], colors=[COLORS["control"]], linewidths=1.4)
    for method in LABELS:
        for r in records[method]:
            polygon = affine_slice(r, fixed)
            if polygon is not None:
                ax.fill(
                    polygon[:, 0],
                    polygon[:, 1],
                    color=COLORS[method],
                    alpha=0.2,
                    edgecolor=COLORS[method],
                    lw=0.6,
                )
    ax.plot(guide[:, 0], guide[:, 1], "k.--", lw=1, label="Shared guide")
    ax.plot(
        [-0.5, 0.5],
        [-0.3, -0.3],
        color=COLORS["control"],
        ls=":",
        label="Straight interpolation",
    )
    for m in LABELS:
        ax.plot([], [], lw=5, color=COLORS[m], alpha=0.6, label=LABELS[m])
    ax.set(
        xlabel="Joint 1 (rad)",
        ylabel="Joint 2 (rad)",
        title="A. Exact 2D slices of the seven-dimensional regions",
    )
    ax.legend(fontsize=8, loc="lower center")
    ax.text(
        0.02,
        0.98,
        "Other five joints fixed; red background violates Inside",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax = axes[0, 1]
    for method in ["control", *LABELS]:
        points = (
            interpolate([guide[0], guide[-1]], 0.001)
            if method == "control"
            else np.array(paths[method]["dense"])
        )
        g = np.array([c.g(q).max() for q in points]) * 1000
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        ax.plot(
            arc / arc[-1],
            g,
            color=COLORS[method],
            label=LABELS.get(method, "Straight interpolation"),
            lw=2,
        )
    ax.axhline(0, color="#555", lw=0.8)
    ax.set(
        xlabel="Normalized joint-space path length",
        ylabel="max g(q) (mm); feasible <= 0",
        title="B. Same compiled predicate evaluated along each path",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    ax = axes[1, 0]
    for i, r in enumerate(results):
        other = r["prepare_seconds"] - r["native_iris_seconds"]
        ax.bar(
            i,
            r["native_iris_seconds"],
            color=COLORS[r["method"]],
            edgecolor="black",
            lw=0.5,
        )
        ax.bar(
            i,
            other,
            bottom=r["native_iris_seconds"],
            color="#CED6DD",
            edgecolor="black",
            lw=0.5,
        )
        ax.text(
            i,
            r["prepare_seconds"] + 0.4,
            f"{r['prepare_seconds']:.2f} s; {r['regions']} regions",
            ha="center",
            fontsize=10,
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=list(LABELS.values()),
        ylabel="Wall time (s)",
        title="C. Preparation: native IRIS + construction/checking",
    )
    ax.set_ylim(0, max(r["prepare_seconds"] for r in results) * 1.18)
    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        [
            LABELS[r["method"]],
            r["regions"],
            f"{1000 * r['gcs_seconds']:.1f}",
            f"{r['path_length_rad']:.3f}",
            r["path_audit"]["samples"],
            f"{-1000 * r['path_audit']['maximum_g_m']:.2f}",
        ]
        for r in results
    ]
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Method",
            "Regions",
            "GCS\n(ms)",
            "Length\n(rad)",
            "Path\nchecks",
            "Min slack\n(mm)",
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.32, 0.12, 0.13, 0.13, 0.13, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.1)
    ax.set_title("D. Fresh measured solve and independent path audit")
    ax.text(
        0,
        0.1,
        "128 holdout samples per region; zero recorded violations.\nFixed base, seven arm joints, no payload or contact execution.\nOne diagnostic case; finite checks are not a safety proof.",
        transform=ax.transAxes,
        fontsize=10,
    )
    fig.suptitle("Inside(robot, work_area) -> g(q) -> IRIS-NP2 -> GCS", fontsize=17)
    fig.savefig(output / "comparison.png", dpi=190)
    fig.savefig(output / "comparison.svg")
    svg = output / "comparison.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    # A nominal-domain point rejected by the nonlinear predicate and IRIS polytope.
    rng = np.random.default_rng(20261104)
    witness = None
    for k, r in enumerate(records["nonlinear"]):
        for q in rng.uniform(r["lower"], r["upper"], (200, 7)):
            g = c.g(q)
            j = int(g.argmax())
            if g[j] > 0.001 and checker.CheckConfigCollisionFree(mapping(q)):
                outside = float(np.max(np.array(r["A"]) @ q - np.array(r["b"])))
                if outside > 1e-7:
                    witness = dict(
                        region_index=k,
                        q=q.tolist(),
                        g_max_m=float(g[j]),
                        constraint=c.labels[j],
                        physical_collision_free=True,
                        nominal_domain_contains=True,
                        final_iris_polytope_contains=False,
                        maximum_polytope_residual=outside,
                    )
                    break
        if witness:
            break
    if witness is None:
        raise RuntimeError("No semantic-cut witness found")
    save(output / "semantic_cut_witness.json", witness)
    save(output / "results.json", results)
    if replay:
        for method in LABELS:
            export_replay(problem, paths[method], output / f"{method}_replay.html")
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows
    )
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><meta charset="utf-8"><title>Inside compiler: measured comparison</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 18px;color:#213547}}img{{width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd}}a{{color:#2865b0}}</style>
<h1>Inside → g(q) → IRIS → GCS</h1><p>同一个 DSL、场景、起终点、导引和 GCS 目标。只改变语义约束接入 IRIS 的方式。</p>
<p>七关节固定底盘诊断例子；这是规划轨迹回放，不是接触动力学执行，也不是完整清桌任务。</p>
<p><a href="conservative_replay.html">保守域：机器人回放</a> · <a href="nonlinear_replay.html">直接非线性：机器人回放</a> · <a href="semantic_cut_witness.json">实际语义切割反例</a></p>
<img src="comparison.png"><table><tr><th>Method</th><th>Regions</th><th>GCS ms</th><th>Length rad</th><th>Path checks</th><th>Min slack mm</th></tr>{rows_html}</table>
<p>红色区域是违反 Inside 的配置；区域图是固定其他五关节后的真实二维切片，未把七维路径投影当成切片轨迹。
路径验收来自独立几何计算与原生碰撞检查。有限抽样不等于连续安全证明。准备时间包含 IRIS，不能重复相加。</p>
<p>播放按路径弧长重采样为 4 秒，便于观察，不代表原始求解时间参数。</p>""")
    print(output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-replay", action="store_true")
    a = p.parse_args()
    report(a.run, a.output, replay=not a.no_replay)
