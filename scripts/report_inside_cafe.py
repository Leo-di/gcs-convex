"""Measured comparison and replay for the complete Inside-constrained cafe task."""

import argparse
import json
from pathlib import Path
import re
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pydrake.geometry import Box, Rgba, Sphere
from pydrake.math import RigidTransform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.build_scene import build_scene, set_robot_configuration
from src.human_passage import HumanPassageEvaluator
from src.meshcat_playback import add_playback_controls
from scripts.package_whole_task_replay import package

COLORS = {"conservative": "#238477", "nonlinear": "#2865B0"}
LABELS = {"conservative": "Conservative domain", "nonlinear": "Direct nonlinear"}


def read(path):
    return json.loads(path.read_text())


def export_replay(run, output, constraints):
    cfg = read(run / "config.json")
    cfg["visualization"] = {
        "global_camera": dict(eye=[5.0, -4.8, 7.7], target=[1.8, 1.3, 0.4])
    }
    scene = build_scene(config=cfg, visualize=True)
    h = HumanPassageEvaluator(scene, force=True)
    routes = [h, *h.connections]
    region = constraints["dsl"]["regions"]["service_workspace"]
    lo = np.array(region["lower"])
    hi = np.array(region["upper"])
    scene.meshcat.SetObject(
        "inside/workspace_floor",
        Box(hi[0] - lo[0], hi[1] - lo[1], 0.005),
        Rgba(0.1, 0.5, 0.9, 0.09),
    )
    scene.meshcat.SetTransform(
        "inside/workspace_floor", RigidTransform([*((lo + hi)[:2] / 2), 0.015])
    )
    for axis in (0, 1):
        size = hi - lo
        size[axis] = 0.006
        for end in (lo[axis], hi[axis]):
            center = (lo + hi) / 2
            center[axis] = end
            path = f"inside/boundary_{axis}_{end}"
            scene.meshcat.SetObject(path, Box(*size), Rgba(0.1, 0.5, 0.9, 0.09))
            scene.meshcat.SetTransform(path, RigidTransform(center))
    for i in range(len(routes)):
        for j in range(48):
            scene.meshcat.SetObject(
                f"access/{i}/{j}",
                Sphere(0.04),
                Rgba(*[(0.05, 0.7, 0.35), (0.15, 0.45, 0.95)][i], 1),
            )
    rows = read(run / "physics.json")["trace"]
    indices = set(range(0, len(rows), 10)) | {len(rows) - 1}
    indices |= {
        i for i in range(1, len(rows)) if rows[i]["phase"] != rows[i - 1]["phase"]
    }
    scene.meshcat.StartRecording(frames_per_second=24)
    for index in sorted(indices):
        row = rows[index]
        t = (row["time_s"] - rows[0]["time_s"]) / 6
        scene.root_context.SetTime(t)
        set_robot_configuration(scene, row["base_q"], row["arm_q"], row["gripper_q"])
        for name, item in row["objects"].items():
            scene.plant.SetFreeBodyPose(
                scene.plant_context,
                scene.object_bodies[name],
                RigidTransform(np.array(item["pose"])),
            )
        for i, route in enumerate(routes):
            report = route._current_single()
            if report["satisfied"]:
                p = np.array(report["path_xy"])
                arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
                sample = np.linspace(0, arc[-1], 48)
                xy = np.column_stack(
                    [np.interp(sample, arc, p[:, j]) for j in range(2)]
                )
            else:
                xy = np.full((48, 2), 100.0)
            for j, point in enumerate(xy):
                scene.meshcat.SetTransform(
                    f"access/{i}/{j}",
                    RigidTransform([*point, 0.04 + i * 0.015]),
                    time_in_recording=t,
                )
        scene.diagram.ForcedPublish(scene.root_context)
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()
    for key in ("contact_forces", "/drake/contact_forces", "/meshcat/contact_forces"):
        scene.meshcat.Delete(key)
    method = run.name
    banner = f"""<div style="position:fixed;top:12px;left:18px;right:18px;background:#fffffff0;padding:12px;z-index:10000;font:15px system-ui">Inside DSL · {LABELS[method]} · 全任务 10D 联合求解后的实测接触执行 · 6 倍速<br>三桌取杯 → 同一回收桌；绿色/蓝色点线表示两条静态人行通路。淡蓝边界是 Inside 工作区。杯子使用实际自由体姿态。</div>"""
    html = add_playback_controls(scene.meshcat.StaticHtml()).replace(
        "<body>", "<body>" + banner, 1
    )
    output.write_text(html)
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                speed=6,
                source=str(run),
                measured_samples=len(rows),
                display_samples=len(indices),
            )
        )
    )
    package(output, run)


def report(root, output, replay=True):
    output.mkdir(parents=True, exist_ok=True)
    results = read(root / "results.json")
    if len(results) != 2 or not all(r["execution_satisfied"] for r in results):
        raise RuntimeError("Both physical runs must pass")
    constraints = read(root / "inputs/inside.json")
    cfg = read(root / "inputs/planning_config.json")
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    summary = []
    for result in results:
        method = result["method"]
        run = root / method
        measured = read(run / "measured_inside_audit.json")
        if not measured["satisfied"]:
            raise RuntimeError("Measured Inside failed")
        trace = read(run / "physics.json")["trace"]
        segments = read(run / "global_segments.json")
        contacts = read(run / "optimized_contacts.json")
        model = read(run / "global_model.json")
        delta = np.array([s["b"] for s in segments]) - np.array(
            [s["a"] for s in segments]
        )
        moving = (np.linalg.norm(delta[:, :2], axis=1) >= 0.001) & (
            np.linalg.norm(delta[:, 3:], axis=1) >= 0.005
        )
        records = [r for f in (run / "covers").glob("*/regions.json") for r in read(f)]
        rejected = [read(p) for p in (run / "covers").glob("*/native_rejected_*.json")]
        active = [
            r
            for r in records
            if r["inside_nonlinear_program_in_iris"]
            and r["inside_active_nonlinear_rows"]
        ]
        stat = dict(
            method=method,
            dimension=10,
            motion_modes=model["motion_modes"],
            variable_contacts=model["variable_contact_states"],
            solve_path_calls=model["solve_path_calls"],
            base_length_m=float(np.linalg.norm(delta[:, :2], axis=1).sum()),
            arm_length_rad=float(np.linalg.norm(delta[:, 3:], axis=1).sum()),
            simultaneous_base_arm_segments=int(moving.sum()),
            simultaneous_thresholds=dict(
                base_translation_m=0.001, arm_joint_norm_rad=0.005
            ),
            base_distance_during_simultaneous_segments_m=float(
                np.linalg.norm(delta[moving, :2], axis=1).sum()
            ),
            maximum_contact_change=float(
                max(c["robot_change_from_anchor"] for c in contacts)
            ),
            nonlinear_iris_regions_with_active_rows=len(active),
            nonlinear_rows_submitted=sum(
                len(r["inside_active_nonlinear_rows"]) for r in active
            ),
            native_iris_and_holdout_seconds=sum(
                r["growth_seconds"]
                for r in records
                if r["kind"] == "official_IRIS_NP2_10D"
            ),
            rejected_native_iris_seconds=sum(r["seconds"] for r in rejected),
            rejected_native_iris_attempts=len(rejected),
            accepted_iris_warning_count=sum(
                len(r.get("iris_warnings", [])) for r in records
            ),
            prepare_seconds=result["prepare_seconds"],
            global_gcs_seconds=result["global_gcs_seconds"],
            measured_inside=measured,
            actions=result["actions"],
            regions=result["regions"],
            iris_regions=result["iris_regions"],
            contact_patches=result["contact_patches"],
            unexpected_contacts=result["unexpected_contacts"],
            human_access_violations=read(run / "independent_passage.json")[
                "violations"
            ],
            execution_duration_s=trace[-1]["time_s"],
        )
        summary.append(stat)
        for name in dict.fromkeys(s["mode"] for s in segments):
            block = [s for s in segments if s["mode"] == name]
            q = np.array([block[0]["a"]] + [s["b"] for s in block])
            axes[0, 0].plot(q[:, 0], q[:, 1], color=COLORS[method], lw=1.5, alpha=0.8)
        axes[0, 0].plot([], [], color=COLORS[method], label=LABELS[method], lw=2)
        gtrace = read(run / "measured_inside_trace.json")
        t = np.array([r["time_s"] for r in gtrace])
        axes[0, 1].plot(
            t,
            [1000 * r["robot_g_m"] for r in gtrace],
            color=COLORS[method],
            lw=1.4,
            label=LABELS[method],
        )
        t = np.array([r["time_s"] for r in trace])
        for i, name in enumerate(("cup", "cup_b", "cup_c")):
            axes[1, 0].plot(
                t,
                [r["objects"][name]["pose"][2][3] for r in trace],
                color=["#2865B0", "#C78426", "#8A5599"][i],
                ls="-" if method == "nonlinear" else "--",
                lw=1.4,
                label=f"Cup {i + 1}"
                + (" (nonlinear)" if method == "nonlinear" else " (conservative)"),
            )
    ax = axes[0, 0]
    tables = dict(cfg.get("support_surfaces", {}))
    tables.update({name: cfg["objects"][name] for name in ("table", "dropoff_table")})
    labels = {
        "table": "Cup 1 table",
        "customer_b": "Cup 2 table",
        "customer_c": "Cup 3 table",
        "customer_d": "Occupied table",
        "dropoff_table": "Return table",
    }
    for name, table in tables.items():
        size = np.array(table["size"][:2])
        center = np.array(table["center"][:2])
        ax.add_patch(
            Rectangle(center - size / 2, *size, color="#D3DADF", alpha=0.6, zorder=0)
        )
        ax.text(*center, labels[name], ha="center", fontsize=8)
    initial = read(root / "inputs/reference.json")[0]["base_q"]
    ax.scatter(
        initial[0],
        initial[1],
        marker="*",
        s=90,
        color="#213547",
        zorder=5,
        label="Start",
    )
    region = constraints["dsl"]["regions"]["service_workspace"]
    lo = np.array(region["lower"])
    hi = np.array(region["upper"])
    ax.add_patch(
        Rectangle(lo[:2], *(hi - lo)[:2], fill=False, ls=":", edgecolor="#667787")
    )
    ax.set(
        xlabel="Base x (m)",
        ylabel="Base y (m)",
        title="A. Complete jointly optimized base routes",
        aspect="equal",
    )
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    ax.axhline(0, color="#BD4545", ls="--", lw=1)
    ax.set(
        xlabel="Measured simulation time (s)",
        ylabel="max g(robot) (mm); feasible <= 0",
        title="B. DSL evaluated at actual measured robot poses",
    )
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    ax.set(
        xlabel="Measured simulation time (s)",
        ylabel="Actual cup height (m)",
        title="C. All three cups: pickup, transport, placement",
    )
    ax.legend(fontsize=8, ncol=2)
    ax = axes[1, 1]
    x = np.arange(2)
    ax.bar(
        x,
        [r["prepare_seconds"] for r in results],
        color=[COLORS[r["method"]] for r in results],
        edgecolor="black",
        lw=0.5,
        label="Preparation",
    )
    ax.bar(
        x,
        [r["global_gcs_seconds"] for r in results],
        bottom=[r["prepare_seconds"] for r in results],
        color="#CED6DD",
        edgecolor="black",
        lw=0.5,
        label="Global GCS",
    )
    ax.set(
        xticks=x,
        xticklabels=list(LABELS.values()),
        ylabel="Computation wall time (s)",
        title="D. Fresh complete-task computation; execution excluded",
    )
    ax.legend(fontsize=9)
    for ax in axes.ravel():
        ax.grid(alpha=0.12)
    fig.suptitle(
        "Inside DSL in one whole-task 10D GCS: three-cup cafe clearing", fontsize=17
    )
    fig.savefig(output / "comparison.png", dpi=190)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    svg = output / "comparison.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 9), sharex="col", sharey="row", layout="constrained"
    )
    for col, method in enumerate(LABELS):
        trace = read(root / method / "physics.json")["trace"]
        t = np.array([r["time_s"] for r in trace])
        base = np.array([r["base_q"] for r in trace])
        arm = np.array([r["arm_q"] for r in trace])
        for j in range(2):
            axes[0, col].plot(t, base[:, j], lw=1, label=["Base x", "Base y"][j])
        for j in range(7):
            axes[1, col].plot(t, arm[:, j], lw=0.8, label=f"q{j + 1}")
        for j, name in enumerate(("cup", "cup_b", "cup_c")):
            axes[2, col].plot(
                t,
                [r["objects"][name]["pose"][2][3] for r in trace],
                lw=1.2,
                label=f"Cup {j + 1}",
            )
        axes[0, col].set_title(LABELS[method])
        axes[2, col].set_xlabel("Measured simulation time (s)")
        for ax in axes[:, col]:
            ax.grid(alpha=0.12)
    for row, label in enumerate(
        ("Base position (m)", "Arm joint angles (rad)", "Cup height (m)")
    ):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].legend(fontsize=8, ncol=4, loc="upper right")
    fig.suptitle(
        "Actual base, arm and all cups on the same execution clock", fontsize=17
    )
    fig.savefig(output / "synchronized_state.png", dpi=190)
    plt.close(fig)
    if replay:
        for m in LABELS:
            export_replay(root / m, output / f"{m}_replay.html", constraints)
    rows = "".join(
        f"<tr><td>{LABELS[s['method']]}</td><td>{s['iris_regions']} + {s['contact_patches']} 接触域</td><td>{s['prepare_seconds']:.1f} s</td><td>{s['global_gcs_seconds']:.1f} s</td><td>{s['simultaneous_base_arm_segments']}</td><td>{s['base_distance_during_simultaneous_segments_m']:.2f} m</td><td>{s['measured_inside']['samples']}</td></tr>"
        for s in summary
    )
    acceptance_rows = "".join(
        f"<tr><td>{LABELS[s['method']]}</td><td>3 / 3</td><td>{1000 * max(a['final_position_error_m'] for a in s['actions'].values()):.3f} mm</td><td>{-1000 * s['measured_inside']['robot_maximum_g_m']:.2f} mm</td><td>{s['human_access_violations']}</td><td>{s['unexpected_contacts']}</td></tr>"
        for s in summary
    )
    direct = next(s for s in summary if s["method"] == "nonlinear")
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><meta charset="utf-8"><title>Inside DSL：完整咖啡馆联合任务</title>
<style>body{{font:16px/1.7 system-ui;max-width:1180px;margin:32px auto;padding:0 18px;color:#213547}}img{{width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd}}a{{color:#2865b0}}pre{{background:#f3f5f7;padding:16px;overflow:auto}}</style>
<h1>Inside DSL → 三杯清桌 → 完整底盘与机械臂联合求解</h1>
<p>输入是相同场景、三杯位置与目标、任务顺序、导引、Inside DSL 和人的通行规则。新编译器输出 10 维 g(q)：空载 120 条；持杯 126 条。两组各自重新生成区域，并各用一次任务级 SolvePath，共同优化 7 个运动模式和 6 个共享抓放构型。</p>
<pre>Inside(robot, service_workspace) + 持杯时 Inside(payload, service_workspace)
  → 同一个 FK 约束编译器
  → A: 整域上界 → 保守域 → IRIS
    B: 不按 Inside 缩域；证明冗余的行省去，其余 g(q) ≤ 0 → IRIS 非线性接口
  → 一张 10D GCS 模式图 → 全任务凸松弛 + 全任务非线性限制问题
  → 独立轨迹检查 → 连续自由体接触执行 → 实测 Inside 与人行通路验收</pre>
<p><a href="conservative_replay.html">保守方式：完整实测回放</a> · <a href="nonlinear_replay.html">直接非线性：完整实测回放</a> · <a href="summary.json">完整比较数据</a></p>
<table><tr><th>方式</th><th>IRIS 区域</th><th>准备</th><th>全任务 GCS</th><th>底盘与机械臂同时变化的段数</th><th>这些段内底盘距离</th><th>实测 Inside 检查点</th></tr>{rows}</table>
<p>同时变化的统计阈值：每段底盘平移至少 1 mm、机械臂关节向量变化至少 0.005 rad。共同导引附近的协调幅度仍很局部，不能用浮点级小变化宣称大范围协同。</p>
<table><tr><th>方式</th><th>杯子成功数</th><th>最大放置误差</th><th>实测机器人最小 Inside 余量</th><th>通路违反</th><th>意外接触</th></tr>{acceptance_rows}</table>
<p>直接非线性方式有 {direct["nonlinear_iris_regions_with_active_rows"]} 个 IRIS 区域实际提交了累计 {direct["nonlinear_rows_submitted"]} 条未获整域证明的标量约束。其余行通过整域上界证明后省去，最终仍全部复核。</p>
<img src="comparison.png"><img src="synchronized_state.png"><p>三杯都必须真实抓起、运输、释放并在回收桌稳定，未把杯子焊在夹爪上；验收使用实际自由体位姿与实际手指位置。人行通路仍使用原有空间评估器，尚未编译成 Inside。</p>
<p>范围：固定任务顺序和抓取类型，已有分阶段轨迹仅提供共同导引；最终各组空间解来自同一个全任务联合模型。局部导引域、固定区域时长、理想平面底盘和静态人物模型仍然保留。接触域有显式非线性碰撞约束，不能称作无碰撞 IRIS 区域。有限时间抽样不等于连续安全证明。</p>""")
    print(output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-replay", action="store_true")
    a = p.parse_args()
    report(a.run, a.output, replay=not a.no_replay)
