"""Build a local review page from saved, independently inspectable results."""
import argparse
import html
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--full-dynamics", type=Path, help="Root containing validated, validated_fine and open_control runs")
    args = parser.parse_args()
    directory = args.directory
    result = json.loads((directory / "delivery_result.json").read_text())
    audit = json.loads((directory / "dense_replay_audit.json").read_text())
    dynamics = json.loads((directory / "dynamics.json").read_text())
    benchmark = json.loads(args.benchmark.read_text())
    if not (result["collision_free"] and result["grasp_geometry_satisfied"]
            and result["semantic_monitor_satisfied"] and audit["satisfied"]
            and dynamics["suite_satisfied"]):
        raise ValueError("This review page requires a validated geometry and dynamics run")
    if len(benchmark["runs"]) != 6 or not all(r["status"] == "success" for r in benchmark["runs"]):
        raise ValueError("Expected the six successful paired regression runs")
    doorway = result["iris_np2_doorway"]["independent_holdout_audit"]
    cover = result["iris_np2_carrying_cover"]["independent_holdout_audit"]
    (directory / "benchmark_summary.json").write_text(json.dumps(benchmark, indent=2))
    rows = "".join(
        f'<tr><td>{run["time_step_s"]*1000:g} ms</td><td>{"夹持" if run["grip_enabled"] else "张开对照"}</td>'
        f'<td>{run["minimum_held_lift_m"]*100:.2f} cm</td>'
        f'<td>{run["maximum_translation_slip_m"]*1000:.2f} mm</td>'
        f'<td>{"通过" if run["satisfied"] else "未抬起"}</td></tr>'
        for run in dynamics["runs"])
    scene_rows = "".join(
        f'<tr><td>{html.escape(run["scenario"])}</td><td>{run["method"]}</td>'
        f'<td>{html.escape(run["status"])}</td></tr>' for run in benchmark["runs"])
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic GCS · 抓取验证</title>
<style>
body{{margin:0;background:#f4f6fa;color:#182234;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:24px}}
h1{{margin:0;font-size:28px}}h2{{font-size:20px}}p{{margin:8px 0}}small{{color:#546074}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}}
.card{{background:white;border:1px solid #d9e1eb;padding:16px 24px;border-radius:10px;flex:1;min-width:220px}}strong{{color:#176640}}
button,a{{color:#225ac4}}button{{background:white;border:1px solid #9badc6;border-radius:6px;padding:10px 18px;font:inherit;cursor:pointer;margin:0 8px 8px 0}}
button[aria-pressed=true]{{background:#225ac4;color:white}}iframe{{width:100%;height:70vh;border:1px solid #c6d1e0;background:white;border-radius:8px}}
.note{{padding:14px 18px;background:#fff3dc;border-left:4px solid #b87509;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #dde4ee}}
</style><main><small>Drake · Semantic GCS · 2026-09-29</small><h1>抓取修复与验证</h1>
<p>完整任务检查几何与语义；局部实验检查杯子在接触力作用下的抬升与保持。</p>
<div class="cards"><div class="card">完整任务采样检查<br><strong>{result["frames"]} 帧 · {audit["samples"]:,} 个加密采样点</strong><br>几何与语义通过；指尖双侧接触检查通过</div>
<div class="card">局部接触动力学<br><strong>自由杯子抬升约 11.9 cm</strong><br>三个步长通过；张开夹爪对照未抬起</div>
<div class="card">场景扰动检查<br><strong>固定位置 3/3 · 自动选点 3/3</strong><br>杯子 x 坐标 0、−5、+5 cm 扰动</div></div>
<button aria-pressed="true" data-scene="review_scene.html" data-note="完整任务：运动学回放。抓取后的杯子使用刚性 attachment；未验证全程动力学。">完整任务 3D</button>
<button aria-pressed="false" data-scene="dynamics_scene.html" data-note="局部动力学：0.5 ms SAP 接触仿真。杯子为自由刚体，夹爪由竖直试验台驱动；不是完整机械臂控制。">真实接触抬升实验 3D</button>
<a id="direct" href="review_scene.html" target="_blank" rel="noopener">单独打开当前 3D</a>
<p id="scope">完整任务：运动学回放。抓取后的杯子使用刚性 attachment；未验证全程动力学。</p>
<iframe title="Drake 3D recording" src="review_scene.html"></iframe>
<p><small>拖动旋转，滚轮缩放；用 Meshcat 的动画控件播放。页面展示保存的可交互实验记录。</small></p>
<div class="cards"><section class="card"><h2>动力学结果</h2><table><thead><tr><th>步长</th><th>条件</th><th>最小抬升</th><th>最大相对位移</th><th>结果</th></tr></thead><tbody>{rows}</tbody></table>
<p>杯重 0.2 kg；杯面摩擦系数 0.8；每指驱动力上限 20 N。总时长 3 s，统计保持阶段 1.7–3.0 s。相对位移阈值 5 mm、转角阈值 0.1 rad。</p>
<a href="dynamics.json">参数、接触力与完整时间序列</a></section>
<section class="card"><h2>扰动结果</h2><table><thead><tr><th>场景</th><th>选点方法</th><th>结果</th></tr></thead><tbody>{scene_rows}</tbody></table>
<p>这是小规模确定性回归检查，不是成功率统计，也没有证明自动选点更优。</p><a href="benchmark_summary.json">实验摘要</a></section></div>
<div class="note"><b>证据边界</b><br>当前物体是无把手圆柱杯，显示与碰撞外形一致。完整任务仍使用 attachment；局部接触实验不等于整机动态执行。
IRIS 门口区域审计仍有 {doorway["collisions"]}/{doorway["samples"]}、区域覆盖有 {cover["collisions"]}/{cover["samples"]} 个碰撞样本。最终轨迹的加密检查通过，但不能声称所有凸区域严格无碰撞或连续时间安全。</div>
<p><a href="grasp_geometry_audit.json">夹爪几何审计</a> · <a href="dense_replay_audit.json">加密轨迹审计</a> · <a href="semantic_monitor.json">语义监测</a> · <a href="scene_config_snapshot.json">场景参数</a></p>
</main><script>document.querySelectorAll('button[data-scene]').forEach(button=>button.onclick=()=>{{document.querySelectorAll('button[data-scene]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));document.querySelector('iframe').src=button.dataset.scene;document.querySelector('#direct').href=button.dataset.scene;document.querySelector('#scope').textContent=button.dataset.note;}});</script></html>'''
    if args.full_dynamics is not None:
        full_runs = {name: json.loads((args.full_dynamics/name/"result.json").read_text())
                     for name in ("validated", "validated_fine", "open_control")}
        if not (full_runs["validated"]["satisfied"] and full_runs["validated_fine"]["satisfied"]
                and not full_runs["open_control"]["lifted_with_bilateral_contact"]):
            raise ValueError("Full-execution checks or open-gripper control failed")
        shutil.copyfile(args.full_dynamics/"validated/scene.html", directory/"full_dynamics_scene.html")
        for name in full_runs:
            shutil.copyfile(args.full_dynamics/name/"result.json", directory/f"full_{name}.json")
        full = full_runs["validated"]
        note = "整机接触仿真：杯子全程为自由刚体，由受力运动；底盘是理想全驱动平面关节，没有轮地动力学。"
        card = f'''<section class="card"><h2>新增：整机接触执行通过</h2>
<p>自由杯子完成抓取、过门运输、释放和落稳；没有 attachment 或运行中重置杯子姿态。1 ms、0.5 ms 两次执行通过，张开夹爪对照未抬起。</p>
<p>执行时长 {full["duration_s"]:.1f} s（参考轨迹放慢 3 倍）；最大机械臂跟踪误差 {full["maximum_arm_tracking_error_rad"]:.4f} rad，底盘位置误差 {full["maximum_base_position_error_m"]*100:.2f} cm。
搬运阶段杯子相对夹爪位移变化 {full["maximum_relative_translation_change_during_carry_m"]*1000:.2f} mm，不能视为完全不滑移。接触力在 100 Hz 采样下未发现超过 0.1 N 的非预期接触。</p>
<p><a href="full_validated.json">1 ms 原始记录</a> · <a href="full_validated_fine.json">0.5 ms 记录</a> · <a href="full_open_control.json">张开夹爪对照</a></p></section>'''
        first = '<button aria-pressed="true" data-scene="review_scene.html"'
        content = content.replace(first, card + f'<button aria-pressed="true" data-scene="full_dynamics_scene.html" data-note="{note}">整机接触仿真 3D</button>\n' + first.replace('"true"','"false"'))
        content = content.replace('id="direct" href="review_scene.html"', 'id="direct" href="full_dynamics_scene.html"')
        content = content.replace('<iframe title="Drake 3D recording" src="review_scene.html">', '<iframe title="Drake 3D recording" src="full_dynamics_scene.html">')
        content = content.replace('<p id="scope">完整任务：运动学回放。抓取后的杯子使用刚性 attachment；未验证全程动力学。</p>', f'<p id="scope">{note}</p>')
        content = content.replace('完整任务仍使用 attachment；局部接触实验不等于整机动态执行。', '运动学视图仍使用 attachment；新增整机接触视图不使用 attachment，但采用理想平面底盘。尚未验证真实轮地运动、硬件执行或参数扰动鲁棒性。')
        content = content.replace('<h2>动力学结果</h2>', '<h2>局部试验台结果</h2>').replace('<h2>扰动结果</h2>', '<h2>运动学扰动结果</h2>')
    (directory / "index.html").write_text(content)


if __name__ == "__main__":
    main()
