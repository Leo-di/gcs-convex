"""Publish a compact review page for the full free-cup dynamics suite."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def summarize_suite(nominal: dict, fine: dict, no_grip: dict) -> dict:
    """Apply the predeclared positive, refinement, and negative-control gates."""

    positive_keys = (
        "lifted_with_bilateral_contact",
        "carried_to_room1_with_bilateral_contact",
        "released_and_settled_in_bin",
        "carry_slip_satisfied",
        "actuator_effort_limits_satisfied",
    )
    nominal_ok = bool(
        nominal["satisfied"]
        and all(nominal[key] for key in positive_keys)
        and nominal["unexpected_contact_count"] == 0
    )
    fine_ok = bool(
        fine["satisfied"]
        and all(fine[key] for key in positive_keys)
        and fine["unexpected_contact_count"] == 0
    )
    negative_ok = bool(
        not no_grip["satisfied"]
        and not no_grip["lifted_with_bilateral_contact"]
        and not no_grip["carried_to_room1_with_bilateral_contact"]
        and no_grip["held_bilateral_contact_fraction"] == 0.0
    )
    attachment_free = all(
        not run["cup_attachment"]
        and run["cup_pose_resets_after_initialization"] == 0
        for run in (nominal, fine, no_grip)
    )
    slip_difference = abs(
        nominal["maximum_relative_translation_change_during_carry_m"]
        - fine["maximum_relative_translation_change_during_carry_m"]
    )
    refinement_ok = slip_difference <= 0.002
    phase_markers: dict[str, float] = {}
    for sample in nominal.get("trace", []):
        phase = sample.get("phase")
        if phase and phase not in phase_markers:
            phase_markers[phase] = float(sample["time_s"])
    return {
        "suite_satisfied": bool(
            nominal_ok and fine_ok and negative_ok and attachment_free and refinement_ok
        ),
        "contact_dynamics_validated": bool(nominal_ok and fine_ok),
        "nominal_1ms_satisfied": nominal_ok,
        "refined_0_5ms_satisfied": fine_ok,
        "open_gripper_negative_control_satisfied": negative_ok,
        "attachment_free_and_no_pose_resets": attachment_free,
        "time_step_refinement_slip_difference_m": slip_difference,
        "time_step_refinement_tolerance_m": 0.002,
        "time_step_refinement_satisfied": refinement_ok,
        "nominal": {
            "time_step_s": nominal["time_step_s"],
            "duration_s": nominal["duration_s"],
            "maximum_carry_slip_m": nominal[
                "maximum_relative_translation_change_during_carry_m"
            ],
            "unexpected_contact_count": nominal["unexpected_contact_count"],
            "held_bilateral_contact_fraction": nominal[
                "held_bilateral_contact_fraction"
            ],
            "maximum_arm_tracking_error_rad": nominal[
                "maximum_arm_tracking_error_rad"
            ],
            "maximum_base_position_error_m": nominal[
                "maximum_base_position_error_m"
            ],
        },
        "refined": {
            "time_step_s": fine["time_step_s"],
            "maximum_carry_slip_m": fine[
                "maximum_relative_translation_change_during_carry_m"
            ],
            "unexpected_contact_count": fine["unexpected_contact_count"],
        },
        "negative_control": {
            "grip_enabled": no_grip["grip_enabled"],
            "lifted": no_grip["lifted_with_bilateral_contact"],
            "carried": no_grip["carried_to_room1_with_bilateral_contact"],
            "held_bilateral_contact_fraction": no_grip[
                "held_bilateral_contact_fraction"
            ],
            "maximum_relative_translation_change_m": no_grip[
                "maximum_relative_translation_change_during_carry_m"
            ],
        },
        "phase_markers_s": phase_markers,
        "scope": (
            "Drake SAP contact simulation with a free 0.2 kg cup, force-limited "
            "iiwa/WSG actuation, and an ideal fully actuated planar base. This is "
            "model-level contact validation, not wheel-ground, hardware, or "
            "continuous-time certification."
        ),
    }


def _page(summary: dict) -> str:
    status = "通过" if summary["suite_satisfied"] else "失败"
    nominal = summary["nominal"]
    refined = summary["refined"]
    negative = summary["negative_control"]
    markers = summary.get("phase_markers_s", {})
    grasp_time = markers.get("close_gripper", 158.06)
    carry_time = markers.get("whole_body_carry_to_dropoff", 164.51)
    release_time = markers.get("release_cup", 178.31)
    retreat_time = markers.get("retreat_arm", 182.06)
    duration = nominal["duration_s"]
    action_sequence = (
        ("navigate_to_pickup", "1", "Base navigation", "底盘空手移动；机械臂保持收拢"),
        ("arm_to_pregrasp", "2", "Arm to pre-grasp", "固定取物位，机械臂到预抓取姿态"),
        ("approach_cup", "3", "Approach", "末端执行器接近杯子"),
        ("close_gripper", "4", "Grasp", "闭合夹爪并建立双侧接触"),
        ("lift_cup", "5", "Lift", "抬起自由杯；不使用 attachment"),
        ("whole_body_carry_to_dropoff", "6", "Base + arm carry", "底盘与七轴机械臂在 10D C-space 携物运动"),
        ("lower_into_bin", "7", "Lower", "机械臂把杯子放入箱内"),
        ("release_cup", "8", "Release", "打开夹爪，杯子由箱底支撑"),
        ("retreat_arm", "9", "Retreat", "机械臂撤离杯子"),
    )
    sequence_html = "".join(
        f'<button class="sequence-step" type="button" data-seek="{markers.get(key, 0.0):.3f}">'
        f'<b>{number}</b><span><strong>{title}</strong><small>{description}</small>'
        f'<em>{markers.get(key, 0.0):.2f} s</em></span></button>'
        for key, number, title, description in action_sequence
    )
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GCS 自由杯接触动力学验证</title><style>
body{{margin:0;background:#eef2f7;color:#172438;font:16px/1.65 system-ui}}
main{{max-width:1300px;margin:auto;padding:24px}}section{{background:white;border:1px solid #d7e0eb;border-radius:12px;padding:18px;margin:16px 0}}
.ok{{color:#087443}}.bad{{color:#a32121}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dde4ee;text-align:left}}iframe{{width:100%;height:72vh;border:0;border-radius:8px;background:#121820}}code{{background:#edf1f6;padding:2px 5px}}
.player{{position:sticky;top:0;z-index:2;display:grid;grid-template-columns:auto auto auto minmax(220px,1fr) auto;gap:10px;align-items:center;margin:-2px 0 12px;padding:12px;background:#f7f9fc;border:1px solid #d7e0eb;border-radius:10px;box-shadow:0 4px 14px #17243818}}
.player button,.player select{{min-height:38px;border:1px solid #aebdce;border-radius:7px;background:white;color:#172438;font:inherit;padding:5px 11px;cursor:pointer}}.player button:hover{{background:#eaf1f8}}.player button:disabled,.player select:disabled,.player input:disabled{{cursor:wait;opacity:.55}}
.timeline{{width:100%;accent-color:#087443;cursor:pointer}}.time{{min-width:112px;text-align:right;font-variant-numeric:tabular-nums;font-weight:650}}.jumps{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}}.jumps button{{border:1px solid #bfd0df;border-radius:999px;background:#edf5fb;padding:6px 12px;cursor:pointer}}.hint{{margin:8px 0 0;color:#53657a}}#playerStatus{{font-weight:650;color:#725400}}
.rerun-actions{{display:flex;flex-wrap:wrap;gap:10px;align-items:center}}.rerun-actions button{{min-height:42px;border:1px solid #88a4bc;border-radius:8px;background:#edf5fb;color:#172438;font:inherit;font-weight:700;padding:7px 15px;cursor:pointer}}.rerun-actions button.primary{{background:#087443;color:white;border-color:#087443}}.rerun-actions button:disabled{{cursor:wait;opacity:.55}}#rerunStatus{{display:block;margin:14px 0 0;font-weight:700}}#rerunResult{{margin-top:8px;color:#24415e}}#rerunLog{{max-height:130px;overflow:auto;white-space:pre-wrap;background:#111827;color:#d8e5f3;border-radius:8px;padding:10px;font:12px/1.45 ui-monospace,monospace}}
.scope-note{{border-left:5px solid #d99700;background:#fff8e6;padding:12px 15px;border-radius:7px}}.scope-note strong{{color:#725400}}.sequence{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:14px 0}}.sequence-step{{display:flex;gap:10px;align-items:flex-start;text-align:left;border:1px solid #bfd0df;border-radius:10px;background:#f6f9fc;color:#172438;padding:11px;cursor:pointer}}.sequence-step:hover{{border-color:#087443;background:#edf8f3}}.sequence-step>b{{display:grid;place-items:center;flex:0 0 28px;height:28px;border-radius:50%;background:#087443;color:white}}.sequence-step span{{display:flex;min-width:0;flex-direction:column}}.sequence-step strong{{line-height:1.25}}.sequence-step small{{color:#53657a;line-height:1.4;margin-top:3px}}.sequence-step em{{color:#087443;font-size:12px;font-style:normal;font-variant-numeric:tabular-nums;margin-top:4px}}
@media(max-width:760px){{main{{padding:10px}}.player{{grid-template-columns:repeat(3,auto)}}.timeline{{grid-column:1/-1}}.time{{grid-column:1/-1;text-align:left}}iframe{{height:62vh}}}}
@media(max-width:900px){{.sequence{{grid-template-columns:1fr}}}}
</style><main><h1>直接 Drake GCS → 自由杯接触动力学</h1>
<section><h2 class="{'ok' if summary['suite_satisfied'] else 'bad'}">整套验收：{status}</h2>
<p>杯子全程为自由刚体；初始化后没有 attachment，也没有重置杯子位姿。</p>
<table><tr><th>检查</th><th>结果</th><th>指标</th></tr>
<tr><td>1 ms SAP 完整执行</td><td>{'通过' if summary['nominal_1ms_satisfied'] else '失败'}</td><td>滑移 {nominal['maximum_carry_slip_m']*1000:.2f} mm；非预期接触 {nominal['unexpected_contact_count']}</td></tr>
<tr><td>0.5 ms 步长复验</td><td>{'通过' if summary['refined_0_5ms_satisfied'] else '失败'}</td><td>滑移 {refined['maximum_carry_slip_m']*1000:.2f} mm；两步长差 {summary['time_step_refinement_slip_difference_m']*1000:.2f} mm</td></tr>
<tr><td>张开夹爪负对照</td><td>{'通过' if summary['open_gripper_negative_control_satisfied'] else '失败'}</td><td>抬起={negative['lifted']}；搬运={negative['carried']}；双侧接触比例 {negative['held_bilateral_contact_fraction']:.0%}</td></tr></table>
<p>名义执行：双指接触比例 {nominal['held_bilateral_contact_fraction']:.0%}；机械臂最大跟踪误差 {nominal['maximum_arm_tracking_error_rad']:.4f} rad；底盘最大位置误差 {nominal['maximum_base_position_error_m']*100:.2f} cm。</p></section>
<section><h2>现场重新运行</h2>
<div class="rerun-actions"><button id="rerunDynamics" type="button">快速重跑：仅 Drake 动力学</button><button id="rerunFull" class="primary" type="button">完整重跑：GCS + IRIS + Drake</button></div>
<output id="rerunStatus">尚未启动新运行。</output><div id="rerunResult"></div><pre id="rerunLog" hidden></pre>
<p class="hint">完整重跑会重新求语义路线、GCS 和 IRIS 区域，再执行自由杯 SAP 动力学并生成新的 Meshcat 记录。一次只运行一个任务；计算期间页面可继续查看旧记录。</p></section>
<section><h2>Long-horizon 动作序列：一次点击执行完整任务</h2>
<p>从初始位置到最终撤离共 <strong>9 个有序动作</strong>，总仿真时间 {duration:.2f} s。点击任一步可直接跳到对应时刻。</p>
<div class="sequence">{sequence_html}</div>
<p class="scope-note"><strong>当前求解边界：</strong>这是一次启动的完整长任务（one-click full sequence），但求解器仍是分层结构：语义 product graph 决定动作顺序，底盘导航、10D 携物和四个操作阶段分别调用 GCS/IRIS，再由 Drake 连续执行。它<strong>不是</strong>把所有离散动作、底盘和机械臂轨迹放进一个 monolithic GCS 中一次联合优化。</p>
</section>
<section><h2>自由杯完整接触动力学回放</h2>
<div class="player" aria-label="回放控制">
<button id="play" type="button" disabled>▶ 播放</button><button id="pause" type="button" disabled>⏸ 暂停</button>
<select id="speed" aria-label="播放速度" disabled><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select>
<input id="timeline" class="timeline" type="range" min="0" max="{duration:.3f}" step="0.01" value="0" aria-label="回放时间轴" disabled>
<output id="timeLabel" class="time">0.00 / {duration:.2f} s</output></div>
<div class="jumps"><button type="button" data-seek="0">从头</button><button type="button" data-seek="{grasp_time:.3f}">抓取</button><button type="button" data-seek="{carry_time:.3f}">搬运</button><button type="button" data-seek="{release_time:.3f}">释放</button><button type="button" data-seek="{retreat_time:.3f}">撤离</button></div>
<iframe id="meshcatFrame" title="自由杯完整接触动力学回放" src="scene.html"></iframe>
<p class="hint"><span id="playerStatus">正在加载 181 MB 自包含 Drake/Meshcat 场景…</span>　快捷键：空格播放/暂停，←/→ 后退/前进 5 秒。墙和杯子均使用 Drake 碰撞几何。</p></section>
<section><h2>证据边界</h2><p>{summary['scope']}</p>
<p><a href="summary.json">验证摘要 JSON</a> · <a href="../report_ready_gcs_v3_dynamics/result.json">1 ms 原始结果</a> · <a href="../report_ready_gcs_v3_dynamics_fine/result.json">0.5 ms 原始结果</a> · <a href="../report_ready_gcs_v3_dynamics_no_grip/result.json">负对照</a></p></section></main>
<script>
const frame = document.getElementById('meshcatFrame');
const play = document.getElementById('play');
const pause = document.getElementById('pause');
const speed = document.getElementById('speed');
const timeline = document.getElementById('timeline');
const timeLabel = document.getElementById('timeLabel');
const playerStatus = document.getElementById('playerStatus');
const rerunDynamics = document.getElementById('rerunDynamics');
const rerunFull = document.getElementById('rerunFull');
const rerunStatus = document.getElementById('rerunStatus');
const rerunResult = document.getElementById('rerunResult');
const rerunLog = document.getElementById('rerunLog');
let dragging = false;
let activeJob = null;
let loadedJob = null;

function animator() {{
  try {{ return frame.contentWindow.viewer.animator; }} catch (_) {{ return null; }}
}}
function showTime(value, duration) {{
  timeLabel.value = `${{value.toFixed(2)}} / ${{duration.toFixed(2)}} s`;
}}
function seekTo(value) {{
  const a = animator();
  if (!a) return;
  const target = Math.max(0, Math.min(Number(value), a.duration));
  a.seek(target);
  a.display_progress(target);
  timeline.value = target;
  showTime(target, a.duration);
}}
function setReady(a) {{
  timeline.max = a.duration;
  for (const element of [play, pause, speed, timeline]) element.disabled = false;
  a.mixer.timeScale = Number(speed.value);
  playerStatus.textContent = '回放已就绪';
  playerStatus.style.color = '#087443';
  showTime(a.time || 0, a.duration);
}}
function waitForAnimator() {{
  const a = animator();
  if (!a || !Number.isFinite(a.duration) || a.duration <= 0) {{
    window.setTimeout(waitForAnimator, 250);
    return;
  }}
  setReady(a);
}}
function setRerunBusy(busy) {{
  rerunDynamics.disabled = busy;
  rerunFull.disabled = busy;
}}
function showJob(job) {{
  if (!job) return;
  activeJob = job.id;
  const seconds = job.elapsed_seconds == null ? '' : ` · ${{job.elapsed_seconds.toFixed(1)}} s`;
  rerunStatus.textContent = `${{job.status}} · ${{job.stage}}${{seconds}} · ${{job.message}}`;
  rerunStatus.style.color = job.status === 'succeeded' ? '#087443' : job.status === 'failed' ? '#a32121' : '#725400';
  const lines = job.log_tail || [];
  rerunLog.hidden = lines.length === 0;
  rerunLog.textContent = lines.join('\n');
  setRerunBusy(job.status === 'queued' || job.status === 'running');
  if (job.result) {{
    const r = job.result;
    rerunResult.textContent = `新运行：${{r.satisfied ? '通过' : '失败'}}；携带滑移 ${{r.maximum_carry_slip_mm.toFixed(2)}} mm；非预期接触 ${{r.unexpected_contact_count}}；双指接触 ${{(100*r.held_bilateral_contact_fraction).toFixed(0)}}%。`;
  }}
  if (job.status === 'succeeded' && job.scene_url && loadedJob !== job.id) {{
    loadedJob = job.id;
    playerStatus.textContent = '正在加载刚刚生成的 Drake/Meshcat 记录…';
    playerStatus.style.color = '#725400';
    frame.src = `${{job.scene_url}}?job=${{encodeURIComponent(job.id)}}`;
  }}
}}
async function pollJob() {{
  if (!activeJob) return;
  try {{
    const response = await fetch(`/api/status?id=${{encodeURIComponent(activeJob)}}`, {{cache:'no-store'}});
    const payload = await response.json();
    if (payload.job) showJob(payload.job);
    if (payload.job && (payload.job.status === 'queued' || payload.job.status === 'running')) {{
      window.setTimeout(pollJob, 1000);
    }}
  }} catch (error) {{
    rerunStatus.textContent = `状态读取失败：${{error}}`;
    rerunStatus.style.color = '#a32121';
    setRerunBusy(false);
  }}
}}
async function startRerun(mode) {{
  setRerunBusy(true);
  rerunResult.textContent = '';
  rerunLog.hidden = true;
  rerunStatus.textContent = mode === 'full' ? '正在启动完整重跑…' : '正在启动动力学重跑…';
  try {{
    const response = await fetch('/api/rerun', {{
      method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{mode}})
    }});
    const payload = await response.json();
    if (!response.ok && !payload.job) throw new Error(payload.error || `HTTP ${{response.status}}`);
    showJob(payload.job);
    window.setTimeout(pollJob, 500);
  }} catch (error) {{
    rerunStatus.textContent = `启动失败：${{error}}`;
    rerunStatus.style.color = '#a32121';
    setRerunBusy(false);
  }}
}}
rerunDynamics.addEventListener('click', () => startRerun('dynamics'));
rerunFull.addEventListener('click', () => startRerun('full'));
play.addEventListener('click', () => {{
  const a = animator();
  if (!a) return;
  a.mixer.timeScale = Number(speed.value);
  a.play();
}});
pause.addEventListener('click', () => {{ const a = animator(); if (a) a.pause(); }});
speed.addEventListener('change', () => {{ const a = animator(); if (a) a.mixer.timeScale = Number(speed.value); }});
timeline.addEventListener('pointerdown', () => {{ dragging = true; }});
timeline.addEventListener('pointerup', () => {{ dragging = false; }});
timeline.addEventListener('input', () => seekTo(timeline.value));
for (const button of document.querySelectorAll('[data-seek]')) {{
  button.addEventListener('click', () => seekTo(button.dataset.seek));
}}
document.addEventListener('keydown', event => {{
  if (event.target.matches('select,input')) return;
  const a = animator();
  if (!a) return;
  if (event.code === 'Space') {{ event.preventDefault(); a.playing ? a.pause() : a.play(); }}
  if (event.code === 'ArrowLeft') {{ event.preventDefault(); seekTo((a.time || 0) - 5); }}
  if (event.code === 'ArrowRight') {{ event.preventDefault(); seekTo((a.time || 0) + 5); }}
}});
function refresh() {{
  const a = animator();
  if (a && !dragging) {{
    const now = Number.isFinite(a.time) ? a.time : 0;
    timeline.value = now;
    showTime(now, a.duration);
  }}
  window.requestAnimationFrame(refresh);
}}
frame.addEventListener('load', waitForAnimator);
waitForAnimator();
refresh();
</script></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--no-grip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    nominal = json.loads((args.nominal / "result.json").read_text())
    fine = json.loads((args.fine / "result.json").read_text())
    no_grip = json.loads((args.no_grip / "result.json").read_text())
    summary = summarize_suite(nominal, fine, no_grip)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.nominal / "scene.html", args.output_dir / "scene.html")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "index.html").write_text(_page(summary))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["suite_satisfied"] else 1)


if __name__ == "__main__":
    main()
