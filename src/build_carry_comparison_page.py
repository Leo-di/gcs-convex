"""Publish measured planning and contact-execution comparisons, including failures."""
import argparse
import html
import json
from pathlib import Path
import shutil
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("outputs/results/paired_carry_comparison"))
    parser.add_argument("--output",type=Path,default=Path("outputs/results/grasp_fix_web/planar/comparison"))
    args=parser.parse_args();source=args.results;output=args.output;output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/"summary.json").read_text())
    physics={m:json.loads((source/(m+"_dynamics")/"result.json").read_text()) for m in ("baseline","disk")}
    summary["dynamics"]=physics
    for m in physics:
        shutil.copy2(source/(m+"_dynamics")/"scene.html",output/(m+"_scene.html"))
        shutil.copy2(source/(m+"_dynamics")/"result.json",output/(m+"_dynamics.json"))
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    rows=[]
    add=lambda name,a,b:rows.append(f'<tr><th>{html.escape(name)}</th><td>{html.escape(str(a))}</td><td>{html.escape(str(b))}</td></tr>')
    runs=summary["planning_runs"]
    for key,label in [("prepare_seconds","区域构造 / 准备（中位数）"),("final_solve_seconds","最终 GCS 求解（中位数）"),("planning_total_seconds","规划总时间（中位数）")]:
        values=[]
        for method in ("baseline","disks"):
            actual_key="solve_seconds" if method=="disks" and key=="final_solve_seconds" else key
            numbers=[r[actual_key] for r in runs[method]]
            values.append(f'{statistics.median(numbers):.2f} s（{min(numbers):.2f}–{max(numbers):.2f}）')
        add(label,*values)
    for key,label,unit in [("base_length_m","携物基座路长","m"),("yaw_travel_rad","底盘转角累计","rad"),("arm_travel_rad","关节向量累计行程","rad")]:
        add(label,*[f'{summary[m][key]:.3f} {unit}' for m in ("baseline","disks")])
    add("统一速度限制后的携物参考时长",*[f'{summary[m]["retiming"]["duration_s"]:.2f} s' for m in ("baseline","disks")])
    yes=lambda v:"通过" if v else "失败"
    add("完整参考轨迹几何 / 语义",*[yes(summary[m]["dense_audit"]["satisfied"] and summary[m]["semantic"] and summary[m]["grasp_geometry"]) for m in ("baseline","disks")])
    for key,label in [("satisfied","完整接触执行"),("lifted_with_bilateral_contact","实际抬杯"),("carried_to_room1_with_bilateral_contact","实际携物到 Room 1"),("released_and_settled_in_bin","释放后落稳"),("actuator_effort_limits_satisfied","执行器限幅")]:
        add(label,*[yes(physics[m][key]) for m in ("baseline","disk")])
    add("非预期接触检查记录数",*[physics[m]["unexpected_contact_count"] for m in ("baseline","disk")])
    add("最大底盘跟踪误差",*[f'{1000*physics[m]["maximum_base_position_error_m"]:.2f} mm' for m in ("baseline","disk")])
    add("最大机械臂跟踪误差",*[f'{physics[m]["maximum_arm_tracking_error_rad"]:.5f} rad' for m in ("baseline","disk")])
    add("携物杯子相对位置最大变化",*[f'{1000*physics[m]["maximum_relative_translation_change_during_carry_m"]:.2f} mm' for m in ("baseline","disk")])
    if physics["baseline"]["satisfied"] and physics["disk"]["satisfied"]:
        conclusion="本场景两种方法都完成接触执行；圆盘方法目前没有体现路径长度和计算时间优势。建议保留原 10D GCS 为主基线，将圆盘方法作为待改进的研究原型。"
    elif physics["baseline"]["satisfied"]:
        conclusion="原方法通过接触执行，圆盘方法未通过。目前不应替换原规划器，需先分析新路线的执行失败。"
    else:
        conclusion="当前统一执行设置下并非两种方法都通过。请结合下表检查失败条件，不能用几何路径成功代替物理任务成功。"
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>携物规划实测对照</title><style>body{margin:0;background:#f1f5f9;color:#172b43;font:16px/1.65 system-ui}main{max-width:1150px;margin:auto;padding:24px}h1{font-size:27px}.card{background:white;padding:20px;border-radius:12px;margin:16px 0}.result{border-left:5px solid #0a8877}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #e1e7ef}thead{background:#e8eef5}th{font-weight:500}iframe{width:100%;height:650px;border:0}button{padding:10px 18px;font:inherit;cursor:pointer}a{color:#087a85}</style><main><h1>原 10D GCS 与有限姿态圆盘 GCS：实测对照</h1><div class="card result">__CONCLUSION__</div><div class="card"><strong>控制条件</strong><p>同一场景、抓取变换、携物起终配置；两边重建区域，均使用 Clarabel、10 条取整候选、固定种子。计时各重复 __REPEATS__ 次。导航、抓取、放置与控制器相同；携物路径按共同分量速度上限重新定时：平移每轴 0.30 m/s，底盘转角与各关节 0.40 rad/s。</p><p>接触仿真每种方法运行一次：1 ms 步长、10 ms 控制周期、同一摩擦与力矩限制、自由杯子，不使用 attachment。两条轨迹时长不同是路径与姿态行程不同所致；不是按各自原始采样密度播放。</p><p>这是单场景实现对照，不是成功率统计，也不是只改变凸集形状的消融：原方法使用 10D 连续配置，圆盘方法使用有限姿态和离散转换；目标函数、区域构造和曲线阶数不同。圆盘姿态库取自此前原方法路线，来源成本未纳入本次冷建图计时。不能声称独立端到端优势。</p></div><div class="card"><table><thead><tr><th>指标</th><th>原 10D GCS</th><th>有限姿态圆盘 GCS</th></tr></thead><tbody>__ROWS__</tbody></table></div><div class="card"><strong>本次真实接触仿真记录</strong><p><button data-scene="baseline_scene.html">原 10D GCS</button> <button data-scene="disk_scene.html">圆盘 GCS</button></p><div id="label">原 10D GCS · 本次执行</div><iframe src="baseline_scene.html" title="本次接触动力学对照"></iframe><p>使用 Meshcat 自带时间控件。这里播放的是本次对照产生的力驱动仿真，不是运动学参考动画。底盘仍是理想全驱动平面模型，未验证轮地接触或硬件。</p></div><div class="card"><a href="summary.json">完整对照结果与时间序列</a> · <a href="baseline_dynamics.json">原方法执行原始记录</a> · <a href="disk_dynamics.json">圆盘执行原始记录</a> · <a href="../carry_planned/">圆盘规划解释页</a></div></main><script>for(const b of document.querySelectorAll('button'))b.onclick=()=>{document.querySelector('iframe').src=b.dataset.scene;document.getElementById('label').textContent=b.textContent+' · 本次执行'};</script></html>'''
    page=page.replace('__CONCLUSION__',html.escape(conclusion)).replace('__ROWS__',''.join(rows)).replace('__REPEATS__',str(len(runs['baseline'])))
    (output/"index.html").write_text(page)
    print(conclusion)


if __name__=="__main__":main()
