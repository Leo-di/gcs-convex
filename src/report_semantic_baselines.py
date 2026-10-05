"""Write measured report and standalone figures after the long benchmark."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .summarize_semantic_baselines import summarize, METHODS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/results/semantic_baselines_long_v2'));args=p.parse_args()
    root=args.root;progress=json.loads((root/'progress.json').read_text());spec=json.loads((root/'protocol.json').read_text())
    if not (root/'complete.json').exists():raise RuntimeError('Do not label partial data as final')
    contacts=json.loads((root/'contacts.json').read_text())
    if contacts['completed']!=contacts['total']:raise RuntimeError('Contact cohort incomplete')
    summary=summarize(progress['runs'],spec['budget_seconds_per_method'])
    summary.update(completed=progress['completed'],total=progress['total'],planning_complete=True,
                   planning_elapsed_seconds=json.loads((root/'complete.json').read_text())['elapsed_seconds'],protocol=spec,contacts=contacts)
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    rows=progress['runs'];positive=[r for r in rows if r.get('allowed')]
    labels=['B1 first valid','B2 enumerate','Joint GCS'];colors=['#2880c4','#0d957a','#dd7b25']
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for m,label,color in zip(METHODS,labels,colors):
        times=np.linspace(0,40,161)
        counts=[sum(any(h['seconds']<=t for h in r.get('incumbent_history',[])) for r in positive if r['method']==m) for t in times]
        axes[0].step(times,counts,where='post',label=label,color=color)
    axes[0].set(xlabel='Wall-time budget (s)',ylabel='Audited solutions (out of 27)',title='Time to first valid solution',ylim=(-.5,28));axes[0].legend(fontsize=8)
    values=[[min(r['parent_wall_seconds'],300) for r in positive if r['method']==m] for m in METHODS]
    axes[1].boxplot(values,tick_labels=labels,showfliers=True)
    axes[1].set(ylabel='Wall time (s)',title='Full method runtime (27 runs each)')
    pairs=summary['paired_costs']
    for index,m in enumerate(['B1_first_valid','joint_gcs']):
        differences=[100*(r['costs'][m]/r['costs']['B2_enumerate']-1) for r in pairs]
        # Deterministic horizontal jitter only, never altering measured values.
        axes[2].scatter(index+np.linspace(-.12,.12,len(differences)),differences,s=22,alpha=.65,color=colors[0 if index==0 else 2])
        axes[2].plot([index-.2,index+.2],[statistics.median(differences)]*2,color='black',lw=2)
    axes[2].axhline(0,color='gray',lw=1);axes[2].set(xticks=[0,1],xticklabels=[labels[0],labels[2]],ylabel='Cost difference relative to B2 (%)',title=f'Paired audited cost ({len(pairs)} cells)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    for extension in ['png','svg']:fig.savefig(root/f'baseline_comparison.{extension}',dpi=180)
    plt.close(fig)
    names={'B1_first_valid':'B1 顺序选择','B2_enumerate':'B2 枚举比较','joint_gcs':'联合 GCS'}
    table=[]
    for m in METHODS:
        r=summary['methods'][m];physical=[c for c in contacts['runs'] if c['method']==m]
        table.append(f"| {names[m]} | {r['audited_successes']}/{r['positive_runs']} | {r['median_first_valid_seconds_successes']:.2f} s | {r['median_wall_seconds_all_runs']:.2f} s | {r['timeouts']} | {sum(c['status']=='passed' for c in physical)}/{len(physical)} |")
    details=[]
    grouped={}
    for r in positive:grouped.setdefault((r['start'][0],r['start'][1],r['policy']),{}).setdefault(r['method'],[]).append(r)
    for (x,y,policy),methods in grouped.items():
        parts=[]
        for m in METHODS:
            valid=[r for r in methods[m] if r.get('best')]
            parts.append(f"{statistics.median(r['best']['cost'] for r in valid):.5f}" if valid else '—')
        details.append(f"| ({x}, {y}) | {policy} | "+' | '.join(parts)+' |')
    first=summary['methods']['B1_first_valid'];enum=summary['methods']['B2_enumerate'];joint=summary['methods']['joint_gcs']
    preparation=sum(a['preparation_seconds'] for a in json.loads(Path(spec['library'],'report.json').read_text())['attempts'])
    text=f'''# 语义 GCS 的正式 baseline 长实验

本轮的主要发现是：简单顺序基线已经很强。联合 GCS 在 {summary['joint_same_candidate_as_first']}/{summary['paired_all_success_count']} 个配对正例中选择了与 B1 相同的抓取候选；路径代价差主要反映几何路径取整差异，不能归因为更好的语义动作选择。联合方法比完整枚举快，但首次合法解比 B1/B2 更晚，本组任务尚未体现联合选择相对简单基线的实质收益。

## 范围与协议

正式结果来自 `{root}`；冻结协议为 `protocol.json`，记录源代码/输入哈希、方法、条件、顺序和预算。3 个起点 × 4 个动作可用性条件 × 3 个取整种子，完成 {summary['completed']}/{summary['total']} 次独立方法运行。正例为 9 个不同起点/动作条件，各重复三个种子；不能称为 27 个独立随机场景。禁用全部动作的结构负例另计。

物理房间、障碍物、杯子、候选与凸区域均不改变。方法共享同一有限库、目标和审计器，各自重新求解。候选按 x 升序排列；B1 返回第一个通过审计的完整解，B2 逐个尝试并保留最好可行解，联合 GCS 同时选择候选和前后路线。固定动作端点使单候选问题的前后路径代价可分离；B1/B2 使用相同的单分支 GCS 表达，避免更换底层规划器引入混杂。

每个方法每组统一 300 秒墙钟预算，包含启动、加载、构图、求解、完整轨迹审计。每次 GCS 最多 10 个取整路径；算法完成既定搜索后即停止，不耗尽剩余预算继续搜索，因此这是固定算法的时间上限和解可用性比较，不是不限次数的 anytime 最优性能比较。每个工作进程固定一个核、数值线程数为 1，四个核并行，配对方法同核且轮换先后顺序；机器未独占，耗时只描述本机本批实验。出现超时仍保留已审计最好解，不将其简单计为任务失败。

共同库准备此前测得 {preparation:.2f} 秒，本轮没有重新测量完整冷启动；这里的主比较是复用共同库后的规划。该库依赖预先完整规划、IK 和 IRIS，不能声称端到端自动语义规划性能。

## 先导运行与正式运行分离

首个约五分钟先导批次存于 `outputs/results/semantic_baselines_long/`。固定姿态的空手圆盘重复携带了 8 个常量坐标，导致显著数值开销。停止该批次并保存源码快照后，对**全部三个方法**统一做代数消元：空手只优化 XY，连接处和输出恢复完整 10D 配置，携物仍使用 10D。

这保持数学可行路径及代价定义，但不同数值表达可能影响松弛求解和取整路径。修正后的单候选校准求解约 1.92 秒并通过完整审计，随后重新冻结 v2 协议。先导数据和历史 145/167 秒单例记录不计入下表，也不把代数消元带来的变化包装成语义算法贡献。

## 主要结果

| 方法 | 正例规划成功 | 首次合法解中位数 | 完整方法耗时中位数 | 超时次数 | 预定接触执行通过 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

首次合法解耗时仅统计有解运行，完整方法耗时包括失败及截断到 300 秒的超时。B2 完整枚举：{enum['fully_completed_enumerations']}/{enum['positive_runs']}。规划程序错误数分别为 {first['errors']}、{enum['errors']}、{joint['errors']}；求解返回路径但未通过统一审计的尝试数分别为 {first['invalid_attempts']}、{enum['invalid_attempts']}、{joint['invalid_attempts']}。

规划批处理墙钟总时间 {summary['planning_elapsed_seconds']/60:.2f} 分钟，接触验证批处理 {contacts['elapsed_seconds']/60:.2f} 分钟；不含实现、先导与校准时间。总共调用 GCS {sum(len(r['attempts']) for r in rows)} 次，额外完成预定 27 个接触执行记录。

三方法均成功的配对条件有 {summary['paired_all_success_count']} 组。联合 GCS 相对 B2 的配对代价差中位数为 {100*summary['median_joint_vs_enumerate_relative']:.4f}%，相对 B1 为 {100*summary['median_joint_vs_first_relative']:.4f}%。按相对差 ±0.1% 作为展示用平局阈值，联合方法相对 B2 更好 / 相近 / 更差分别为 {summary['joint_better_than_enumerate']} / {summary['joint_tied_with_enumerate']} / {summary['joint_worse_than_enumerate']}。这是与枚举可行解的比较，**不是最优性 gap**。

![Baseline comparison](../{root}/baseline_comparison.png)

## 各条件路径代价（三种种子的中位数）

| 起点 XY | 动作可用性 | B1 | B2 | 联合 GCS |
|---|---|---:|---:|---:|
{chr(10).join(details)}

`all` 表示四个候选均可用；`exclude_110` 禁用 x110；`exclude_110_115` 禁用 x110 和 x115。所有方法均必须完成取杯与送达，不使用无语义直接到终点作为性能 baseline。

## 接触执行与负例

执行组提前固定为 seed 0 的九个正例条件 × 三方法，共 27 个结果；无规划则标为未执行，没有按代价或动画效果筛选。各条新轨迹分别运行自由杯子、力驱动、1 ms 仿真步长、10 ms 控制周期；原始结果位于每个方法目录的 `dynamics/result.json`。仅做一次执行，未测扰动成功率或时间步收敛；底盘仍为理想全驱动平面模型。

三方法结构负例正确拒绝数分别为 {first['negative_rejected']}/{first['negative_runs']}、{enum['negative_rejected']}/{enum['negative_runs']}、{joint['negative_rejected']}/{joint['negative_runs']}。这是当前动作库无接受路径，不证明物理世界无解。

## 局限

- 固定场景、有限候选、相同原语和区域库；没有跨障碍布局、杯子位置或新物体的泛化实验。
- 本轮是相同底层 GCS 上的策略对照，尚未加入 PDDLStream 等外部任务与运动规划实现。
- 自动机与可用性条件人工给定，没有自然语言解析，取放动作内部未联合优化。
- IRIS 区域已有非零抽样碰撞记录；输出路径的密集检查不是连续时间证明。
- B2 每个候选也使用有限取整，因此其最好结果不是全局最优真值。
- 简单顺序方法可能已经足够，不能仅凭联合方法比完整枚举快就宣称总体优势。

## 下一轮应验证的研究假设

当前四个候选只在同一取物台前相隔 5 cm，且候选顺序从靠近左侧房间的位置开始。顺序策略是一个合理但在当前场景很强的启发式；三种起点与可用性变化没有形成明显的“去程容易但携物回程昂贵”的选择冲突。

下一轮应提前固定多种障碍布局、杯子位置和更分散的候选，测试局部动作选择与完整任务代价发生冲突时的表现，并为每种场景重新生成区域、计入准备成本。也应加入多阶段语义任务，验证任务进度与物理可达性真正影响选择。不能事后只保留联合方法获胜的场景，也不应以增加同一简单场景的随机种子数量代替任务多样性。

网页：<http://localhost:7001/baselines/>。完整记录、最好解随时间变化、冻结协议和 27 个执行结果均可从网页查看。
'''
    Path('reports/19_semantic_baseline_long_experiment.md').write_text(text)
    print('Wrote report, summary and PNG/SVG figures')


if __name__=='__main__':main()
