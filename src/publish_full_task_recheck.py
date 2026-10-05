"""Publish a fresh contact execution without conflating it with online IRIS."""
import json
from pathlib import Path
import shutil


def main():
    source = Path('outputs/results/full_task_recheck_v1')
    out = Path('outputs/results/grasp_fix_web/planar/full_task_recheck')
    result = json.loads((source / 'result.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    for name in ('result.json', 'config.json', 'scene.html'):
        shutil.copy2(source / name, out / name)
    checks = {
        '双侧接触抬起杯子': result['lifted_with_bilateral_contact'],
        '双侧接触搬回大房间': result['carried_to_room1_with_bilateral_contact'],
        '释放后杯子在箱内落稳': result['released_and_settled_in_bin'],
        '无超阈值非预期接触记录': result['unexpected_contact_count'] == 0,
        '综合执行判据': result['satisfied'],
    }
    rows = ''.join(f'<li>{name}：{"通过" if ok else "失败"}</li>' for name, ok in checks.items())
    replay = ('<iframe src="scene.html" title="完整抓放接触执行"></iframe>'
              if result['satisfied'] else '<p>本次执行未通过，不展示为成功任务。请查看结果文件。</p>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>完整抓取—搬运—放置</title>
<style>body{{font:17px/1.8 system-ui;background:#eef3f7;color:#20354b;margin:0}}main{{max-width:1200px;margin:auto;padding:24px}}section{{background:white;padding:20px;margin:18px 0;border-radius:12px}}iframe{{width:100%;height:700px;border:0}}a{{color:#086b98}}</style>
<main><h1>完整任务：去取杯子，再搬回并放入箱子</h1>
<section><strong>本次重新执行已有语义 GCS 参考轨迹。</strong>
<p>使用原双房间三维场景及已有 all_candidates 路线，重新运行 Drake 接触动力学；没有重新求解 GCS，也没有把最新七维在线 IRIS 接入此路线。</p>
<p>导航 → 接近杯子 → 闭合夹爪 → 抬升 → 携物穿门 → 放入箱子 → 释放 → 撤回。</p>
<ul>{rows}</ul><p>仿真时间 {result['duration_s']:.2f} 秒；步长 {result['time_step_s']*1000:g} ms；携物观测双侧接触比例 {result['held_bilateral_contact_fraction']:.1%}；杯子相对夹爪最大平移变化 {result['maximum_relative_translation_change_during_carry_m']*1000:.2f} mm。</p></section>
<section>{replay}<p>这是保存的本次物理执行记录。杯子为自由刚体，初始化后无位姿重置或 attachment；底盘为理想全驱动平面模型。可在 Meshcat 动画控件中播放和拖动时间。</p></section>
<section><h2>为什么上一页没有抓放？</h2><p>上一页只测固定底盘下七个机械臂关节的运动规划：杯子静止、夹爪张开。其 73/100 和 72/100 是运动查询成功数，不能解释成抓放任务成功率。</p>
<p>完整任务已有携物 10D IRIS 区域和语义 GCS，但取放仍采用 IK 生成的固定动作原语。下一步需要按空手、接近、携物、释放各模式接入适配的区域与在线扩展；不同模式的杯子碰撞关系不同，七维区域不能直接复用。</p>
<p>单次确定性复跑不构成成功率或鲁棒性证据。接触检查是离散采样；已有区域也不能称严格安全认证。</p>
<a href="result.json">本次原始执行数据</a> · <a href="config.json">动力学配置</a> · <a href="../candidates/">已有候选选择与完整任务对照</a> · <a href="../online_iris_100q/">七维在线 IRIS 子实验</a></section></main></html>'''
    (out / 'index.html').write_text(page)
    # Keep the published dashboard and its reproducible source consistent.
    note = '<section id="full-task-link"><b>想看抓取和放置？</b><p>本页是固定底盘七维运动查询，不包含抓放；成功率不是完整任务成功率。<a href="../full_task_recheck/">打开完整抓取—搬运—放置的接触仿真</a>（已有语义 GCS 路线重新执行，尚未接入本轮在线 IRIS）。</p></section>'
    for path in (Path('src/online_iris_batch_view.html'), out.parent / 'online_iris_100q/index.html'):
        text = path.read_text()
        if 'id="full-task-link"' not in text:
            text = text.replace('<section>', note + '\n<section>', 1)
            path.write_text(text)
    print(out / 'index.html')


if __name__ == '__main__':
    main()
