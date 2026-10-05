"""Reproduce fixed-base semantic counterexamples in the existing 3D scene."""
import json
from pathlib import Path

import numpy as np
from pydrake.geometry import Box, Rgba
from pydrake.math import RigidTransform

from .arm_boundary_semantics import ArmBoundaryOracle, monitor
from .build_scene import build_scene
from .collision_checker import evaluate_collision_status


def main():
    source = Path('outputs/results/semantic_multiscene_v2/libraries/nominal')
    output = Path('outputs/results/arm_boundary_v1')
    web = Path('outputs/results/grasp_fix_web/planar/arm_boundary')
    output.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'config.json').read_text())
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    scene = build_scene(config=config, visualize=True)
    oracle = ArmBoundaryOracle(scene)
    rng = np.random.default_rng(1702)
    poses = {}
    lower = scene.plant.GetPositionLowerLimits()[scene.plant.GetJointByName('iiwa_joint_1').position_start():][:7]
    upper = scene.plant.GetPositionUpperLimits()[scene.plant.GetJointByName('iiwa_joint_1').position_start():][:7]
    for i in range(4000):
        arm = scene.arm_home_positions if i == 0 else rng.uniform(-1.8, 1.8, 7)
        if np.any(arm < lower) or np.any(arm > upper):
            continue
        state = np.r_[[-.6, 0, 0], arm]
        value = oracle.query(state)
        kind = ('safe' if value['residual_m'] > .05 else
                'origin_misses' if value['residual_m'] < -.02 and
                value['gripper_origin_residual_m'] > .03 else 'unsafe')
        if kind in poses:
            continue
        collision = evaluate_collision_status(scene, state[:3], arm_q=arm)
        if collision.collision_free:
            poses[kind] = dict(state=state.tolist(), predicate=value,
                               clearance_m=collision.minimum_signed_distance,
                               proposal_index=i)
        if len(poses) == 3:
            break
    if len(poses) != 3:
        raise RuntimeError('Could not reproduce all three collision-free poses')
    scene.meshcat.SetObject('semantic_boundary', Box(.008, 1.2, 2.2), Rgba(1, .15, .15, .22))
    scene.meshcat.SetTransform('semantic_boundary', RigidTransform([0, 0, 1.1]))
    scene.meshcat.SetCameraPose([.8, -1.8, 7.0], [-.6, 0, .8])
    labels = {'safe': '全臂满足边界', 'unsafe': '底盘满足，手臂越界',
              'origin_misses': '底盘和夹爪原点满足，连杆仍越界'}
    cards = []
    for kind in ('safe', 'unsafe', 'origin_misses'):
        record = poses[kind]
        oracle.query(record['state'])
        scene.diagram.ForcedPublish(scene.root_context)
        (web / f'{kind}.html').write_text(scene.meshcat.StaticHtml())
        v = record['predicate']
        cards.append(f'<button onclick="show(\'{kind}\')">{labels[kind]}</button>'
                     f'<p>{labels[kind]}：全臂 residual {v["residual_m"]*100:.2f} cm；'
                     f'夹爪原点 residual {v["gripper_origin_residual_m"]*100:.2f} cm；'
                     f'限制部件 {v["worst"]["body"]}。</p>')
    audit = {}
    for path in sorted(source.glob('*/delivery_trace.json')):
        rows = json.loads(path.read_text())
        result = monitor(rows, oracle)
        for key in ('first_violation', 'worst_violation'):
            if result[key]:
                result[key]['phase'] = rows[result[key]['edge']]['phase']
        audit[path.parent.name] = result
    (output / 'poses.json').write_text(json.dumps(poses, indent=2))
    (output / 'reference_audit.json').write_text(json.dumps(audit, indent=2))
    (web / 'poses.json').write_text(json.dumps(poses, indent=2))
    (web / 'index.html').write_text('''<!doctype html><meta charset="utf-8">
<title>全臂边界语义对照</title><style>body{font:17px system-ui;margin:24px;color:#172b4d}
button{padding:12px;margin:5px;cursor:pointer}iframe{width:100%;height:650px;border:1px solid #ccd}
p{margin:8px 0}</style><h1>同一底盘，不同机械臂姿态</h1>
<p>原始 Drake 双房间场景；底盘固定为 (-0.6, 0, 0)。红色透明平面表示 x=0 的语义边界，不是新增实体障碍物。</p>
<p>要求：所有机械臂和夹爪碰撞几何均在 x≤0。Residual ≥0 表示满足。三个姿态均通过当前场景碰撞检查。</p>
''' + ''.join(cards) + '''<iframe id="scene" src="safe.html"></iframe>
<p>这是静态姿态对照，不是三种姿态之间的可执行轨迹。精确检查当前碰撞模型的盒体和球体，不等同于视觉网格或连续时间证明。
当前仅实现谓词与诊断；尚未证明 GCS 区域整体满足此新约束。</p>
<script>function show(x){document.getElementById('scene').src=x+'.html'}</script>''')
    print(json.dumps({k: {'residual_m': v['predicate']['residual_m'],
                         'clearance_m': v['clearance_m']} for k, v in poses.items()}, indent=2))
    print('Web: http://localhost:7001/arm_boundary/')


if __name__ == '__main__':
    main()
