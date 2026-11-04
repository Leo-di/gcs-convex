"""Fixed-base arm guide checked against unfiltered cup geometry."""
import time
import numpy as np
from .build_scene import set_robot_configuration
from .grasp_audit import measure_cup_gripper,summarize_grasp_audit
from .collision_checker import evaluate_collision_status


def plan_pregrasp(scene,base,start,goal,*,state_constraint=None,diagnostics=None):
    from .planner_randomness import planning_seed
    began=time.perf_counter();rng=np.random.default_rng(planning_seed(scene.config, "pregrasp", 1702))
    lo=np.array(scene.config['whole_body_planning']['joint_lower_limits'])
    hi=np.array(scene.config['whole_body_planning']['joint_upper_limits'])
    start,goal=np.asarray(start,float),np.asarray(goal,float)
    if (start.shape != lo.shape or goal.shape != lo.shape
            or not np.all(np.isfinite([start,goal]))):
        raise ValueError('Finite arm endpoints required')
    def valid(q):
        if np.any(q < lo) or np.any(q > hi):
            return False
        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,scene.cup_initial_pose)
        collision=evaluate_collision_status(scene,base,arm_q=q)
        set_robot_configuration(scene,base,q,scene.gripper_open_positions)
        audit = measure_cup_gripper(scene)
        finger_clearance = all(pair['distance_m'] >= .002
            for pair in audit['pairs'] if pair['body'] in ('left_finger', 'right_finger'))
        return collision.collision_free and finger_clearance and summarize_grasp_audit([{'attached':False,'cup_gripper_audit':audit}])['satisfied'] and (state_constraint is None or state_constraint(q))
    def edge(a,b):
        n=max(2,int(np.ceil(np.max(np.abs(b-a))/.025))+1)
        return all(valid((1-t)*a+t*b) for t in np.linspace(0,1,n))
    if not valid(start) or not valid(goal):raise RuntimeError('Pregrasp endpoint rejected by raw geometry')
    def finish(path, method):
        from .operation_path import shortcut_path
        path, report = shortcut_path(path, edge, max_checks=128)
        # Recheck the entire result with geometry, raw fingers and semantics,
        # including edges retained when the shortcut budget was exhausted.
        if not all(edge(a,b) for a,b in zip(path,path[1:])):
            raise RuntimeError('Simplified pregrasp failed final edge audit')
        dense=[]
        for a,b in zip(path,path[1:]):
            n=max(2,int(np.ceil(np.max(np.abs(b-a))/.025))+1)
            dense.extend((1-t)*a+t*b for t in np.linspace(0,1,n)[:-1])
        dense.append(goal.copy())
        if diagnostics is not None:
            diagnostics.update(report, method=method, final_edge_audit=True,
                edge_step_rad=.025, elapsed_seconds=time.perf_counter()-began,
                dense_vertices=len(dense),
                scope='Bounded guide search and finite edge sampling; no completeness or continuous collision certificate')
        return np.asarray(dense)
    if edge(start,goal):
        return finish([start,goal], 'validated_direct')
    nodes=[start.copy()];parents=[-1]
    for _ in range(3000):
        if time.perf_counter()-began>20:break
        target=goal if rng.random()<.25 else np.clip((1-(t:=rng.random()))*start+t*goal+rng.normal(0,.4,7),lo,hi)
        idx=int(np.argmin(np.linalg.norm(np.array(nodes)-target,axis=1)));delta=target-nodes[idx]
        q=nodes[idx]+delta*min(1.,.25/max(np.linalg.norm(delta),1e-12))
        if not edge(nodes[idx],q):continue
        nodes.append(q);parents.append(idx)
        if not edge(q,goal):continue
        path=[np.array(goal)];idx=len(nodes)-1
        while idx!=-1:path.append(nodes[idx]);idx=parents[idx]
        return finish(path[::-1], 'rrt_with_validated_shortcuts')
    raise RuntimeError('Pregrasp search budget exhausted; no infeasibility claim')
