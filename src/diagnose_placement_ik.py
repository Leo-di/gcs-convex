"""Constraint ablations for failed IK targets; NEVER produce executable plans."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve
from .build_scene import build_scene, set_robot_configuration
from .grasp_planner import _arm_position_indices
from .semantic_candidate_gcs import read_library


def probe(scene,base,initial,target,orientation,config):
    set_robot_configuration(scene,base,initial,scene.gripper_open_positions)
    full=scene.plant.GetPositions(scene.plant_context).copy();idx=np.array(_arm_position_indices(scene))
    fixed=np.array([i for i in range(len(full)) if i not in idx])
    ik=InverseKinematics(scene.plant,scene.plant_context,True);p=ik.prog();q=ik.q()
    p.AddBoundingBoxConstraint(full[fixed],full[fixed],q[fixed])
    p.AddBoundingBoxConstraint(np.array(config['whole_body_planning']['joint_lower_limits']),
                              np.array(config['whole_body_planning']['joint_upper_limits']),q[idx])
    point=np.array(config['gripper']['grasp_point_in_body']);position=target.multiply(point)
    ik.AddPositionConstraint(scene.gripper_body.body_frame(),point,scene.plant.world_frame(),position-.000199,position+.000199)
    if orientation:ik.AddOrientationConstraint(scene.plant.world_frame(),target.rotation(),scene.gripper_body.body_frame(),RotationMatrix(),.00499)
    # No minimum-distance constraint: only diagnose why strict IK failed.
    p.AddQuadraticErrorCost(np.eye(len(full)),full,q)
    rng=np.random.default_rng(1701);seeds=[full]
    for _ in range(config['ik_solver']['max_restarts']):
        seed=full.copy();seed[idx]=rng.uniform(scene.plant.GetPositionLowerLimits()[idx],scene.plant.GetPositionUpperLimits()[idx]);seeds.append(seed)
    attempts=[];answer=None
    for seed in seeds:
        p.SetInitialGuess(q,seed);r=Solve(p);attempts.append(str(r.get_solution_result()))
        if r.is_success():answer=r.GetSolution(q)[idx];break
    return {'success':answer is not None,'attempts':attempts,'q':answer.tolist() if answer is not None else None,
            'orientation_constraint':orientation,'collision_constraint':False}


def worker(job):
    source,cell=job;bundle=json.loads((Path(source)/'inputs'/(cell+'.json')).read_text());config=bundle['config']
    _,library=read_library(Path('outputs/results/semantic_candidates/library'));rows=library['x110']['rows']
    anchor=[r for r in rows if r['phase']=='lift_cup'][-1];old=[r for r in rows if r['phase']=='lower_into_bin']
    X_GC=RigidTransform(np.array(anchor['gripper_pose'])).inverse()@RigidTransform(np.array(anchor['cup_pose']))
    reason=bundle['primitive_rejections']['x110'];stage='preplace' if 'development_preplace' in reason else 'place'
    center=np.array(config['delivery_task']['place_cup_center']);base=np.array(bundle['cell']['placement_base'])
    if stage=='preplace':center[2]+=config['delivery_task']['preplace_height']
    target=RigidTransform(RotationMatrix(),center)@X_GC.inverse()
    scene=build_scene(config=config,visualize=False);initial=np.array(old[0 if stage=='preplace' else -1]['arm_q'])
    pose=probe(scene,base,initial,target,True,config)
    position=probe(scene,base,initial,target,False,config) if not pose['success'] else None
    return {'cell':cell,'representative_candidate':'x110','original_reason':reason,'failed_stage':stage,
            'pose_without_collision':pose,'position_only':position,
            'scope':'Diagnostic local IK, no collision constraint; never accepted for execution. Uses old-stage warm start, not identical to the new continuation warm start at place. Local failure is not an infeasibility certificate.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('outputs/results/full_task_evaluation_40_v1'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    cells=[f.stem for f in sorted((args.source/'inputs').glob('*.json')) if not json.loads(f.read_text())['candidates']]
    (args.output/'protocol.json').write_text(json.dumps({'cases':cells,'candidate':'x110','scope':'All 27 no-primitive cases, representative grasp; diagnostic ablations only.'},indent=2))
    results=[]
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(worker,[(str(args.source),c) for c in cells]):
            results.append(r);(args.output/'results.json').write_text(json.dumps(results,indent=2))
            print(r['cell'],r['failed_stage'],'pose',r['pose_without_collision']['success'],'position',r['position_only']['success'] if r['position_only'] else None,flush=True)


if __name__=='__main__':main()
