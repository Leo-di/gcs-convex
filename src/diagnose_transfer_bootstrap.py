"""Post-hoc controlled check of old bootstrap lowering, not new benchmark scores."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json,file_hash
from .build_scene import build_scene
from .audit_replay import audit_replay
ROOT=Path('outputs/results/transfer_bootstrap_diagnostic_v1')
BASE=Path('outputs/results/transfer_scene_evaluation_v1')


def main():
    ROOT.mkdir();source=BASE/'source_snapshot/animate_delivery_demo.py';text=source.read_text()
    needle='    for arm_q in _interpolate(delivery_plan.preplace_q, delivery_plan.place_q, 35):'
    replacement='''    # Controlled diagnostic: replace only bootstrap lowering with IK continuation.
    from dataclasses import replace
    lowering = cartesian_arm_path(scene, delivery_plan.base_q,
                                  delivery_plan.preplace_q, delivery_plan.place_q, 35,
                                  stage="diagnostic_lower")
    set_robot_configuration(scene, delivery_plan.base_q, lowering[-1], scene.gripper_closed_positions)
    achieved = _attached_cup_pose(scene, delivery_plan.base_q, lowering[-1], delivery_plan.X_GC)
    delivery_plan = replace(delivery_plan, place_q=lowering[-1], place_cup_pose=achieved)
    for arm_q in lowering:'''
    assert text.count(needle)==1;model=ROOT/'model.py';model.write_text(text.replace(needle,replacement));spec=importlib.util.spec_from_file_location('src._transfer_bootstrap_diagnostic',model);module=importlib.util.module_from_spec(spec);import sys;sys.modules[spec.name]=module;spec.loader.exec_module(module)
    jobs=[('dropoff_east','x115'),('dropoff_east','x120'),('dropoff_east','x125'),('cup_left_back','x125'),('dropoff_east','x110')]
    design=json.loads((BASE/'design.json').read_text());atomic_json(ROOT/'protocol.json',{'jobs':jobs,'scope':'Four observed bootstrap audit rejections and one accepted control. Only lowering interpolation replaced by Cartesian IK continuation, with actual endpoint propagated to release/retreat. Post-hoc conditional diagnosis; not physical execution, not replacing frozen benchmark.','hashes':{str(p):file_hash(p) for p in [source,model,BASE/'design.json',Path(__file__)]}});rows=[]
    for name,candidate in jobs:
        out=ROOT/(name+'_'+candidate);out.mkdir();attempt={'scene':name,'candidate':candidate};begin=time.perf_counter();old=json.loads((BASE/'libraries'/name/'preparation.json').read_text());old=next(a for a in old['attempts'] if a['name']==candidate);attempt['original_status']=old['status']
        try:
            config=design['scenes'][name];scene=build_scene(config=config,visualize=False);data=module.plan_delivery_candidate(scene,np.array([int(candidate[1:])/100,-.1,0.]),attempt)
            audit=audit_replay(data['trace'],config,10);atomic_json(out/'audit.json',audit);atomic_json(out/'delivery_trace.json',data['trace']);atomic_json(out/'scene_config_snapshot.json',config)
            attempt.update(status='accepted' if audit['satisfied'] else 'rejected',dense_satisfied=audit['satisfied'])
            if not audit['satisfied']:attempt['failure']='dense_audit';attempt['first_violation']=audit.get('first_violation')
        except RuntimeError as e:attempt.update(status='rejected',failure=str(e),details=getattr(e,'details',{}))
        except Exception as e:attempt.update(status='error',failure=repr(e))
        attempt['seconds']=time.perf_counter()-begin;atomic_json(out/'record.json',attempt);rows.append(attempt);atomic_json(ROOT/'summary.json',{'records':rows});print(name,candidate,attempt['status'],attempt.get('failure',''),flush=True)
    atomic_json(ROOT/'complete.json',{'runs':len(rows)})


if __name__=='__main__':main()
