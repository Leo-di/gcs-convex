"""Publish joint grasp/placement selection, shared-library costs, and executions."""
import json,shutil
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json
ROOT=Path('outputs/results/coupled_goal_evaluation_v1')
LIB=Path('outputs/results/coupled_goal_library_v1')
METHODS=['joint_gcs','enumerate_gcs','portal_socp']


def analyze():
    records=json.loads((ROOT/'summary.json').read_text())['records'];protocol=json.loads((ROOT/'protocol.json').read_text());methods={};lookup={m:{r['cell']:r for r in records if r['method']==m} for m in METHODS}
    for m,rr in lookup.items():
        rows=list(rr.values());physical=[r for r in rows if r.get('contact_selected')];wall=[r['plan_parent_wall_seconds'] for r in rows]
        methods[m]={'n':len(rows),'planning_successes':sum(r['planning_success'] for r in rows),'contact_n':len(physical),'contact_successes':sum(r['end_to_end_success'] for r in physical),
            'full_enumeration':sum(r.get('enumeration_complete',False) for r in rows),'planner_median_s':float(np.median(wall)),'planner_p90_s':float(np.percentile(wall,90)),
            'audit_rejections':sum(a.get('audited',False) and not a.get('accepted',False) for r in rows for a in r.get('attempts',[])),'solver_calls':sum(len(r.get('attempts',[])) for r in rows),'distinct_selected_pairs':sorted({r['selected_candidate']+'/'+r['selected_goal'] for r in rows if r['planning_success']})}
    comparisons={}
    for m in ['enumerate_gcs','portal_socp']:
        pairs=[]
        for k,a in lookup['joint_gcs'].items():
            b=lookup[m][k]
            if a['planning_success'] and b['planning_success']:
                pairs.append({'cell':k,'joint_cost':a['cost'],'baseline_cost':b['cost'],'joint_relative_percent':100*(a['cost']/b['cost']-1),
                              'joint_pair':[a['selected_candidate'],a['selected_goal']],'baseline_pair':[b['selected_candidate'],b['selected_goal']]})
        v=[r['joint_relative_percent'] for r in pairs]
        comparisons[m]={'pairs':pairs,'median_joint_relative_percent':float(np.median(v)) if v else None,'range':[min(v),max(v)] if v else None,
            'joint_better':sum(x<-.01 for x in v),'within_0_01_percent':sum(abs(x)<=.01 for x in v),'baseline_better':sum(x>.01 for x in v)}
    library=[]
    for f in sorted(LIB.glob('*/library.json')):
        r=json.loads(f.read_text());library.append({'grasp':f.parent.name,'attempts':len(r['attempts']),'valid_goals':len(r['goals']),'regions':len(r['candidate']['cover']['regions']),
            'endpoint_iris_calls':sum(len(a.get('online',{}).get('attempts',[])) for a in r['attempts']),
            'bridge_iris_calls':sum(len(a.get('bridge',{}).get('attempts',[])) for a in r['attempts']),
            'seconds':r['seconds'],'goals':[{'id':k,'base':g['base'],'height':g['height']} for k,g in r['goals'].items()]})
    case_by_id={c['id']:c for c in protocol['cases']};reference_checks=[];physical_checks=[]
    for r in records:
        if r['planning_success']:
            trace=json.loads((Path(r['reference'])/'delivery_trace.json').read_text());base=np.array(trace[-1]['base_q']);dock=case_by_id[r['cell']]['docking']
            margin=float(min(np.min(base[:2]-dock['lower']),np.min(np.array(dock['upper'])-base[:2])))
            reference_checks.append({'cell':r['cell'],'method':r['method'],'raw_final_base':base.tolist(),'docking_margin_m':margin,'satisfied':margin>=-1e-6,'selected_pose_max_error':float(np.max(abs(base-r['selected_base'])))})
        f=ROOT/r['cell']/r['method']/'physics.json'
        if f.exists():
            d=json.loads(f.read_text());physical_checks.append({'cell':r['cell'],'method':r['method'],'target_error_m':d['actual_target_xy_error_m'],'unexpected_contacts':d['unexpected_contact_count'],'docking_satisfied':r['actual_docking_satisfied']})
    atomic_json(ROOT/'independent_goal_checks.json',{'reference':reference_checks,'physics':physical_checks})
    result={'reference_goal_checks_passed':sum(x['satisfied'] for x in reference_checks),'reference_goal_checks_total':len(reference_checks),'coupling_diagnostic':json.loads((LIB/'coupling_diagnostic.json').read_text()),'methods':methods,'comparisons':comparisons,'library':library,'completion':json.loads((ROOT/'complete.json').read_text()),'protocol_scope':protocol['scope']}
    atomic_json(ROOT/'analysis.json',result);print(json.dumps(methods),flush=True);return records,result,protocol


def main():
    from pydrake.math import RigidTransform
    from pydrake.geometry import Sphere,Rgba,Box
    from .build_scene import build_scene,set_robot_configuration
    records,analysis,protocol=analyze();out=Path('outputs/results/grasp_fix_web/planar/coupled_goals');out.mkdir(parents=True,exist_ok=True);cases={c['id']:c for c in protocol['cases']};rows=[]
    for r in records:
        directory=ROOT/r['cell']/r['method'];key=r['cell']+'_'+r['method'];dest=out/key;dest.mkdir(exist_ok=True);shutil.copy2(directory/'record.json',dest/'record.json')
        cell=cases[r['cell']];row={'key':key,'cell':r['cell'],'method':r['method'],'planning_success':r['planning_success'],'success':r['end_to_end_success'],'contact_selected':r.get('contact_selected',False),
            'failure':r.get('failure'),'candidate':r.get('selected_candidate'),'goal':r.get('selected_goal'),'base':r.get('selected_base'),'height':r.get('selected_height'),
            'docking':cell['docking']['name'],'available_pairs':r.get('available_pairs'),'cost':r.get('cost'),'seconds':r.get('plan_parent_wall_seconds'),'replay':False}
        if (directory/'physics.json').exists():
            raw=json.loads((directory/'physics.json').read_text());shutil.copy2(directory/'physics.json',dest/'physics.json')
            row['physics']={k:raw[k] for k in ['actual_target_xy_error_m','unexpected_contact_count','duration_s']};row['physics']['actual_docking_satisfied']=r['actual_docking_satisfied']
            (dest/'trace.json').write_text(json.dumps([{'time':x['time_s'],'phase':x['phase']} for x in raw['trace']]))
            if not (dest/'scene.html').exists():
                config=json.loads((Path(r['reference'])/'scene_config_snapshot.json').read_text());scene=build_scene(config=config,visualize=True)
                lo=np.array(cell['docking']['lower']);hi=np.array(cell['docking']['upper']);mid=(lo+hi)/2
                scene.meshcat.SetObject('/task/allowed_docking',Box(*(hi-lo),.015),Rgba(.1,.8,.3,.25));scene.meshcat.SetTransform('/task/allowed_docking',RigidTransform([*mid,.03]))
                for site in config.get('inspection_sites',[]):
                    name='/task/checkpoint/'+site['name'];scene.meshcat.SetObject(name,Sphere(.08),Rgba(1.,.45,0.,1.));scene.meshcat.SetTransform(name,RigidTransform([*site['xy'],.12]))
                scene.meshcat.StartRecording()
                for x in raw['trace'][::3]+[raw['trace'][-1]]:
                    set_robot_configuration(scene,x['base_q'],x['arm_q'],x['gripper_q']);scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.array(x['cup_pose'])))
                    scene.root_context.SetTime(x['time_s']);scene.diagram.ForcedPublish(scene.root_context)
                scene.meshcat.StopRecording();scene.meshcat.PublishRecording();(dest/'scene.html').write_text(scene.meshcat.StaticHtml());del scene
            row['replay']=True
        rows.append(row);print('EXPORTED',key,flush=True)
    atomic_json(out/'data.json',{'analysis':analysis,'rows':rows});shutil.copy2('src/coupled_goals_view.html',out/'index.html')


if __name__=='__main__':main()
