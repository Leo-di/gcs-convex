"""Publish every new physical condition, including missing plans and failures."""
import json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json
ROOT=Path('outputs/results/transfer_scene_evaluation_v1')
METHODS=['joint_10','joint_50','global_portal']


def analyze():
    records=json.loads((ROOT/'summary.json').read_text())['records'];protocol=json.loads((ROOT/'protocol.json').read_text());methods={};lookup={m:{r['cell']:r for r in records if r['method']==m} for m in METHODS}
    for m,rr in lookup.items():
        rows=list(rr.values());physical=[r for r in rows if r['contact_selected']];passed=[r for r in rows if r['planning_success']]
        methods[m]={'n':len(rows),'planning_successes':len(passed),'contact_n':len(physical),'contact_successes':sum(r['end_to_end_success'] for r in physical),'physics_executions':sum((ROOT/r['cell']/m/'physics.json').exists() for r in physical),
            'all_query_median_s':float(np.median([r['plan_parent_wall_seconds'] for r in rows])),
            'solved_query_median_s':float(np.median([r['plan_parent_wall_seconds'] for r in passed])) if passed else None,
            'solved_query_p90_s':float(np.percentile([r['plan_parent_wall_seconds'] for r in passed],90)) if passed else None,
            'failures':dict(Counter(r.get('failure','unspecified') for r in rows if not r['planning_success'] or (r['contact_selected'] and not r['end_to_end_success'])))}
    comparisons={}
    for left,right in [('joint_10','global_portal'),('joint_50','global_portal'),('joint_50','joint_10')]:
        pairs=[]
        for key,a in lookup[left].items():
            b=lookup[right][key]
            if a['planning_success'] and b['planning_success']:
                pairs.append({'cell':key,'left_cost':a['cost'],'right_cost':b['cost'],'relative_percent':100*(a['cost']/b['cost']-1)})
        vals=[x['relative_percent'] for x in pairs]
        comparisons[left+'_vs_'+right]={'pairs':pairs,'n':len(pairs),'median_percent':float(np.median(vals)) if vals else None,'range':[min(vals),max(vals)] if vals else None,'left_better':sum(x<-.01 for x in vals),'within_0_01_percent':sum(abs(x)<=.01 for x in vals),'right_better':sum(x>.01 for x in vals)}
    spec=json.loads((ROOT/'design.json').read_text());scenes=[]
    for name,change in spec['changes'].items():
        p=json.loads((ROOT/'libraries'/name/'preparation.json').read_text());scenes.append({'name':name,'change':change,**p})
    cases={c['id']:c for c in protocol['cases']};checks=[];physics=[]
    for r in records:
        if r['planning_success']:
            trace=json.loads((Path(r['reference'])/'delivery_trace.json').read_text());base=np.array(trace[-1]['base_q']);dock=cases[r['cell']]['docking'];margin=float(min(np.min(base[:2]-dock['lower']),np.min(np.array(dock['upper'])-base[:2])))
            checks.append({'cell':r['cell'],'method':r['method'],'margin':margin,'satisfied':margin>=-1e-6,'selected_error':float(np.max(abs(base-r['selected_base'])))})
        p=ROOT/r['cell']/r['method']/'physics.json'
        if p.exists():
            d=json.loads(p.read_text());physics.append({'cell':r['cell'],'method':r['method'],'target_error_m':d['actual_target_xy_error_m'],'unexpected_contacts':d['unexpected_contact_count'],'docking_satisfied':r.get('actual_docking_satisfied'),'complete_task_satisfied':r['end_to_end_success']})
    atomic_json(ROOT/'independent_checks.json',{'reference':checks,'physics':physics})
    result={'methods':methods,'comparisons':comparisons,'scenes':scenes,'completion':json.loads((ROOT/'complete.json').read_text()),'scope':protocol['scope'],'reference_checks':{'n':len(checks),'passed':sum(x['satisfied'] for x in checks)}}
    atomic_json(ROOT/'analysis.json',result);return records,result,protocol


def main():
    from pydrake.math import RigidTransform
    from pydrake.geometry import Sphere,Rgba,Box
    from .build_scene import build_scene,set_robot_configuration
    records,analysis,protocol=analyze();out=Path('outputs/results/grasp_fix_web/planar/transfer_scenes');out.mkdir(parents=True,exist_ok=True);cases={c['id']:c for c in protocol['cases']};rows=[]
    for r in records:
        directory=ROOT/r['cell']/r['method'];key=r['cell']+'_'+r['method'];dest=out/key;dest.mkdir(exist_ok=True);shutil.copy2(directory/'record.json',dest/'record.json')
        cell=cases[r['cell']];row={'key':key,'cell':r['cell'],'method':r['method'],'planning_success':r['planning_success'],'success':r['end_to_end_success'],'contact_selected':r.get('contact_selected',False),
            'scene':cell['scene'],'failure':r.get('failure'),'candidate':r.get('selected_candidate'),'goal':r.get('selected_goal'),'base':r.get('selected_base'),'height':r.get('selected_height'),
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
    for item in analysis['scenes']:
        dest=out/'scenes'/item['name'];dest.mkdir(parents=True,exist_ok=True)
        config=json.loads((ROOT/'design.json').read_text())['scenes'][item['name']]
        if not (dest/'scene.html').exists():
            scene=build_scene(config=config,visualize=True);set_robot_configuration(scene,config['robot']['start'],scene.arm_home_positions,scene.gripper_open_positions);scene.diagram.ForcedPublish(scene.root_context)
            (dest/'scene.html').write_text(scene.meshcat.StaticHtml());del scene
        shutil.copy2(ROOT/'libraries'/item['name']/'preparation.json',dest/'preparation.json')
    shutil.copy2(ROOT/'preparation_matrix.png',out/'preparation_matrix.png')
    shutil.copy2(ROOT/'preparation_analysis.json',out/'preparation_analysis.json')
    atomic_json(out/'data.json',{'analysis':{**analysis,'scenes':[{k:v for k,v in s.items() if k not in ['attempts','goal_attempts','connectivity']} for s in analysis['scenes']]},'rows':rows});shutil.copy2('src/transfer_scenes_view.html',out/'index.html')


if __name__=='__main__':main()
