"""Publish matched raw/screened query results without overwriting prior runs."""
import json,shutil,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from .benchmark_semantic_baselines import atomic_json
ROOT=Path('outputs/results/screened_transfer_v1')
OUT=Path('outputs/results/grasp_fix_web/planar/screened_transfer')
METHODS=['joint_10','joint_50','global_portal']


def analyze():
    records=json.loads((ROOT/'summary.json').read_text())['records'];spec=json.loads((ROOT/'design.json').read_text());metrics={};pairs={}
    lookup={(r['condition'],r['method'],r['cell']):r for r in records}
    for condition in ['raw','screened']:
        metrics[condition]={}
        for method in METHODS:
            rr=[r for r in records if r['condition']==condition and r['method']==method];passed=[r for r in rr if r['planning_success']];physical=[r for r in rr if r['contact_selected']]
            metrics[condition][method]={'n':len(rr),'planning_passed':len(passed),'contact_n':len(physical),'contact_passed':sum(r['end_to_end_success'] for r in physical),'all_median_s':float(np.median([r['plan_parent_wall_seconds'] for r in rr])),
                'solved_median_s':float(np.median([r['plan_parent_wall_seconds'] for r in passed])) if passed else None,'failures':dict(Counter(r.get('failure','unknown') for r in rr if not r['planning_success'] or (r['contact_selected'] and not r['end_to_end_success'])))}
    for m in METHODS:
        transitions=[]
        for c in spec['cases']:
            a,b=(lookup[condition,m,c['id']] for condition in ['raw','screened']);transitions.append({'cell':c['id'],'raw':a['planning_success'],'screened':b['planning_success'],'raw_failure':a.get('failure'),'screened_failure':b.get('failure')})
        pairs[m]={'rows':transitions,'gains':sum(not x['raw'] and x['screened'] for x in transitions),'losses':sum(x['raw'] and not x['screened'] for x in transitions)}
    scenes=[];invariants=[]
    for name in spec['scenes']:
        pre=json.loads((ROOT/'raw/libraries'/name/'preparation.json').read_text());report=json.loads((ROOT/'screened/libraries'/name/'screen_report.json').read_text());raw=json.loads((ROOT/'raw/libraries'/name/'shared.json').read_text());screened=json.loads((ROOT/'screened/libraries'/name/'shared.json').read_text())
        same=all(raw[k]==screened[k] for k in ['config','disks','goals']) and list(raw['candidates'])==list(screened['candidates']) and all({k:v for k,v in c.items() if k!='cover'}=={k:v for k,v in screened['candidates'][n].items() if k!='cover'} for n,c in raw['candidates'].items());assert same,name;invariants.append({'scene':name,'same_actions_config_and_navigation':same})
        hist=[x for c in report['candidates'].values() for x in c['history']]
        scenes.append({'name':name,'candidates':pre['candidate_count'],'goals':pre['goal_count'],'regions_before':sum(x['input_regions'] for x in report['candidates'].values()),'regions_after':sum(x['output_regions'] for x in report['candidates'].values()),'original_regions_rejected':sum(not x['original_audit']['passed'] for x in hist),'regeneration_calls':sum(x['regeneration_calls'] for x in report['candidates'].values()),'regenerated_retained':sum(x['retained'] and not x['original_audit']['passed'] for x in hist),'connected_before':report['before_connected'],'connected_after':report['after_connected'],'screen_seconds':report['seconds']})
    checks=[]
    cases={c['id']:c for c in spec['cases']}
    for r in records:
        if not r['planning_success']:continue
        trace=json.loads((Path(r['reference'])/'delivery_trace.json').read_text());xy=np.array(trace[-1]['base_q'][:2]);dock=cases[r['cell']]['docking'];margin=float(min(np.min(xy-dock['lower']),np.min(np.array(dock['upper'])-xy)));checks.append({'condition':r['condition'],'cell':r['cell'],'method':r['method'],'docking_margin':margin,'passed':margin>=-1e-6})
    atomic_json(ROOT/'independent_checks.json',{'input_invariants':invariants,'reference':checks})
    comparisons={}
    for condition in ['raw','screened']:
        for m in ['joint_10','joint_50']:
            rr=[]
            for c in spec['cases']:
                a,b=lookup[condition,m,c['id']],lookup[condition,'global_portal',c['id']]
                if a['planning_success'] and b['planning_success']:rr.append({'cell':c['id'],'relative_percent':100*(a['cost']/b['cost']-1)})
            comparisons[condition+'_'+m]={'pairs':rr,'n':len(rr),'median_percent':float(np.median([x['relative_percent'] for x in rr])) if rr else None}
    analysis={'metrics':metrics,'transitions':pairs,'scenes':scenes,'comparisons':comparisons,'completion':json.loads((ROOT/'complete.json').read_text()),'scope':spec['scope']};atomic_json(ROOT/'analysis.json',analysis);return records,analysis,spec


def main():
    from pydrake.math import RigidTransform
    from .build_scene import build_scene,set_robot_configuration
    records,analysis,spec=analyze();OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for r in records:
        directory=ROOT/r['condition']/r['cell']/r['method'];key=r['condition']+'_'+r['cell']+'_'+r['method'];dest=OUT/key;dest.mkdir(exist_ok=True);shutil.copy2(directory/'record.json',dest/'record.json');row={k:r.get(k) for k in ['condition','cell','scene','method','planning_success','end_to_end_success','failure','contact_selected','selected_candidate','selected_goal','cost','plan_parent_wall_seconds']};row.update(key=key,replay=False,success=r['end_to_end_success'],candidate=r.get('selected_candidate'),goal=r.get('selected_goal'),base=r.get('selected_base'),height=r.get('selected_height'))
        p=directory/'physics.json'
        if p.exists():
            raw=json.loads(p.read_text());shutil.copy2(p,dest/'physics.json');(dest/'trace.json').write_text(json.dumps([{'time':x['time_s'],'phase':x['phase']} for x in raw['trace']]))
            row['physics']={k:raw[k] for k in ['actual_target_xy_error_m','unexpected_contact_count','duration_s']};row['physics']['actual_docking_satisfied']=r.get('actual_docking_satisfied')
            if not (dest/'scene.html').exists():
                config=json.loads((Path(r['reference'])/'scene_config_snapshot.json').read_text());scene=build_scene(config=config,visualize=True);scene.meshcat.StartRecording()
                for f in raw['trace'][::3]+[raw['trace'][-1]]:
                    set_robot_configuration(scene,f['base_q'],f['arm_q'],f['gripper_q']);scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.array(f['cup_pose'])));scene.root_context.SetTime(f['time_s']);scene.diagram.ForcedPublish(scene.root_context)
                scene.meshcat.StopRecording();scene.meshcat.PublishRecording();(dest/'scene.html').write_text(scene.meshcat.StaticHtml());del scene
            row['replay']=True
        rows.append(row);print('EXPORTED',key,flush=True)
    for s in analysis['scenes']:
        dest=OUT/'scenes'/s['name'];dest.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'screened/libraries'/s['name']/'screen_report.json',dest/'screen_report.json')
    atomic_json(OUT/'data.json',{'analysis':analysis,'rows':rows});shutil.copy2('src/screened_transfer_view.html',OUT/'index.html')


if __name__=='__main__':main()
