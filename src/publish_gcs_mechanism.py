"""Descriptive same-cover comparisons, full raw failures, and actual replays."""
import copy,json,shutil
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .benchmark_semantic_baselines import atomic_json
ROOT=Path('outputs/results/full_task_gcs_mechanism_v1')
METHODS=['product_gcs','portal_socp','ignore_inspection']


def analyze():
    raw_records=json.loads((ROOT/'summary.json').read_text())['records'];protocol=json.loads((ROOT/'protocol.json').read_text())
    from .inspection_polyline_audit import inspect_polyline
    records=copy.deepcopy(raw_records);corrections=[]
    for r in records:
        r['raw_planning_success']=r['planning_success'];r['raw_end_to_end_success']=r['end_to_end_success']
        if 'reference' not in r:continue
        ref=Path(r['reference']);config=json.loads((ref/'scene_config_snapshot.json').read_text())
        if 'inspection_sites' in config:
            trace=json.loads((ref/'delivery_trace.json').read_text());check=inspect_polyline(trace,config['inspection_sites'])
            r['corrected_inspection']=check;r['audit']['inspection']=check
            r['planning_success']=bool(r['geometric_success'] and check['satisfied']);r['audit']['accepted']=r['planning_success']
            physical=ROOT/r['cell']/r['method']/'physics.json'
            if physical.exists():r['end_to_end_success']=bool(r['planning_success'] and json.loads(physical.read_text())['complete_task_satisfied'])
            if r['raw_planning_success']!=r['planning_success']:corrections.append({'cell':r['cell'],'method':r['method'],'raw':r['raw_planning_success'],'corrected':r['planning_success'],'minimum_distance_m':check['minimum_distance_m']})
    atomic_json(ROOT/'corrected_evaluation.json',{'records':records,'corrections':corrections,'note':'Post-run common evaluator correction; raw records unchanged, no replanning or dynamics rerun'})
    methods={};by={m:{r['cell']:r for r in records if r['method']==m} for m in METHODS}
    for m,rr in by.items():
        rs=list(rr.values());contact=[r for r in rs if r.get('contact_selected')]
        methods[m]={'n':len(rs),'raw_audited_task_pass':sum(r['raw_planning_success'] for r in rs),'raw_full_contact_pass':sum(r['raw_end_to_end_success'] for r in contact),'geometry_pass':sum(r.get('geometric_success',False) for r in rs),
                    'audited_task_pass':sum(r['planning_success'] for r in rs),'contact_n':len(contact),
                    'full_contact_pass':sum(r['end_to_end_success'] for r in contact),
                    'inspection_violations':sum(not r.get('audit',{}).get('inspection',{'satisfied':True})['satisfied'] for r in rs),
                    'planner_median_s':float(np.median([r['planner_seconds'] for r in rs if 'planner_seconds' in r])),
                    'planner_p90_s':float(np.percentile([r['planner_seconds'] for r in rs if 'planner_seconds' in r],90))}
    paired=[]
    for cell,a in by['product_gcs'].items():
        b=by['portal_socp'][cell]
        if a['planning_success'] and b['planning_success']:
            paired.append({'cell':cell,'gcs_cost':a['cost'],'portal_cost':b['cost'],'gcs_relative_cost_percent':100*(a['cost']/b['cost']-1)})
    differences=[r['gcs_relative_cost_percent'] for r in paired]
    counts={'gcs_better':sum(v<-.01 for v in differences),'within_0_01_percent':sum(abs(v)<=.01 for v in differences),'portal_better':sum(v>.01 for v in differences)}
    wins=sum(by['product_gcs'][k]['planning_success'] and not by['portal_socp'][k]['planning_success'] for k in by['product_gcs'])
    losses=sum(by['portal_socp'][k]['planning_success'] and not by['product_gcs'][k]['planning_success'] for k in by['product_gcs'])
    result={'evaluator_corrections':corrections,'methods':methods,'paired_costs':paired,'cost_comparison':{**counts,'median_gcs_relative_percent':float(np.median(differences)) if differences else None,'range_gcs_relative_percent':[min(differences),max(differences)] if differences else None},
            'paired_success':{'gcs_only':wins,'portal_only':losses,'descriptive_exact_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.},
            'scope':protocol['scope'],'completion':json.loads((ROOT/'complete.json').read_text())}
    atomic_json(ROOT/'analysis.json',result);print(json.dumps(result['methods']),flush=True);return records,result


def main():
    from pydrake.math import RigidTransform
    from pydrake.geometry import Sphere,Rgba
    from .build_scene import build_scene,set_robot_configuration
    records,analysis=analyze();out=Path('outputs/results/grasp_fix_web/planar/gcs_mechanism');out.mkdir(parents=True,exist_ok=True);rows=[]
    for r in records:
        directory=ROOT/r['cell']/r['method'];key=r['cell']+'_'+r['method'];dest=out/key;dest.mkdir(exist_ok=True)
        shutil.copy2(directory/'record.json',dest/'record.json');atomic_json(dest/'evaluation.json',r)
        row={'key':key,'cell':r['cell'],'method':r['method'],'planning_success':r['planning_success'],'success':r['end_to_end_success'],
             'contact_selected':r.get('contact_selected',False),'failure':r.get('failure'),'candidate':r.get('selected_candidate'),
             'cost':r.get('cost'),'planner_seconds':r.get('planner_seconds'),'replay':False}
        if (directory/'physics.json').exists():
            raw=json.loads((directory/'physics.json').read_text());shutil.copy2(directory/'physics.json',dest/'physics.json')
            row['physics']={k:raw[k] for k in ['complete_task_satisfied','unexpected_contact_count','actual_target_xy_error_m','duration_s']}
            row['physics']['actual_inspection_satisfied']=r['actual_inspection_satisfied']
            row['physics']['inspection_checks']=raw.get('actual_inspection_checks',[])
            (dest/'trace.json').write_text(json.dumps([{'time':x['time_s'],'phase':x['phase']} for x in raw['trace']]))
            if not (dest/'scene.html').exists():
                config=json.loads((Path(r['reference'])/'scene_config_snapshot.json').read_text());scene=build_scene(config=config,visualize=True)
                for site in config.get('inspection_sites',[]):
                    name='/task/checkpoint/'+site['name'];scene.meshcat.SetObject(name,Sphere(.08),Rgba(1.,.45,0.,1.));scene.meshcat.SetTransform(name,RigidTransform([*site['xy'],.12]))
                scene.meshcat.StartRecording()
                for x in raw['trace'][::3]+[raw['trace'][-1]]:
                    set_robot_configuration(scene,x['base_q'],x['arm_q'],x['gripper_q']);scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.array(x['cup_pose'])))
                    scene.root_context.SetTime(x['time_s']);scene.diagram.ForcedPublish(scene.root_context)
                scene.meshcat.StopRecording();scene.meshcat.PublishRecording();(dest/'scene.html').write_text(scene.meshcat.StaticHtml());del scene
            row['replay']=True
        rows.append(row);print('EXPORTED',key,flush=True)
    atomic_json(out/'data.json',{'analysis':analysis,'rows':rows});shutil.copy2('src/gcs_mechanism_view.html',out/'index.html')


if __name__=='__main__':main()
