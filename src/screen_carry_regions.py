"""Common numerical screening and bounded official IRIS regeneration.

Boundary probes are a project validation heuristic, not a new IRIS algorithm
or a certificate. No evaluation solver path is used to tune the screened cover.
"""
import copy,json,time
from pathlib import Path
import numpy as np
from pydrake.common import RandomGenerator
from pydrake.geometry.optimization import HPolyhedron
from .benchmark_semantic_baselines import atomic_json
from .carry_mode_gcs import CarryOracle
from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES,WEIGHTS
from .online_carry_iris import grow_carry_region,sets
from .online_iris_attachment import connection_status
SPEC={'uniform_points':64,'random_ray_directions':64,'ray_fractions':[.95,1.],
      'collision_tolerance_m':1e-7,'max_regenerations_per_grasp':4,'seed':20261007}
OLD=Path('outputs/results/transfer_scene_evaluation_v1')


def known_witnesses(scene,candidate):
    out=[]
    for w in json.loads((OLD/'failure_witnesses.json').read_text()):
        if not w['cell'].startswith(scene+'__') or w['candidate']!=candidate:continue
        trace=json.loads((Path(w['reference'])/'delivery_trace.json').read_text())
        if 'frame' in w:x=q(trace[w['frame']])
        else:x=(1-w['alpha'])*q(trace[w['edge']])+w['alpha']*q(trace[w['edge']+1])
        if not any(np.linalg.norm(x-y)<1e-7 for y in out):out.append(x)
    return out


def probe_region(record,distance,known=(),*,seed=0,spec=SPEC):
    region=HPolyhedron(np.array(record['A']),np.array(record['b']));A=region.A();b=region.b();center=region.ChebyshevCenter();counts={};minimum=float('inf')
    def check(x,kind):
        nonlocal minimum
        if not region.PointInSet(x,1e-6):raise ValueError('Probe is outside its region')
        d=float(distance(x));counts[kind]=counts.get(kind,0)+1;minimum=min(minimum,d)
        if not np.isfinite(d) or d < -spec['collision_tolerance_m']:
            return {'passed':False,'counts':dict(counts),'minimum_distance_m':minimum,'witness':{'kind':kind,'q':np.asarray(x).tolist(),'distance_m':d}}
    probes=[(center,'center')]
    anchor=np.array(record['seed'])
    if region.PointInSet(anchor,1e-7):probes.append((anchor,'seed'))
    probes.extend((np.array(x),'previous_counterexample') for x in known if region.PointInSet(np.asarray(x),1e-7))
    for x,kind in probes:
        bad=check(x,kind)
        if bad:return bad
    generator=RandomGenerator(seed);point=center
    for _ in range(spec['uniform_points']):
        point=region.UniformSample(generator,point,mixing_steps=20);bad=check(point,'uniform')
        if bad:return bad
    n=len(center);weights=WEIGHTS if n==10 else np.ones(n);rng=np.random.default_rng(seed)
    directions=list(np.eye(n))+list(-np.eye(n))+list(rng.normal(size=(spec['random_ray_directions'],n))/weights)
    slack=b-A@center
    for direction in directions:
        dot=A@direction;positive=dot>1e-12
        if not np.any(positive):raise ValueError('Expected bounded carry region')
        reach=float(np.min(slack[positive]/dot[positive]))
        for fraction in spec['ray_fractions']:
            x=center+direction*reach*fraction*(1-1e-8);bad=check(x,'boundary_ray')
            if bad:return bad
    return {'passed':True,'counts':counts,'minimum_distance_m':minimum,'witness':None}


def screen_scene(root,name,*,use_known_witnesses=True):
    raw=root/'raw/libraries'/name;dest=root/'screened/libraries'/name;dest.mkdir(parents=True,exist_ok=True);shared=json.loads((raw/'shared.json').read_text());prep=json.loads((raw/'preparation.json').read_text());begin=time.perf_counter();histories={};si=list(json.loads((root/'design.json').read_text())['scenes']).index(name)
    for ci,(candidate_name,candidate) in enumerate(shared['candidates'].items()):
        anchor=[r for r in candidate['rows'] if r['phase'] in PICK_PHASES][-1];oracle=CarryOracle(shared['config'],anchor);known=known_witnesses(name,candidate_name) if use_known_witnesses else [];accepted=[];history=[];calls=0
        distance=lambda x:oracle.query(x[:2],x[2:])[1]
        for ri,record in enumerate(candidate['cover']['regions']):
            seed=SPEC['seed']+10000*si+500*ci+ri;result=probe_region(record,distance,known,seed=seed);entry={'original_index':ri,'original_audit':result,'retained':False}
            if result['passed']:
                kept=copy.deepcopy(record);kept['common_screen']=result;kept['original_index']=ri;accepted.append(kept);entry['retained']=True
            elif calls<SPEC['max_regenerations_per_grasp']:
                calls+=1
                try:
                    replacement=grow_carry_region(shared['config'],anchor,np.array(record['seed']),record['semantic_region'],900000+10000*si+500*ci+ri)
                    entry['regeneration']=replacement
                    if replacement['accepted']:
                        checked=probe_region(replacement,distance,known,seed=seed+1000000);entry['regenerated_audit']=checked
                        if checked['passed']:
                            replacement['common_screen']=checked;replacement['original_index']=ri;replacement['common_regenerated']=True;accepted.append(replacement);entry['retained']=True
                except RuntimeError as e:entry['regeneration_error']=str(e)
            history.append(entry);atomic_json(dest/'screen_progress.json',{'completed_candidates':histories,'current_candidate':candidate_name,'history':history})
        candidate['cover']['regions']=accepted;histories[candidate_name]={'history':history,'input_regions':len(history),'output_regions':len(accepted),'regeneration_calls':calls,'known_points':len(known),'oracle_calls':oracle.calls};print('SCREEN',name,candidate_name,len(history),len(accepted),calls,flush=True)
    prep['before_screen_connectivity']=prep['connectivity'];prep['connectivity']=[]
    for cname,c in shared['candidates'].items():
        start=q([r for r in c['rows'] if r['phase'] in PICK_PHASES][-1]);regions=sets(c['cover']['regions'])
        for gid,goal in shared['goals'][cname].items():
            end=q([r for r in goal['rows'] if r['phase'] in PLACE_PHASES][0]);prep['connectivity'].append({'candidate':cname,'goal':gid,**connection_status(regions,start,end)})
    prep['regions']=sum(len(c['cover']['regions']) for c in shared['candidates'].values());prep['screen_seconds']=time.perf_counter()-begin
    report={'use_known_witnesses':use_known_witnesses,'spec':SPEC,'candidates':histories,'seconds':prep['screen_seconds'],'before_connected':sum(x['region_graph_connected'] for x in prep['before_screen_connectivity']),'after_connected':sum(x['region_graph_connected'] for x in prep['connectivity']),
            'scope':'Finite validation on old known witnesses, uniform interior points and center rays near boundaries; four official IRIS regeneration calls per grasp, then reject unverified regions. Not certification; no test-query paths used in screening.'}
    atomic_json(dest/'shared.json',shared);atomic_json(dest/'preparation.json',prep);atomic_json(dest/'screen_report.json',report)
