"""Fresh physical scene libraries; all attempts and failure stages retained."""
import copy,json,time
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json
from .scene_spec import load_candidate_library_config


def design():
    base=load_candidate_library_config()
    base['pick_task']['collision_checked_pregrasp']=True;base['pick_task']['cartesian_approach']=True
    base['iris_np2_cover']['counterexample_refinement']=True
    scenes={};changes={}
    for width in [1.10,1.00]:
        name='door_'+str(round(width*100));c=copy.deepcopy(base)
        for w in c['world']['walls']:
            if w['name'] in ['shared_wall_upper','shared_wall_lower']:
                sign=1 if w['name'].endswith('upper') else -1;w['size'][1]=3-width/2;w['center'][1]=sign*(3+width/2)/2
            elif w['name']=='doorway_header':w['size'][1]=width
        scenes[name]=c;changes[name]={'door_clear_width_m':width}
    for name,center in [('cart_south',[-1.25,-.65,.41]),('cart_north',[-1.25,.85,.41])]:
        c=copy.deepcopy(base);c['objects']['mobile_cart']['center']=center;scenes[name]=c;changes[name]={'cart_center':center}
    for name,xy in [('cup_left_back',[1.53,.69]),('cup_right_back',[1.67,.69])]:
        c=copy.deepcopy(base);c['objects']['cup']['center'][:2]=xy;scenes[name]=c;changes[name]={'cup_xy':xy}
    for name,dx in [('dropoff_west',-.07),('dropoff_east',.07)]:
        c=copy.deepcopy(base)
        for obj in ['dropoff_table','dropoff_bin']:c['objects'][obj]['center'][0]+=dx
        c['task_zones']['dropoff']['center'][0]+=dx;c['delivery_task']['place_cup_center'][0]+=dx
        scenes[name]=c;changes[name]={'dropoff_table_bin_and_target_dx':dx}
    rng=np.random.default_rng(20261006);cases=[]
    for name in scenes:
        for j in range(3):
            cases.append({'id':f'{name}__q{j}','scene':name,'start':[*rng.uniform([-3.15,-.8],[-2.75,.8]).tolist(),0.],
                          'sites':None if j==0 else [{'name':['south','north'][j-1],'xy':[-2.5,[-1.,1.][j-1]]}],
                          'contact':j==0,'allowed':[f'x{round(x*100)}' for x in [1.10,1.15,1.20,1.25]],
                          'docking':{'name':'shared_goal_domain','lower':[-2.6,1.0],'upper':[-2.1,1.5]}})
    return {'scenes':scenes,'changes':changes,'cases':cases,'pickup_x':[1.10,1.15,1.20,1.25],
            'goal_x':[-2.50,-2.35,-2.20],'goal_y':1.25,'goal_heights':[.14,.22],
            'scope':'Eight prespecified physical variations; shared eager preparation uses official GCS/IRIS, so the no-GCS label concerns query solver only. Full bootstrap failure can reject an otherwise feasible grasp. No scene/candidate exclusion based on results. Not independent random scenes. Fresh geometry, actions, IRIS and empty disks for each scene; no old regions reused.'}


def prepare(root,name):
    from .build_scene import build_scene
    from .animate_delivery_demo import plan_delivery_candidate
    from .audit_replay import audit_replay
    from .semantic_gcs_experiment import guards
    from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES
    from .iris_region_generator import cover_to_dict
    from .scene_slice_gcs import SliceOracle
    from .prepare_full_task_development import change_place
    from .online_carry_iris import attach_carry_endpoints,sets
    from .online_carry_bridge import bridge
    from .online_iris_attachment import connection_status
    spec=json.loads((root/'design.json').read_text());config=spec['scenes'][name];out=root/'libraries'/name;out.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter();attempts=[];candidates={};all_goals={};goal_attempts=[]
    for ci,x in enumerate(spec['pickup_x']):
        cname=f'x{round(x*100)}';attempt={'name':cname,'base_q':[x,-.1,0.]};tick=time.perf_counter()
        try:
            scene=build_scene(config=copy.deepcopy(config),visualize=False);data=plan_delivery_candidate(scene,np.array(attempt['base_q']),attempt)
            audit=audit_replay(data['trace'],config,10)
            if not audit['satisfied']:raise RuntimeError('Shared bootstrap failed dense audit')
            cover=cover_to_dict(data['carrying_iris_cover']);removed=[i for i,r in enumerate(cover['regions']) if r['independent_holdout_audit']['collisions']]
            cover['regions']=[r for i,r in enumerate(cover['regions']) if i not in removed]
            candidate={'rows':data['trace'],'cover':cover,'guards':guards(data['trace'],config)}
            candidates[cname]=candidate;all_goals[cname]={};anchor=[r for r in data['trace'] if r['phase'] in PICK_PHASES][-1]
            attempt.update(status='accepted',known_colliding_regions_removed=removed)
        except RuntimeError as e:attempt.update(status='rejected',failure=str(e),details=getattr(e,'details',{}))
        except Exception as e:attempt.update(status='error',failure=repr(e))
        attempt['seconds']=time.perf_counter()-tick;attempts.append(attempt)
        atomic_json(out/'preparation_progress.json',{'attempts':attempts,'goal_attempts':goal_attempts});print('PREP',name,cname,attempt['status'],flush=True)
        if cname not in candidates:continue
        regions=candidate['cover']['regions']
        for gi,(gx,h) in enumerate((gx,h) for gx in spec['goal_x'] for h in spec['goal_heights']):
            gid=f'g{gi:02d}';base=[gx,spec['goal_y'],float(np.pi/2)];c=copy.deepcopy(config);c['delivery_task']['preplace_height']=h;c['delivery_task']['transfer_path'][-1]=base
            entry={'candidate':cname,'goal':gid,'base':base,'height':h,'primitive_valid':False}
            try:goal=change_place(c,candidate,placement_base=base)
            except RuntimeError as e:entry['failure']=str(e)
            else:
                entry['primitive_valid']=True;goal['base']=base;goal['height']=h;endpoint=q([r for r in goal['rows'] if r['phase'] in PLACE_PHASES][0]);seed=610000+10000*list(spec['scenes']).index(name)+100*ci+gi*3
                regions,online=attach_carry_endpoints(c,regions,anchor,q(anchor),endpoint,seed_base=seed)
                regions,bridging=bridge(c,regions,anchor,q(anchor),endpoint,seed_base=seed+200000,max_calls=2)
                entry.update(online=online,bridge=bridging);goal.pop('cover');all_goals[cname][gid]=goal
            goal_attempts.append(entry);atomic_json(out/'preparation_progress.json',{'attempts':attempts,'goal_attempts':goal_attempts})
        candidate['cover']['regions']=regions
    scene=build_scene(config=copy.deepcopy(config),visualize=False);oracle=SliceOracle(scene,scene.arm_home_positions,scene.gripper_open_positions)
    lower=np.array(config['robot']['lower_limits'][:2]);upper=np.array(config['robot']['upper_limits'][:2]);seeds=[c['start'][:2] for c in spec['cases'] if c['scene']==name]
    seeds+=[[x,-.1] for x in spec['pickup_x']]+[[-2.5,-1.],[-2.5,1.]]
    seeds += [[float(x),float(y)] for x in np.arange(lower[0]+.2,upper[0],.4) for y in np.arange(lower[1]+.2,upper[1],.4)]
    seeds += [[float(x),y] for x in np.arange(-.8,.81,.2) for y in [-.2,0.,.2]]
    disks=[];seen=set()
    for xy in seeds:
        key=tuple(np.round(xy,7))
        if key in seen:continue
        seen.add(key);distance,self_distance,closest=oracle.query(xy)
        if self_distance<0:raise RuntimeError('Fixed posture self collision')
        radius=float(min(.65,distance-.0201,np.min(np.array(xy)-lower),np.min(upper-np.array(xy))))
        if radius>=.035:disks.append({'center':xy,'radius':radius})
    connectivity=[]
    for cname,c in candidates.items():
        start=q([r for r in c['rows'] if r['phase'] in PICK_PHASES][-1]);rs=sets(c['cover']['regions'])
        for gid,goal in all_goals[cname].items():
            endpoint=q([r for r in goal['rows'] if r['phase'] in PLACE_PHASES][0]);connectivity.append({'candidate':cname,'goal':gid,**connection_status(rs,start,endpoint)})
    summary={'attempts':attempts,'goal_attempts':goal_attempts,'connectivity':connectivity,'seconds':time.perf_counter()-began,'candidate_count':len(candidates),'goal_count':sum(map(len,all_goals.values())),'disks':len(disks),'regions':sum(len(c['cover']['regions']) for c in candidates.values())}
    atomic_json(out/'shared.json',{'config':config,'candidates':candidates,'goals':all_goals,'disks':disks});atomic_json(out/'preparation.json',summary);print('PREPARED',name,summary['candidate_count'],summary['goal_count'],flush=True)
