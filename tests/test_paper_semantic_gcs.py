from copy import deepcopy
import numpy as np
from pydrake.geometry.optimization import HPolyhedron,Hyperellipsoid
from src.paper_gcs import Config,solve,audit
from src.paper_semantic_gcs import solve as product_solve,region_search


def test_exact_disk_c2_model():
    regions=[Hyperellipsoid.MakeHypersphere(1.,np.zeros(2))];start=np.array([-.4,0.]);goal=np.array([.4,0.]);limits=np.array([.3,.3])
    info,segments=solve(regions,start,goal,limits)
    assert info['solved'] and audit(segments,regions,start,goal,limits)['passed']


def library():
    def row(x,phase):return {'base_q':[x,0.,0.],'arm_q':[0.]*7,'phase':phase}
    box=HPolyhedron.MakeBox(np.full(10,-2.),np.full(10,2.))
    d={'config':{},'disks':[{'center':[0.,0.],'radius':1.5}],'candidates':{},'goals':{}}
    for name,x in [('a',.2),('b',.4)]:
        d['candidates'][name]={'rows':[row(0.,'navigate_to_pickup'),row(x,'arm_to_pregrasp'),row(x,'lift_cup')],'guards':{'pick':{'valid':1.}},'cover':{'regions':[{'A':box.A().tolist(),'b':box.b().tolist()}]}}
        d['goals'][name]={g:{'rows':[row(y,'lower_into_bin'),row(y,'retreat_arm')],'guards':{'place':{'valid':1.}}} for g,y in [('near',.7),('far',1.)]}
    return d


def test_joint_semantics_and_identical_restriction_baseline():
    d=library();g=product_solve(d);b=region_search(d,route_cap=10)
    assert g['solved'] and b['solved']
    assert len(g['legal_branches'])==4
    assert abs(g['objective']-b['objective'])<1e-4
    assert [p['kind'] for p in g['path'] if p['kind'] in ['pick','place']]==['pick','place']
    chosen=g['selected_candidate'];other=product_solve(d,allowed=[c for c in d['candidates'] if c!=chosen])
    assert other['solved'] and other['selected_candidate']!=chosen
    assert not product_solve(d,allowed=[])['solved']
    blocked=deepcopy(d)
    for goals in blocked['goals'].values():goals['near']['guards']['place']['valid']=-1.
    assert product_solve(blocked)['selected_goal']=='far'
