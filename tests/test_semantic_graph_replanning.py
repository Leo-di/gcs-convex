import copy
import numpy as np
import pytest

from src.semantic_graph_replanning import compile_graph,pool_digest,replan,make_initial_proposal,violation_region_ids
from src.multibranch_payload import to_sets,terminal_path_indices
from pydrake.geometry.optimization import Hyperellipsoid,Intersection


def pool():
    return [dict(center=c,radius=r,lower=[-1,-4],upper=[5,4]) for c,r in
            [([0,0],.3),([2,0],.3),([1,.35],1.05),([1,-1.5],1.51)]]


def certify(i,record):
    return (None if i==2 else record),dict(satisfied=i!=2)


def audit(xy):
    bad=np.flatnonzero(xy[:,1]>1e-6)
    return dict(satisfied=not len(bad),violations=len(bad))


def test_lazy_repairs_an_actual_selected_branch_without_changing_pool():
    physical=pool();before=pool_digest(physical)
    geometry=replan(physical,[0,0,0],[2,0,0],certify,audit,policy='geometry')
    early=replan(physical,[0,0,0],[2,0,0],certify,audit,policy='early')
    lazy=replan(physical,[0,0,0],[2,0,0],certify,audit,policy='lazy')
    assert geometry['status']=='geometric_plan_violates_access'
    assert early['satisfied'] and early['solver_calls']==1
    assert lazy['satisfied'] and lazy['solver_calls']==2
    solved=[e for e in lazy['iterations'] if e['status'] in {'accepted','rejected'}]
    assert 2 in solved[0]['selected_physical_pool_ids']
    assert 2 not in solved[1]['selected_physical_pool_ids']
    assert before==pool_digest(physical)==early['physical_pool_sha256']==lazy['physical_pool_sha256']
    assert lazy['final_audit']['violations']==0


def test_one_solve_budget_retains_failed_path_and_never_calls_certifier():
    def forbidden(*args):raise AssertionError('No repair call after exhausted budget')
    result=replan(pool(),[0,0,0],[2,0,0],forbidden,audit,policy='lazy',max_solves=1)
    assert result['status']=='budget_exhausted' and not result['satisfied']
    assert result['path_xy'] and result['solver_calls']==1


def test_reject_certificate_that_moves_or_enlarges_a_physical_set():
    physical=pool();bad=copy.deepcopy(physical[0]);bad['radius']+=.1
    with pytest.raises(ValueError,match='only shrink'):
        compile_graph(physical,{0:(bad,{})},[0,0,0],[2,0,0])


def test_disconnected_finite_cover_reports_scope_without_solver():
    physical=pool();physical[2]['radius']=.01;physical[3]['radius']=.01
    r=replan(physical,[0,0,0],[2,0,0],certify,audit,policy='geometry')
    assert r['status']=='bounded_cover_disconnected' and r['solver_calls']==0
    assert not r['satisfied']


def test_omit_only_redundant_box_constraints_and_preserve_membership():
    records=[dict(center=[0,0],radius=1.,lower=[-2,-2],upper=[2,2]),
             dict(center=[0,0],radius=1.,lower=[-.5,-2],upper=[2,2]),
             dict(center=[0,0],radius=1.,lower=[-1+1e-12,-2],upper=[2,2])]
    sets=to_sets(records)
    assert isinstance(sets[0],Hyperellipsoid)
    assert all(isinstance(s,Intersection) for s in sets[1:])
    points=np.random.default_rng(7).uniform(-1.5,1.5,(1000,2))
    for record,s in zip(records,sets):
        exact=(np.linalg.norm(points,axis=1)<=1)&np.all(points>=record['lower'],axis=1)&np.all(points<=record['upper'],axis=1)
        assert np.array_equal(exact,[s.PointInSet(q,0.) for q in points])


def test_prune_cyclic_deadends_but_keep_both_terminal_branches():
    # Two terminal branches 0-1-3 and 0-2-3, plus a triangle attached at 1.
    undirected=[(0,1),(1,3),(0,2),(2,3),(1,4),(4,5),(5,6),(6,4)]
    edges=[edge for a,b in undirected for edge in [(a,b),(b,a)]]
    assert terminal_path_indices(7,edges,[0],[3])==[0,1,2,3]


def test_terminal_pruning_matches_exhaustive_simple_paths_on_small_graphs():
    rng=np.random.default_rng(93)
    for _ in range(40):
        n=7;undirected=[(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<.35]
        edges=[e for i,j in undirected for e in [(i,j),(j,i)]]
        adj={i:set() for i in range(n+2)}
        for i,j in undirected:adj[i].add(j);adj[j].add(i)
        for i in [0,1]:adj[n].add(i);adj[i].add(n)
        for i in [5,6]:adj[n+1].add(i);adj[i].add(n+1)
        expected=set()
        def visit(v,path):
            if v==n+1:expected.update(i for i in path if i<n);return
            for w in adj[v]-set(path):visit(w,path+[w])
        visit(n,[n])
        if expected:assert terminal_path_indices(n,edges,[0,1],[5,6])==sorted(expected)
        else:
            with pytest.raises(RuntimeError):terminal_path_indices(n,edges,[0,1],[5,6])


def test_geometry_and_lazy_use_identical_initial_data_and_reaudit(monkeypatch):
    physical=pool();start=np.array([0,0,0]);goal=np.array([2,0,0])
    original=replan(physical,start,goal,certify,audit,policy='geometry')
    first=next(e for e in original['iterations'] if e['status']=='rejected')
    proposal=make_initial_proposal(physical,start,goal,first,provenance={'kind':'test archived actual solve'})
    import src.semantic_graph_replanning as module
    solver=module.solve_graph
    def forbidden(*a,**k):raise AssertionError('Geometry must load the exact shared solution')
    monkeypatch.setattr(module,'solve_graph',forbidden)
    geometry=replan(physical,start,goal,certify,audit,policy='geometry',initial_proposal=proposal)
    assert geometry['actual_new_solver_calls']==0 and geometry['final_audit']['violations']>0
    monkeypatch.setattr(module,'solve_graph',solver)
    lazy=replan(physical,start,goal,certify,audit,policy='lazy',initial_proposal=proposal)
    event=next(e for e in lazy['iterations'] if e['status']=='rejected')
    assert event['path_xy']==geometry['path_xy']==proposal['path_xy']
    assert geometry['shared_initial_sha256']==lazy['shared_initial_sha256']
    assert lazy['satisfied'] and lazy['actual_new_solver_calls']==1


def test_shared_initial_rejects_modified_content_and_changed_endpoints():
    physical=pool();start=np.array([0,0,0]);goal=np.array([2,0,0])
    original=replan(physical,start,goal,certify,audit,policy='geometry')
    event=next(e for e in original['iterations'] if e['status']=='rejected')
    proposal=make_initial_proposal(physical,start,goal,event)
    bad=copy.deepcopy(proposal);bad['path_xy'][0][0]+=.01
    with pytest.raises(ValueError,match='content hash'):
        replan(physical,start,goal,certify,audit,policy='geometry',initial_proposal=bad)
    with pytest.raises(ValueError,match='different planning problem'):
        replan(physical,start,np.array([2.01,0,0]),certify,audit,policy='geometry',initial_proposal=proposal)


def test_directed_mapping_uses_all_bad_samples_and_all_containing_regions():
    regions=[dict(center=[i,0],radius=.6,lower=[-1,-1],upper=[5,1],physical_pool_id=i) for i in range(5)]
    xy=np.c_[np.arange(0,4.1,.5),np.zeros(9)]
    rows=[dict(sample=i,xy=q.tolist(),access_satisfied=i not in [3,5]) for i,q in enumerate(xy)]
    result=violation_region_ids(regions,list(range(5)),xy,dict(sample_results=rows),known_ids=[2])
    assert result['bad_sample_indices']==[3,5]
    assert result['bad_sample_intervals']==[[3,3],[5,5]]
    assert result['implicated_physical_pool_ids']==[1,2,3]
    assert result['new_physical_pool_ids']==[1,3]
    assert result['previously_certified_conflicts']==[2]
    rows[3]['xy']=[90,0]
    with pytest.raises(ValueError,match='coordinates'):
        violation_region_ids(regions,list(range(5)),xy,dict(sample_results=rows))


def test_directed_policy_repairs_the_real_gcs_branch_with_full_audit():
    def full_audit(xy):
        rows=[dict(sample=i,xy=q.tolist(),access_satisfied=bool(q[1]<=1e-6)) for i,q in enumerate(xy)]
        return dict(satisfied=all(r['access_satisfied'] for r in rows),geometry_violations=0,sample_results=rows)
    calls=[]
    def record_cert(i,r):calls.append(i);return certify(i,r)
    result=replan(pool(),np.array([0,0,0]),np.array([2,0,0]),record_cert,full_audit,policy='directed')
    assert result['satisfied'] and result['solver_calls']==2
    failed=next(e for e in result['iterations'] if e['status']=='rejected')
    assert calls==failed['directed_repair']['new_physical_pool_ids']
    assert set(calls)<=set(failed['selected_physical_pool_ids'])
    assert 3 not in calls  # The unused safe detour is never pre-certified.
