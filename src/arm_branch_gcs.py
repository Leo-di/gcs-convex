"""Choose between two full-arm posture families under spatial semantics."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pydrake.geometry.optimization import CartesianProduct, GraphOfConvexSets, GraphOfConvexSetsOptions, Point
from pydrake.solvers import Binding, ClarabelSolver, Cost, L2NormCost

from .arm_boundary_gcs import PostureFamily, build_cover, audit_path
from .arm_boundary_semantics import ArmBoundaryOracle
from .build_scene import build_scene


ROOT = Path('outputs/results/arm_branch_gcs_v1')
METHODS = ['base_only', 'fixed_negative_y', 'enumerate', 'joint_gcs']


def make_families(scene, fixture):
    negative = np.array(fixture['poses']['safe']['state'][3:])
    positive = negative * np.array([-1, 1, -1, 1, -1, 1, -1])
    return {'negative_y': PostureFamily(scene, negative),
            'positive_y': PostureFamily(scene, positive)}


def lateral_bounds(family, lo, hi):
    grid = np.linspace(lo, hi, int(np.ceil((hi-lo)/.001))+1)
    padding = family.support_lipschitz * np.max(np.diff(grid))/2
    bounds = {}
    for name, normal in [('positive_y', [0,1,0]), ('negative_y', [0,-1,0])]:
        oracle = ArmBoundaryOracle(family.scene, normal)
        values = [-oracle.query(family.state(0,a))['residual_m'] for a in grid]
        bounds[name] = {'lower': float(min(values)-padding), 'upper': float(max(values)+padding)}
    return bounds


def region_specs(families, covers, case, allowed, aware, transit_boundary_margin=.0001):
    specs = []
    for branch in allowed:
        for cell in covers[branch]:
            if not cell['accepted']:
                continue
            if aware and any(cell['lateral'][side]['upper']+.001 > bound
                             for side,bound in case['lateral_limits'].items()):
                continue
            lo,hi = cell['alpha']
            for mode in ('room','transit'):
                xlo,xhi = (-1.4,case['gate_x']) if mode=='room' else (case['gate_x'],.6)
                if aware and mode=='room':xhi=min(xhi,-cell['support_upper_m']-.001)
                if aware and mode=='transit' and cell['support_upper_m']+.001>-case['gate_x']:
                    xlo+=transit_boundary_margin
                if xhi<xlo:continue
                specs.append(dict(branch=branch,mode=mode,index=cell['index'],rect=[xlo,xhi,lo,hi]))
    return specs


def solve_branches(families, covers, case, allowed, aware=True, seed=0, measure_relaxation=False, transit_boundary_margin=.0001):
    began=time.perf_counter();graph=GraphOfConvexSets();items=[];edges=[]
    weights=np.diag([1,1,.35]+[.12]*7)
    for spec in region_specs(families,covers,case,allowed,aware,transit_boundary_margin):
        region=families[spec['branch']].polytope(*spec['rect'])
        vertex=graph.AddVertex(CartesianProduct([region,region]),
                              f'{spec["branch"]}_{spec["mode"]}_{spec["index"]}')
        vertex.AddCost(Binding[Cost](L2NormCost(np.c_[-weights,weights],np.zeros(10)),vertex.x()))
        vertex.AddCost(1e-6)
        items.append(dict(**spec,vertex=vertex))
    family=next(iter(families.values()))
    start=family.state(case['start_x'],0);goal=family.state(.6,0)
    source=graph.AddVertex(Point(np.r_[start,start]),'start')
    target=graph.AddVertex(Point(np.r_[goal,goal]),'goal')
    def edge(a,b):
        e=graph.AddEdge(a,b)
        for i in range(10):e.AddConstraint(e.xu()[10+i]==e.xv()[i])
        edges.append(e)
    for i,a in enumerate(items):
        r=a['rect']
        if a['mode']=='room' and r[0]<=case['start_x']<=r[1] and r[2]<=0<=r[3]:edge(source,a['vertex'])
        if a['mode']=='transit' and r[0]<=.6<=r[1] and r[2]<=0<=r[3]:edge(a['vertex'],target)
        for j,b in enumerate(items):
            if i==j or a['branch']!=b['branch'] or (a['mode']=='transit' and b['mode']=='room'):continue
            s=b['rect']
            if max(r[0],s[0])<=min(r[1],s[1])+1e-10 and max(r[2],s[2])<=min(r[3],s[3])+1e-10:
                edge(a['vertex'],b['vertex'])
    reachable={source.id()}
    for _ in range(len(items)+2):
        size=len(reachable)
        for e in edges:
            if e.u().id() in reachable:reachable.add(e.v().id())
        if len(reachable)==size:break
    info=dict(vertices=len(items)+2,edges=len(edges),build_seconds=time.perf_counter()-began,
              allowed_branches=allowed,aware=aware,seed=seed)
    if target.id() not in reachable:return dict(**info,status='unreachable_in_cover',solve_seconds=0.,path=[])
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10
    options.rounding_seed=seed;options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver()
    began=time.perf_counter();result=graph.SolveShortestPath(source,target,options)
    info['solve_seconds']=time.perf_counter()-began
    if not result.is_success():return dict(**info,status=str(result.get_solution_result()),path=[])
    if measure_relaxation:
        options.max_rounded_paths=0
        relaxed_start=time.perf_counter()
        relaxed=graph.SolveShortestPath(source,target,options)
        info['extra_bound_solve_seconds']=time.perf_counter()-relaxed_start
        if relaxed.is_success():
            lower=float(relaxed.get_optimal_cost());upper=float(result.get_optimal_cost())
            info['relaxation_lower_bound']=lower
            info['signed_relative_gap']=(upper-lower)/max(abs(lower),1e-12)
            info['relative_gap_bound']=info['signed_relative_gap'] if lower<=upper else None
            info['bound_order_consistent']=lower<=upper
            info['bound_scope']='Numerical objectives on this fixed cover; no verified dual or interval arithmetic certificate'
    selected=graph.GetSolutionPath(source,target,result);meta={a['vertex'].id():a for a in items};path=[]
    for edge_ in selected:
        vertex=edge_.v()
        if vertex.id() not in meta:continue
        value=result.GetSolution(vertex.x());spec=meta[vertex.id()]
        path.append(dict(branch=spec['branch'],mode=spec['mode'],rect=spec['rect'],
                         start=value[:10].tolist(),end=value[10:].tolist()))
    branches={p['branch'] for p in path}
    if len(branches)!=1:raise RuntimeError('Unexpected switch between posture families')
    return dict(**info,status='solved',selected_branch=next(iter(branches)),
                objective=float(result.get_optimal_cost()),path=path)


def spatial_residuals(scene, state, gripper=None):
    return {name: -ArmBoundaryOracle(scene,normal).query(state,gripper)['residual_m']
            for name,normal in [('positive_y',[0,1,0]),('negative_y',[0,-1,0])]}


def audit(families, result, case):
    if not result['path']:return {'accepted':False,'samples':0},[]
    family=families[result['selected_branch']]
    report,rows=audit_path(family,result,case['gate_x'],case['start_x'],.6)
    minima={side:float('inf') for side in case['lateral_limits']}
    oracles={side:ArmBoundaryOracle(family.scene,[0,1 if side=='positive_y' else -1,0],bound)
             for side,bound in case['lateral_limits'].items()}
    for row in rows:
        residuals={side:oracle.query(row['q'])['residual_m'] for side,oracle in oracles.items()}
        row['lateral_residuals_m']=residuals
        for side,value in residuals.items():minima[side]=min(minima[side],value)
    report['minimum_lateral_residuals_m']=minima
    report['accepted']=bool(report['accepted'] and all(v>=-1e-7 for v in minima.values()))
    return report,rows


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    fixture=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())
    scene=build_scene(config=fixture['config'],visualize=False);families=make_families(scene,fixture)
    cases=[]
    for gate in (-.6,-.5):
        for policy,limits in [('free',{}),('avoid_positive_y',{'positive_y':.20}),
                              ('avoid_negative_y',{'negative_y':.20})]:
            cases.append(dict(name=f'{policy}_gate{gate}',start_x=-1.3,gate_x=gate,lateral_limits=limits,expected='positive'))
    cases.append(dict(name='both_sides_restricted',start_x=-1.3,gate_x=-.6,
                      lateral_limits={'positive_y':.20,'negative_y':.20},expected='negative_in_two_branch_cover'))
    protocol=dict(cases=cases,methods=METHODS,seed=0,
                  branches={k:(v.home+v.delta).tolist() for k,v in families.items()},
                  geometry='original nominal 3D scene, no added physical obstacle; lateral limits are semantic planes',
                  state='10D representation; union of two 2D affine posture families; branch selected once',
                  shared_start_goal=True,margin_m=.001,transit_boundary_margin_m=.0001,
                  enumeration='Solve each branch with same semantic constraints and choose least-cost audited plan',
                  execution_policy='All returned positive-case plans, including controls, are executed. Planning failures remain failures. Negative case is planning-only.',
                  semantic='full-arm x<=0 until gate; requested lateral halfspaces throughout the subtask',
                  task_design='Constructed mechanism cases after inspecting posture geometry; not random held-out task sample',
                  sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                           ['src/arm_branch_gcs.py','src/arm_boundary_gcs.py','tests/fixtures/arm_boundary.json']})
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    covers={}
    for name,family in families.items():
        cells=build_cover(family)
        for cell in cells:cell['lateral']=lateral_bounds(family,*cell['alpha'])
        covers[name]=cells
        print(name,'accepted',sum(c['accepted'] for c in cells),'holdout collisions',sum(c['holdout']['collisions'] for c in cells),flush=True)
    (ROOT/'cover.json').write_text(json.dumps(covers,indent=2))
    records=[]
    for case in cases:
        for method in METHODS:
            trials=[]
            if method=='enumerate':
                for branch in families:
                    attempt=solve_branches(families,covers,case,[branch]);check,_=audit(families,attempt,case)
                    trials.append(dict(branch=branch,result=attempt,audit=check))
                accepted=[t for t in trials if t['audit']['accepted']]
                result=copy.deepcopy(min(accepted,key=lambda t:t['result']['objective'])['result']) if accepted else dict(status='no_audited_branch',path=[])
                result['build_seconds']=sum(t['result']['build_seconds'] for t in trials)
                result['solve_seconds']=sum(t['result']['solve_seconds'] for t in trials)
            else:
                allowed=['negative_y'] if method=='fixed_negative_y' else list(families)
                result=solve_branches(families,covers,case,allowed,aware=method!='base_only')
            check,rows=audit(families,result,case)
            record=dict(case=case,method=method,result=result,audit=check,branch_trials=trials)
            directory=ROOT/case['name']/method;directory.mkdir(parents=True,exist_ok=True)
            (directory/'result.json').write_text(json.dumps(record,indent=2))
            (directory/'trace.json').write_text(json.dumps(rows))
            records.append(record)
            print(case['name'],method,result['status'],result.get('selected_branch'),check['accepted'],flush=True)
    (ROOT/'summary.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
