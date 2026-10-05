"""Full-geometry semantic GCS experiment on an affine posture family.

GCS states are 10D; each region has intrinsic dimension two (base x and a
shared arm interpolation coordinate). Collision coverage is sampled. A global
kinematic Lipschitz bound, rather than sampling alone, bounds arm x-support.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pydrake.geometry import Box, Rgba
from pydrake.geometry.optimization import (
    CartesianProduct, GraphOfConvexSets, GraphOfConvexSetsOptions, HPolyhedron, Point,
)
from pydrake.math import RigidTransform
from pydrake.solvers import Binding, ClarabelSolver, Cost, L2NormCost

from .arm_boundary_semantics import ArmBoundaryOracle
from .build_scene import build_scene, set_robot_configuration
from .collision_checker import evaluate_collision_status


class PostureFamily:
    def __init__(self, scene, folded):
        self.scene = scene
        self.home = np.array(scene.arm_home_positions)
        self.delta = np.asarray(folded) - self.home
        self.oracle = ArmBoundaryOracle(scene)
        # A deliberately loose upper bound on distance from ANY arm revolute
        # joint to ANY arm/gripper collision point: sum all fixed offsets,
        # weld translations, fixed finger displacements, and largest shape reach.
        reach = 0.
        for instance in (scene.arm_instance, scene.gripper_instance):
            for index in scene.plant.GetJointIndices(instance):
                joint = scene.plant.get_joint(index)
                for frame in (joint.frame_on_parent(), joint.frame_on_child()):
                    reach += np.linalg.norm(frame.GetFixedPoseInBodyFrame().translation())
                if type(joint).__name__ == 'WeldJoint':
                    reach += np.linalg.norm(joint.X_FM().translation())
                elif type(joint).__name__ not in ('RevoluteJoint', 'PrismaticJoint'):
                    raise TypeError('Unsupported joint in reach bound')
        reach += np.abs(scene.gripper_open_positions).sum()
        reach += max(np.linalg.norm(p.translation()) +
                     (np.linalg.norm(extent) if extent is not None else radius)
                     for _, _, p, extent, radius, _ in self.oracle.geometry)
        self.reach_bound = float(reach)
        self.support_lipschitz = float(reach * np.abs(self.delta).sum())

    def state(self, x, alpha):
        return np.r_[x, 0., 0., self.home + alpha * self.delta]

    def alpha(self, state):
        return float((state[3] - self.home[0]) / self.delta[0])

    def support_bounds(self, lo, hi):
        samples = np.linspace(lo, hi, int(np.ceil((hi-lo)/.001)) + 1)
        support = [-self.oracle.query(self.state(0., a))['residual_m'] for a in samples]
        # Every alpha is at most half one grid gap from a sampled alpha.
        padding = self.support_lipschitz * np.max(np.diff(samples))/2
        return float(min(support)-padding), float(max(support)+padding)

    def support_upper_bound(self, lo, hi):
        return self.support_bounds(lo,hi)[1]

    def polytope(self, xlo, xhi, alo, ahi):
        """Embed a 2D rectangle exactly in q=(x,y,yaw,arm[7])."""
        if xhi < xlo or ahi < alo:
            raise ValueError('Empty rectangle')
        rows = []; bounds = []
        def inequality(row, bound):
            rows.append(row); bounds.append(bound)
        def equality(row, bound):
            inequality(row, bound); inequality(-row, -bound)
        eye = np.eye(10)
        inequality(eye[0], xhi); inequality(-eye[0], -xlo)
        inequality(eye[3]/self.delta[0], ahi+self.home[0]/self.delta[0])
        inequality(-eye[3]/self.delta[0], -alo-self.home[0]/self.delta[0])
        equality(eye[1], 0.); equality(eye[2], 0.)
        for j in range(1, 7):
            ratio = self.delta[j]/self.delta[0]
            equality(eye[3+j]-ratio*eye[3], self.home[j]-ratio*self.home[0])
        return HPolyhedron(np.array(rows), np.array(bounds))


def build_cover(family):
    cells = []
    rng = np.random.default_rng(3101)
    for i in range(12):
        lo, hi = i*.05, (i+1)*.05
        grid = [(x,a) for x in np.linspace(-1.4,.6,41)
                for a in np.linspace(lo,hi,6)]
        holdout = list(zip(rng.uniform(-1.4,.6,200),rng.uniform(lo,hi,200)))
        def check(points):
            values = [evaluate_collision_status(family.scene,[x,0,0],
                      arm_q=family.state(x,a)[3:]).minimum_signed_distance for x,a in points]
            return dict(samples=len(values), minimum_clearance_m=float(min(values)),
                        collisions=sum(v<0 for v in values))
        design = check(grid); audit = check(holdout)
        cells.append(dict(index=i, alpha=[lo,hi], x=[-1.4,.6],
                          support_upper_m=family.support_upper_bound(lo,hi),
                          construction=design, holdout=audit,
                          accepted=design['collisions']==0))
    return cells


def solve(family, cells, start_x, gate_x, max_alpha, aware, seed=0):
    began = time.perf_counter()
    graph = GraphOfConvexSets(); vertices = []; edges = []
    weights = np.diag([1.,1.,.35]+[.12]*7)
    def motion(region, name, rect, mode):
        vertex = graph.AddVertex(CartesianProduct([region,region]), name)
        vertex.AddCost(Binding[Cost](L2NormCost(np.c_[-weights,weights],np.zeros(10)),vertex.x()))
        vertex.AddCost(1e-6)
        item = dict(vertex=vertex, rect=rect, mode=mode)
        vertices.append(item)
        return item
    for cell in cells:
        if not cell['accepted'] or cell['alpha'][1]>max_alpha+1e-9:
            continue
        lo,hi = cell['alpha']
        for mode in ('room','transit'):
            xlo,xhi = (-1.4,gate_x) if mode=='room' else (gate_x,.6)
            if aware and mode=='room':
                xhi = min(xhi,-cell['support_upper_m']-.001)
            if xhi<xlo:
                continue
            motion(family.polytope(xlo,xhi,lo,hi),f'{mode}_{cell["index"]}',
                   [xlo,xhi,lo,hi],mode)
    start=family.state(start_x,0); goal=family.state(.6,0)
    source=graph.AddVertex(Point(np.r_[start,start]),'start')
    target=graph.AddVertex(Point(np.r_[goal,goal]),'goal')
    def edge(a,b):
        e=graph.AddEdge(a,b)
        for k in range(10):e.AddConstraint(e.xu()[10+k]==e.xv()[k])
        edges.append(e)
    def overlaps(a,b):
        return max(a[0],b[0])<=min(a[1],b[1])+1e-10 and max(a[2],b[2])<=min(a[3],b[3])+1e-10
    for i,a in enumerate(vertices):
        r=a['rect']
        if a['mode']=='room' and r[0]<=start_x<=r[1] and r[2]<=0<=r[3]:edge(source,a['vertex'])
        if a['mode']=='transit' and r[0]<=.6<=r[1] and r[2]<=0<=r[3]:edge(a['vertex'],target)
        for j,b in enumerate(vertices):
            if i==j or (a['mode']=='transit' and b['mode']=='room'):continue
            if overlaps(a['rect'],b['rect']):edge(a['vertex'],b['vertex'])
    reached={source.id()}
    for _ in range(len(vertices)+2):
        old=len(reached)
        for e in edges:
            if e.u().id() in reached:reached.add(e.v().id())
        if len(reached)==old:break
    info=dict(vertices=len(vertices)+2,edges=len(edges),build_seconds=time.perf_counter()-began)
    if target.id() not in reached:
        return dict(**info,status='unreachable_in_cover',solve_seconds=0.,path=[])
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True
    options.max_rounded_paths=10;options.rounding_seed=seed
    options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver()
    began=time.perf_counter();result=graph.SolveShortestPath(source,target,options)
    info['solve_seconds']=time.perf_counter()-began
    if not result.is_success():return dict(**info,status=str(result.get_solution_result()),path=[])
    selected=graph.GetSolutionPath(source,target,result)
    meta={v['vertex'].id():v for v in vertices};path=[]
    for e in selected:
        vertex=e.v()
        if vertex.id() not in meta:continue
        q=result.GetSolution(vertex.x())
        path.append(dict(mode=meta[vertex.id()]['mode'],name=vertex.name(),rect=meta[vertex.id()]['rect'],
                         start=q[:10].tolist(),end=q[10:].tolist()))
    return dict(**info,status='solved',objective=float(result.get_optimal_cost()),path=path)


def audit_path(family, result, gate_x, start_x, max_alpha):
    rows=[];previous=None;continuity=0.;collision_min=float('inf');semantic_min=float('inf')
    family_error=0.;domain_error=0.;modes=[]
    for segment in result['path']:
        a,b=np.array(segment['start']),np.array(segment['end'])
        if previous is not None:continuity=max(continuity,float(np.max(np.abs(previous-a))))
        previous=b
        if not modes or modes[-1]!=segment['mode']:modes.append(segment['mode'])
        count=max(2,int(np.ceil(np.max(np.abs(b-a))/.005))+1)
        for t in np.linspace(0,1,count):
            q=(1-t)*a+t*b
            alpha=family.alpha(q)
            family_error=max(family_error,float(np.max(np.abs(q-family.state(q[0],alpha)))))
            domain_error=max(domain_error,-alpha,alpha-max_alpha,
                             q[0]-gate_x if segment['mode']=='room' else gate_x-q[0])
            value=family.oracle.query(q)
            collision=evaluate_collision_status(family.scene,q[:3],arm_q=q[3:])
            collision_min=min(collision_min,collision.minimum_signed_distance)
            active=segment['mode']=='room' or q[0]<=gate_x+1e-8
            if active:semantic_min=min(semantic_min,value['residual_m'])
            rows.append(dict(q=q.tolist(),mode=segment['mode'],active=bool(active),
                             residual_m=value['residual_m'],alpha=family.alpha(q)))
    if not rows:return dict(accepted=False,samples=0),rows
    start_error=float(np.max(np.abs(np.array(rows[0]['q'])-family.state(start_x,0))))
    goal_error=float(np.max(np.abs(np.array(rows[-1]['q'])-family.state(.6,0))))
    report=dict(samples=len(rows),minimum_collision_distance_m=collision_min,
                minimum_active_semantic_residual_m=semantic_min,continuity_error=continuity,
                goal_error=goal_error,start_error=start_error,family_error=family_error,
                domain_error=domain_error,modes=modes,
                max_alpha=max(r['alpha'] for r in rows))
    report['accepted']=bool(collision_min>=0 and semantic_min>=-1e-7 and
                            max(continuity,goal_error,start_error,family_error,domain_error)<1e-6 and modes==['room','transit'])
    return report,rows


def export_replay(scene, rows, destination):
    scene.meshcat.DeleteRecording()
    scene.meshcat.StartRecording(set_visualizations_while_recording=True)
    # Display linear reference interpolation at <=0.3 m/s and <=0.4 rad/s.
    timestamp=0.;previous=None
    for row in rows:
        q=np.array(row['q'])
        if previous is not None:
            timestamp+=float(np.max(np.abs(q-previous)/np.array([.3,.3,.4]+[.4]*7)))
        scene.root_context.SetTime(timestamp)
        set_robot_configuration(scene,q[:3],q[3:])
        scene.diagram.ForcedPublish(scene.root_context)
        previous=q
    scene.meshcat.StopRecording();scene.meshcat.PublishRecording()
    destination.write_text(scene.meshcat.StaticHtml())


def main():
    root=Path('outputs/results/arm_boundary_gcs_v1');root.mkdir(parents=True,exist_ok=True)
    web=Path('outputs/results/grasp_fix_web/planar/arm_gcs');web.mkdir(parents=True,exist_ok=True)
    data=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())
    scene=build_scene(config=data['config'],visualize=False)
    family=PostureFamily(scene,data['poses']['safe']['state'][3:])
    cases=[dict(name=f'x{x}_gate{g}',start_x=x,gate_x=g,max_alpha=.6,expected='positive')
           for x in (-1.3,-1.1) for g in (-.6,-.5)]
    cases.append(dict(name='limited_posture',start_x=-1.3,gate_x=-.6,max_alpha=.15,
                      expected='negative_within_fixed_posture_family'))
    protocol=dict(cases=cases,seed=0,methods=['base_only','postcheck','semantic_gcs'],
                  collision_scope='grid-filtered cover; independent random holdout; dense trajectory audit',
                  geometry='original nominal Drake scene, fixed open gripper, no carried object',
                  state='10D affine regions of intrinsic dimension 2: x and coupled arm alpha',
                  policy='arm/gripper x<=0 through room mode; one-way transit begins at base x=gate',
                  semantic_margin_m=.001,reach_bound_m=family.reach_bound,
                  support_lipschitz=family.support_lipschitz,
                  negative_gate_support_lower_m=family.support_bounds(0,.15)[0],
                  sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                           ['src/arm_boundary_gcs.py','src/arm_boundary_semantics.py','tests/fixtures/arm_boundary.json']})
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cells=build_cover(family);(root/'cover.json').write_text(json.dumps(cells,indent=2))
    print('cover',len(cells),'accepted',sum(c['accepted'] for c in cells),flush=True)
    records=[]
    for case in cases:
        baseline=None
        for method in protocol['methods']:
            if method=='postcheck':
                result=json.loads(json.dumps(baseline))
                result['reused_base_only_plan']=True
            else:
                result=solve(family,cells,case['start_x'],case['gate_x'],case['max_alpha'],method=='semantic_gcs')
                if method=='base_only':baseline=result
            audit,rows=audit_path(family,result,case['gate_x'],case['start_x'],case['max_alpha'])
            record=dict(case=case,method=method,result=result,audit=audit)
            directory=root/case['name']/method;directory.mkdir(parents=True,exist_ok=True)
            (directory/'result.json').write_text(json.dumps(record,indent=2))
            (directory/'trace.json').write_text(json.dumps(rows))
            records.append(record)
            print(case['name'],method,result['status'],audit,flush=True)
    (root/'summary.json').write_text(json.dumps(records,indent=2))
    visual=build_scene(config=data['config'],visualize=True)
    visual.meshcat.SetObject('semantic_boundary',Box(.008,1.2,2.2),Rgba(1,.15,.15,.22))
    visual.meshcat.SetTransform('semantic_boundary',RigidTransform([0,0,1.1]))
    visual.meshcat.SetCameraPose([.8,-1.8,7.],[-.6,0,.8])
    for method in ('base_only','semantic_gcs'):
        rows=json.loads((root/cases[0]['name']/method/'trace.json').read_text())
        if rows:export_replay(visual,rows,web/f'{method}.html')
    table=''
    for r in records:
        a=r['audit'];residual=a.get('minimum_active_semantic_residual_m')
        val='—' if residual is None else f'{residual*100:.2f} cm'
        table+=f'<tr><td>{r["case"]["name"]}</td><td>{r["method"]}</td><td>{r["result"]["status"]}</td><td>{val}</td><td>{a["accepted"]}</td></tr>'
    (web/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>收臂后过门：语义 GCS</title>
<style>body{font:17px system-ui;margin:24px;color:#172b4d}button{padding:12px;margin:8px}iframe{width:100%;height:650px}td,th{padding:8px;border:1px solid #ccd}table{border-collapse:collapse}</style>
<h1>原始三维场景：语义约束是否让 GCS 提前收臂？</h1>
<p>固定底盘 y、yaw；优化底盘 x 与耦合收臂程度 α。图中区域是完整 10 维构型的二维仿射子集。模型包含所有机械臂和夹爪碰撞几何。</p>
<p>任务：在底盘到达 x=−0.6 的模式切换位置前，全臂必须留在 x≤0；随后允许穿门。起终点姿态相同。红平面为语义边界。</p>
<button onclick="show('base_only')">底盘语义：未通过全臂规则的参考轨迹</button>
<button onclick="show('semantic_gcs')">全臂语义 GCS：提前收臂的参考轨迹</button>
<iframe id="scene" src="semantic_gcs.html"></iframe>
<p>动画是规划参考轨迹回放，不是动力学执行。碰撞以采样验证；全臂语义采用运动学 Lipschitz 上界收紧区域。postcheck 复用同一底盘基线并拒绝违规轨迹，不进行重规划。</p>
<table><tr><th>条件</th><th>方法</th><th>求解状态</th><th>最小有效语义 residual</th><th>参考轨迹验证通过</th></tr>'''+table+'''</table>
<p>四个正例来自同一场景；第五项仅在限定姿态族内不可行。这是机制对照，不能用于声称一般成功率或完整取放任务成功。</p>
<script>function show(x){document.getElementById('scene').src=x+'.html'}</script>''')
    print('http://localhost:7001/arm_gcs/',flush=True)


if __name__=='__main__':main()
