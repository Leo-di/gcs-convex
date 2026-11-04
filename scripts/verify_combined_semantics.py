"""Independent cylinder projections and physical region witnesses for ablations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pydrake.geometry import Cylinder
from pydrake.math import RigidTransform
from src.build_scene import build_scene
from src.stateful_transport_grounding import make_config


def conflict_cross_section(root):
    """Check a blocking slice analytically in the fixed-orientation adapter.

    A cylinder fits a cell iff its center belongs to that cell eroded by
    its coordinate supports and corridor margin. Exclusion fails when the
    center lies strictly inside the box expanded by those same supports
    and the exclusion margin. If every eroded cell at an intermediate x
    is inside this expanded box, no continuous center path can cross.
    """
    spec=json.loads((root/'conflict/spec.json').read_text())
    file=root/'conflict/room_corridor/run/reference.json'
    if not file.exists():return {'checked':False,'reason':'Missing single-constraint reference'}
    frames=[f for f in json.loads(file.read_text()) if f['mode']=='carry']
    scene=build_scene(config=make_config(spec),visualize=False);inspector=scene.scene_graph.model_inspector()
    corridor,zone=spec['object_corridors'][0],spec['object_exclusions'][0]
    gids=scene.plant.GetCollisionGeometriesForBody(scene.object_bodies[corridor['object_id']])
    if len(gids)!=1 or not isinstance(inspector.GetShape(gids[0]),Cylinder):
        return {'checked':False,'reason':'Certificate supports one cylinder only'}
    shape=inspector.GetShape(gids[0]);X_BG=inspector.GetPoseInFrame(gids[0])
    poses=[RigidTransform(np.array(f['world_objects'][corridor['object_id']]))@X_BG for f in frames]
    R=poses[0].rotation().matrix()
    if any(np.max(abs(p.rotation().matrix()-R))>1e-10 for p in poses):
        return {'checked':False,'reason':'Reference orientation is not fixed'}
    h=shape.radius()*np.sqrt(R[:,0]**2+R[:,1]**2)+shape.length()/2*np.abs(R[:,2])
    blo=np.array(zone['lower_xyz'])-h-zone['margin_m'];bhi=np.array(zone['upper_xyz'])+h+zone['margin_m']
    x=(zone['lower_xyz'][0]+zone['upper_xyz'][0])/2
    cells=[]
    for i,cell in enumerate(corridor['cells']):
        lo=np.array(cell['lower_xyz'])+h+corridor['margin_m'];hi=np.array(cell['upper_xyz'])-h-corridor['margin_m']
        if np.any(lo>hi) or not lo[0]<=x<=hi[0]:continue
        blocked=bool(blo[0]<x<bhi[0] and np.all(lo[1:]>blo[1:]+1e-8) and np.all(hi[1:]<bhi[1:]-1e-8))
        cells.append(dict(cell=i,allowed_center_lower_yz_m=lo[1:].tolist(),allowed_center_upper_yz_m=hi[1:].tolist(),blocked=blocked))
    separated=bool(min(poses[0].translation()[0],poses[-1].translation()[0])<x<max(poses[0].translation()[0],poses[-1].translation()[0]))
    return dict(checked=True,blocking_slice=bool(separated and cells and all(c['blocked'] for c in cells)),
        shape_center_x_m=x,axis_supports_m=h.tolist(),expanded_exclusion_lower_m=blo.tolist(),expanded_exclusion_upper_m=bhi.tolist(),
        endpoint_centers_separated_by_slice=separated,active_cells=cells,
        scope='Analytic obstruction for this fixed-orientation, one-cell-at-a-time containment and axis-separation model; not arbitrary regrasp/rotation planning.')


def verify(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());rows=[]
    for case in protocol['study']['cases']:
        spec=json.loads((root/case['id']/'spec.json').read_text())
        scene=build_scene(config=make_config(spec),visualize=False);inspector=scene.scene_graph.model_inspector()
        geometry={}
        for key,body in scene.object_bodies.items():
            geometry[key]=[]
            for gid in scene.plant.GetCollisionGeometriesForBody(body):
                shape=inspector.GetShape(gid)
                if not isinstance(shape,Cylinder):raise ValueError('Independent checker supports cylinder objects only')
                points=np.array([[.99*shape.radius()*np.cos(t),.99*shape.radius()*np.sin(t),z]
                    for t in np.linspace(0,2*np.pi,96,endpoint=False)
                    for z in (-.49*shape.length(),0.,.49*shape.length())])
                geometry[key].append((shape,inspector.GetPoseInFrame(gid),points))
        for method in case['methods']:
            folder=root/case['id']/method
            file=folder/'run/physics.json'
            if not file.exists():
                rows.append(dict(case=case['id'],method=method,execution_present=False));continue
            trace=json.loads(file.read_text())['trace'];shadow=json.loads((folder/'shadow_trace.json').read_text())
            assert len(trace)==len(shadow)
            witnesses={'exclusion':[],'corridor':[]};values={'exclusion':[],'corridor':[]}
            for f,stored in zip(trace,shadow):
                transformed={}
                for key,geometries in geometry.items():
                    transformed[key]=[]
                    for shape,X_BG,points in geometries:
                        pose=RigidTransform(np.array(f['objects'][key]['pose']))@X_BG
                        R,c=pose.rotation().matrix(),pose.translation()
                        h=shape.radius()*np.sqrt(R[:,0]**2+R[:,1]**2)+shape.length()/2*np.abs(R[:,2])
                        transformed[key].append((c,h,points@R.T+c))
                for kind,spec_key in [('exclusion','object_exclusions'),('corridor','object_corridors')]:
                    active=[]
                    for constraint in spec[spec_key]:
                        if kind=='corridor' and f['phase']!=constraint['action_id']+'.carry':continue
                        shape_values=[]
                        for c,h,points in transformed[constraint['object_id']]:
                            if kind=='exclusion':
                                lo,hi=np.array(constraint['lower_xyz']),np.array(constraint['upper_xyz'])
                                shape_values.append(float(max(np.max(c-h-hi),np.max(lo-c-h))-constraint['margin_m']))
                                material=points[np.all((points>lo+1e-6)&(points<hi-1e-6),axis=1)]
                            else:
                                lo=np.array([v['lower_xyz'] for v in constraint['cells']]);hi=np.array([v['upper_xyz'] for v in constraint['cells']])
                                shape_values.append(np.minimum(c-h-lo,hi-c-h).min(axis=1)-constraint['margin_m'])
                                outside=np.maximum(lo[:,None,:]-points[None,:,:],points[None,:,:]-hi[:,None,:]).max(axis=2).min(axis=0)
                                material=points[outside>1e-6]
                            if len(material) and len(witnesses[kind])<10:
                                witnesses[kind].append(dict(time_s=f['time_s'],phase=f['phase'],constraint=constraint['id'],point_world_m=material[0].tolist()))
                        active.append(min(shape_values) if kind=='exclusion' else float(np.min(shape_values,axis=0).max()))
                    residual=min(active) if active else None
                    expected=stored[kind]['minimum_residual_m']
                    assert (residual is None and expected is None) or (residual is not None and expected is not None and abs(residual-expected)<1e-12)
                    if residual is not None:values[kind].append(residual)
            rows.append(dict(case=case['id'],method=method,execution_present=True,independent_residual_agrees=True,
                constraints={k:dict(samples=len(v),violations=sum(x< -1e-8 for x in v),minimum_residual_m=min(v),
                    actual_material_witnesses=witnesses[k]) for k,v in values.items()}))
    result={'rows':rows,'conflict_cross_section':conflict_cross_section(root),
        'scope':'Exact cylinder axis projections independently match residuals at saved times. Material points prove actual region intrusion/exit when present; negative buffered residual alone does not.'}
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2))
    print([(r['case'],r['method'],{k:(v['violations'],len(v['actual_material_witnesses'])) for k,v in r.get('constraints',{}).items()}) for r in rows])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);verify(p.parse_args().root)
