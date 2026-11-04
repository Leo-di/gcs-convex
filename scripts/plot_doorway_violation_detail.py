"""Show actual link collision-shape projections and exact x-extreme witnesses."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pydrake.geometry import Box,Cylinder,Capsule,Sphere
from src.build_scene import build_scene,set_robot_configuration
from src.green_room_gcs import room_oracle


def support_point(shape,d):
    d=np.asarray(d,float)
    if isinstance(shape,Box):return np.sign(d)*np.array([shape.width(),shape.depth(),shape.height()])/2
    if isinstance(shape,Cylinder):
        n=np.linalg.norm(d[:2]);xy=shape.radius()*d[:2]/n if n>1e-14 else np.zeros(2)
        return np.r_[xy,np.sign(d[2])*shape.length()/2]
    if isinstance(shape,Sphere):return shape.radius()*d/np.linalg.norm(d)
    if isinstance(shape,Capsule):return shape.radius()*d/np.linalg.norm(d)+[0,0,np.sign(d[2])*shape.length()/2]
    raise ValueError(type(shape))


def plot(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    verification=json.loads((root/'verification.json').read_text());geometries=[];evidence=[]
    for method in ('base_only','full_motion'):
        row=next(r for r in verification['rows'] if r['method']==method)
        folder=root/method/'operation_first'/row['selected']
        cfg=json.loads((folder/'scene_config_snapshot.json').read_text())
        trace=json.loads((folder/'execution/execution_grounded_candidate/physics.json').read_text())['trace']
        frame=min(trace,key=lambda f:abs(f['time_s']-row['worst_sample']['time_s']))
        scene=build_scene(config=cfg,visualize=False);set_robot_configuration(scene,frame['base_q'],frame['arm_q'],frame['gripper_q'])
        oracle=room_oracle(scene);body_name=row['worst_sample']['worst']['body']
        polygons=[];extremes=[]
        for body,shape,X_BG,name in oracle.geometry:
            if body.name()!=body_name:continue
            pose=scene.plant.EvalBodyPoseInWorld(scene.plant_context,body)@X_BG;rot=pose.rotation().matrix();origin=pose.translation()
            directions=[np.array([np.cos(t),np.sin(t),0.]) for t in np.linspace(0,2*np.pi,256,endpoint=False)]
            points=np.array([rot@support_point(shape,rot.T@d)+origin for d in directions])
            polygons.append(points[ConvexHull(points[:,:2]).vertices,:2])
            extremes.append(rot@support_point(shape,rot.T@np.array([-1.,0.,0.]))+origin)
        extreme=min(extremes,key=lambda p:p[0]);xlo=cfg['green_room']['x_bounds'][0];margin=cfg['green_room']['margin_m']
        assert abs(extreme[0]-xlo-margin-row['minimum_room_residual_m'])<1e-8
        evidence.append(dict(method=method,body=body_name,time_s=frame['time_s'],phase=frame['phase'],
            extreme_point_world_m=extreme.tolist(),room_boundary_x_m=xlo,required_margin_m=margin,
            raw_residual_m=float(extreme[0]-xlo),buffered_residual_m=float(extreme[0]-xlo-margin)))
        geometries.append(polygons)
    all_points=np.vstack([p for shapes in geometries for p in shapes]);lo=all_points.min(axis=0)-.05;hi=all_points.max(axis=0)+.05
    lo[0]=min(lo[0],.05);hi[0]=max(hi[0],.36)
    fig,axes=plt.subplots(1,2,figsize=(11,5.2),constrained_layout=True)
    for ax,shapes,e,color,title in zip(axes,geometries,evidence,['#b47a13','#187c62'],['Base containment','Whole-body containment']):
        xlo=e['room_boundary_x_m'];ax.axvspan(lo[0],xlo,color='#f9e4df');ax.axvspan(xlo,hi[0],color='#e5f2ea')
        ax.axvline(xlo,color='#ae4c38',lw=2,label='Room boundary: x = 0.10 m')
        ax.axvline(xlo+e['required_margin_m'],color='#526b5c',ls='--',lw=1.4,label='Required margin: x ≥ 0.12 m')
        for points in shapes:ax.add_patch(Polygon(points,facecolor=color,edgecolor=color,alpha=.55,lw=2))
        x,y=e['extreme_point_world_m'][:2];ax.plot(x,y,'o',color='#9e3125' if e['raw_residual_m']<0 else color,zorder=5)
        ax.annotate(f"Leftmost shape point\nx = {x:.4f} m",xy=(x,y),xytext=(.43,.82),textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->',color='#293e50'),fontsize=11)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='x (m)',ylabel='y (m)',title=title);ax.set_aspect('equal')
        ax.text(.98,.03,f"Raw room residual: {e['raw_residual_m']*1000:+.1f} mm\nWith 20 mm margin: {e['buffered_residual_m']*1000:+.1f} mm",transform=ax.transAxes,ha='right',va='bottom',fontsize=10,
                bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    fig.suptitle('The base is inside, but an arm collision volume can extend outside',fontsize=15)
    fig.supxlabel('XY projection of iiwa_link_5 at each run’s minimum-margin operation sample (times differ).\nRed line: room boundary; dashed line: required 20 mm inset.\nExtreme points use exact shape support; polygon outlines approximate the projection.',fontsize=9)
    for ext in ('png','svg','pdf'):fig.savefig(output/f'violation_detail.{ext}',dpi=180)
    plt.close(fig);(output/'violation_detail.json').write_text(json.dumps(evidence,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();plot(a.root,a.output)
