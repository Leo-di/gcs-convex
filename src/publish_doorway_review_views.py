"""Publish a compact whole-task Meshcat replay and a doorway side snapshot."""
import json
import sys
from pathlib import Path
import numpy as np
from pydrake.math import RigidTransform
from pydrake.geometry import Box, Rgba
from .build_scene import build_scene,set_robot_configuration

root=Path(__file__).resolve().parents[1]
d=(Path(sys.argv[1]) if len(sys.argv)>1 else root/'outputs/results/doorway_arm_reach_semantic_v1').resolve()
config=json.loads((d/'dynamics/config.json').read_text())
report=json.loads((d/'dynamics/result.json').read_text())
trace=report['trace']; grasp_point=np.asarray(config['gripper']['grasp_point_in_body'],float)
# Keep every second saved physical sample for a responsive display; the full
# physics report/audit continues to use all 3,790 samples.
scene=build_scene(config=config,visualize=True)
scene.meshcat.StartRecording()
indices=list(range(0,len(trace),2))
if indices[-1]!=len(trace)-1: indices.append(len(trace)-1)
for j,i in enumerate(indices):
 row=trace[i]
 set_robot_configuration(scene,row['base_q'],row['arm_q'],row['gripper_q'])
 scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.asarray(row['cup_pose'])))
 X_WG=scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
 ee=np.asarray(X_WG.multiply(grasp_point),float)
 marker='/semantic/arm_penetration_span'
 if ee[0]>0.01:
  scene.meshcat.SetObject(marker,Box(float(ee[0]),0.026,0.026),Rgba(1.0,0.52,0.04,0.82))
  scene.meshcat.SetTransform(marker,RigidTransform([float(ee[0]/2),float(ee[1]),float(ee[2])]))
 else:
  scene.meshcat.Delete(marker)
 scene.root_context.SetTime(j*0.05)
 scene.diagram.ForcedPublish(scene.root_context)
scene.meshcat.StopRecording(); scene.meshcat.PublishRecording()
(d/'scene_compact.html').write_text(scene.meshcat.StaticHtml())
# Snapshot the instant at maximum EE depth for the doorway side view.
max_i=0; max_x=-1e9
for i,row in enumerate(trace):
 set_robot_configuration(scene,row['base_q'],row['arm_q'],row['gripper_q'])
 X=scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
 x=float(X.multiply(grasp_point)[0])
 if x>max_x: max_i,max_x=i,x
row=trace[max_i]
side=dict(config)
side['visualization']={'global_camera':{'eye':[-.5,-7.0,7.0],'target':[-.5,0.0,1.0]}}
side_scene=build_scene(config=side,visualize=True)
set_robot_configuration(side_scene,row['base_q'],row['arm_q'],row['gripper_q'])
side_scene.plant.SetFreeBodyPose(side_scene.plant_context,side_scene.cup_body,RigidTransform(np.asarray(row['cup_pose'])))
X_WG=side_scene.gripper_body.body_frame().CalcPoseInWorld(side_scene.plant_context)
ee=np.asarray(X_WG.multiply(grasp_point),float)
if ee[0]>0.01:
 side_scene.meshcat.SetObject('/semantic/arm_penetration_span',Box(float(ee[0]),0.026,0.026),Rgba(1.0,0.52,0.04,0.82))
 side_scene.meshcat.SetTransform('/semantic/arm_penetration_span',RigidTransform([float(ee[0]/2),float(ee[1]),float(ee[2])]))
side_scene.diagram.ForcedPublish(side_scene.root_context)
(d/'doorway_side_view.html').write_text(side_scene.meshcat.StaticHtml())
# Update dashboard with the two fixed views.
from .audit_doorway_dynamics import dashboard
metrics=json.loads((d/'dynamics/doorway_semantic_audit.json').read_text())
kine=json.loads((d/'kinematic/delivery_result.json').read_text())
(d/'doorway_semantic_dashboard.html').write_text(dashboard(metrics,kine,'scene_compact.html'))
print('compact frames:',len(indices),'of',len(trace),'| side snapshot frame:',max_i,'| EE x:',max_x)
