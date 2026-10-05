"""Whole-arm box/sphere support predicates in the original Drake scene."""
import numpy as np
from pydrake.geometry import Box,Sphere
from .build_scene import set_robot_configuration


class ArmBoundaryOracle:
    def __init__(self,scene,normal=(1.,0.,0.),offset=0.):
        self.scene=scene;self.normal=np.asarray(normal,dtype=float);self.offset=float(offset)
        if not np.isclose(np.linalg.norm(self.normal),1.):raise ValueError('Unit normal required for metric residual')
        inspector=scene.scene_graph.model_inspector();self.geometry=[]
        for instance in [scene.arm_instance,scene.gripper_instance]:
            for index in scene.plant.GetBodyIndices(instance):
                body=scene.plant.get_body(index)
                for gid in scene.plant.GetCollisionGeometriesForBody(body):
                    shape=inspector.GetShape(gid)
                    if isinstance(shape,Box):extent=np.array([shape.width(),shape.depth(),shape.height()])/2;radius=None
                    elif isinstance(shape,Sphere):extent=None;radius=shape.radius()
                    else:raise TypeError('Unsupported collision shape: '+type(shape).__name__)
                    self.geometry.append((body,gid,inspector.GetPoseInFrame(gid),extent,radius,inspector.GetName(gid)))
    def query(self,state,gripper=None):
        s=self.scene;state=np.asarray(state)
        set_robot_configuration(s,state[:3],state[3:],s.gripper_open_positions if gripper is None else gripper)
        items=[]
        for body,gid,X_BG,extent,radius,name in self.geometry:
            X=s.plant.EvalBodyPoseInWorld(s.plant_context,body)@X_BG
            support=radius if extent is None else np.abs(X.rotation().matrix().T@self.normal)@extent
            maximum=float(self.normal@X.translation()+support)
            items.append({'body':body.name(),'geometry':name,'residual_m':self.offset-maximum,'support_coordinate_m':maximum})
        worst=min(items,key=lambda x:x['residual_m'])
        ee=s.plant.EvalBodyPoseInWorld(s.plant_context,s.gripper_body).translation()
        return {'residual_m':worst['residual_m'],'satisfied':worst['residual_m']>=-1e-8,'worst':worst,
                'base_center_residual_m':float(self.offset-self.normal@s.plant.EvalBodyPoseInWorld(s.plant_context,s.base_body).translation()),
                'gripper_origin_residual_m':float(self.offset-self.normal@ee),'geometry':items,
                'scope':'Exact support of current Drake arm/gripper collision boxes and spheres; collision-model geometry, not visual mesh or continuous-time certificate.'}


def monitor(rows,oracle,activation_x=-.6,subdivisions=10):
    samples=0;violations=[];minimum=float('inf')
    for i,(a,b) in enumerate(zip(rows,rows[1:])):
        qa=np.r_[a['base_q'],a['arm_q']];qb=np.r_[b['base_q'],b['arm_q']]
        for t in np.linspace(0,1,subdivisions+1):
            state=(1-t)*qa+t*qb
            if state[0]>activation_x:continue
            value=oracle.query(state,(1-t)*np.array(a['gripper_q'])+t*np.array(b['gripper_q']))
            samples+=1;minimum=min(minimum,value['residual_m'])
            if not value['satisfied']:violations.append({'edge':i,'alpha':float(t),'state':state.tolist(),'residual_m':value['residual_m'],'body':value['worst']['body']})
    return {'satisfied':not violations,'samples':samples,'minimum_residual_m':minimum if samples else None,
            'violation_count':len(violations),'first_violation':violations[0] if violations else None,
            'worst_violation':min(violations,key=lambda v:v['residual_m']) if violations else None,
            'activation':'base_x <= '+str(activation_x),'scope':'Sampled reference segments, all arm/gripper collision volumes, x <= 0; transit strip permits room change.'}
