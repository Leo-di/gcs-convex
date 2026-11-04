import numpy as np
import pytest
from src.joint_approach_gcs import motion_bound,interpolate_q10
from src.stateful_transport_grounding import semantic_plan

def test_motion_bound_includes_translation_yaw_and_all_arm_joints():
 h=np.array([.03,.04,.01]+[.02]*7)
 assert motion_bound(h,2.)==pytest.approx(.05+2*(.01+7*.02))
 with pytest.raises(ValueError):motion_bound(np.zeros(7),2.)

def test_joint_interpolation_preserves_endpoints_and_limits_both_motion_types():
 q=np.zeros((2,10));q[1,0]=.1;q[1,3:]=.2
 p=interpolate_q10(q,.01,.02)
 assert np.array_equal(p[0],q[0]) and np.array_equal(p[-1],q[-1])
 assert np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).max()<=.01+1e-12
 assert np.abs(np.diff(p[:,2:],axis=0)).max()<=.02+1e-12
 assert np.any((np.diff(p[:,0])>0)&(np.diff(p[:,3])>0))

def test_joint_approach_fixes_only_gripper_while_preserving_spatial_invariant():
 f=dict(phase='clear_table_2.arrive',mode='joint_approach',base_q=[1,2,0],arm_q=[0]*7,gripper_q=[-.055,.055])
 p=semantic_plan({},[f]);pred=p['predicates']['clear_table_2.arrive_fixed']
 assert pred['indices']==[10,11]
 assert p['phases'][0]['invariants']==['inside','clear_table_2.arrive_fixed']
