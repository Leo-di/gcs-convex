import numpy as np
import pytest
from pydrake.autodiffutils import InitializeAutoDiff,ExtractGradient,ExtractValue
from pydrake.math import RigidTransform
from src.whole_task_joint_gcs import contact_region,PairwiseContactClearance
from src.stateful_transport_grounding import semantic_plan

def test_contact_gate_has_four_robot_dofs_and_does_not_pin_configuration():
 class FK:
  def jacobian(self,q,target):return np.c_[np.eye(6),np.zeros((6,4))]
 region,record=contact_region(dict(q0=np.zeros(10),fk=FK(),target=RigidTransform(),name='grasp'))
 q=np.zeros(10);q[8]=.01
 assert region.PointInSet(q) and record['local_free_dimensions']==4
 q[0]=.001
 assert not region.PointInSet(q)

def test_pairwise_distances_allow_simultaneous_exact_nonpenetrating_contacts(monkeypatch):
 import src.whole_task_joint_gcs as module
 class Clearance:
  def __init__(self,q):self.q=q
  def robot_indices(self):return [1,2]
  def other_indices(self):return [3,4]
  def distances(self):return self.q[:2]
  def jacobians(self):return np.eye(2,10)
 class Checker:
  def CalcRobotClearance(self,q,threshold):return Clearance(q)
 class Mode:
  checker=Checker();scene=None;oracle=None
  mapping=staticmethod(lambda q:q)
 monkeypatch.setattr(module,'_reduced_position_indices',lambda s:np.arange(10))
 monkeypatch.setattr(module,'arm_point_reach_bound',lambda s,o:2.)
 f=PairwiseContactClearance(Mode(),dict(lower=[-.01]*10,upper=[.01]*10))
 y=f(InitializeAutoDiff(np.zeros(10)).ravel())
 assert np.array_equal(ExtractValue(y).ravel(),[0,0])
 assert np.array_equal(ExtractGradient(y),np.eye(2,10))

def test_global_task_semantics_does_not_reintroduce_fixed_base_or_fixed_arm():
 frames=[dict(phase='task.'+mode,mode=mode,joint_whole_task=True,base_q=[0,0,0],arm_q=[0]*7,gripper_q=[-.05,.05]) for mode in ['arrive','carry','lower','close','release','retreat']]
 plan=semantic_plan({},frames)
 predicates=[p for p in plan['predicates'].values() if p['kind']=='configuration_equals']
 assert all(p['indices']==[10,11] for p in predicates)
 assert len(predicates)==4
 assert all('inside' in p['invariants'] for p in plan['phases'])

def test_unmarked_legacy_modes_keep_their_original_invariants():
 frames=[dict(phase='task.arrive',mode='arrive',base_q=[0,0,0],arm_q=[0]*7,gripper_q=[-.05,.05])]
 assert semantic_plan({},frames)['predicates']['task.arrive_fixed']['indices']==list(range(3,10))
