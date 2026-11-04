import numpy as np
from pydrake.geometry import Box, Cylinder
from pydrake.math import RigidTransform, RollPitchYaw
from scripts.verify_transport_semantics_cohort import interior_points, inside_points, object_residuals
from src.object_exclusion import shape_box_residual


def test_arm_material_witness_is_strictly_inside_original_zone():
    shape=Box(.2,.2,.2);points=interior_points(shape)
    zone=dict(lower_xyz=[.4,.4,.4],upper_xyz=[.6,.6,.6])
    hits,indices=inside_points(points,RigidTransform([.5,.5,.5]),zone)
    assert len(hits)>0
    assert np.all(np.abs(points[indices])<.1)
    assert np.all(hits>.4) and np.all(hits<.6)
    assert len(inside_points(points,RigidTransform([2.,2.,2.]),zone)[0])==0


def test_cup_exclusion_can_pass_while_arm_material_enters_same_box():
    zone=dict(id='cup_only',object_id='cup',lower_xyz=[.4,.4,.4],upper_xyz=[.6,.6,.6],margin_m=.02)
    spec=dict(object_exclusions=[zone],object_corridors=[])
    geom={'cup':[(Cylinder(.05,.15),RigidTransform())]}
    values=object_residuals(geom,{'cup':RigidTransform([.5,.5,.9]).GetAsMatrix4()},spec,'a.carry')
    assert values['cup_only']>0
    assert len(inside_points(interior_points(Box(.1,.1,.1)),RigidTransform([.5,.5,.5]),zone)[0])>0


def test_independent_rotated_cylinder_residual_and_phase_activation():
    shape=Cylinder(.05,.15)
    zone=dict(id='excluded',object_id='cup',lower_xyz=[.4,.4,.4],upper_xyz=[.6,.6,.6],margin_m=.02)
    corridor=dict(id='corridor',object_id='cup',action_id='move',mode='carry',margin_m=.02,
                  cells=[dict(lower_xyz=[0.,0.,.7],upper_xyz=[1.,1.,1.2])])
    spec=dict(object_exclusions=[zone],object_corridors=[corridor])
    for angles in ([0.,0.,0.],[.3,-.2,.6],[1.2,.9,-.7]):
        pose=RigidTransform(RollPitchYaw(angles),[.5,.5,.9])
        values=object_residuals({'cup':[(shape,RigidTransform())]}, {'cup':pose.GetAsMatrix4()},spec,'move.carry')
        expected=shape_box_residual(shape,pose,np.array(zone['lower_xyz']),np.array(zone['upper_xyz']),.02)
        assert abs(values['excluded']-expected)<1e-12
        assert values['corridor']>0
        inactive=object_residuals({'cup':[(shape,RigidTransform())]}, {'cup':pose.GetAsMatrix4()},spec,'move.reach')
        assert 'corridor' not in inactive
