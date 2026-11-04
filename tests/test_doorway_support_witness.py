import numpy as np
import pytest
from pydrake.geometry import Box,Cylinder,Capsule,Sphere
from scripts.plot_doorway_violation_detail import support_point
from src.whole_robot_room import shape_support


@pytest.mark.parametrize('shape',[Box(.3,.2,.4),Cylinder(.2,.5),Capsule(.2,.5),Sphere(.2)])
def test_support_witness_matches_exact_support(shape):
    rng=np.random.default_rng(25)
    for direction in [*np.eye(3),*-np.eye(3),*rng.normal(size=(25,3))]:
        direction=direction/np.linalg.norm(direction)
        point=support_point(shape,direction)
        assert np.dot(direction,point)==pytest.approx(shape_support(shape,direction),abs=1e-12)
