import numpy as np
import pytest
from scripts.publish_gcs_inspector import projected_envelope


def test_projection_eliminates_hidden_coordinate_and_allows_negative_joints():
    # x is negative, y <= z <= 0.5: projecting out z must keep y <= 0.5.
    A = np.array([[1,0,0],[-1,0,0],[0,1,-1],[0,-1,0],[0,0,1],[0,0,-1]])
    b = np.array([-1,2,0,1,.5,0])
    p = np.array(projected_envelope(A,b,axes=(0,1)))
    np.testing.assert_allclose(p.min(axis=0),[-2,-1],atol=1e-7)
    np.testing.assert_allclose(p.max(axis=0),[-1,.5],atol=1e-7)


def test_oblique_constraint_survives_projection():
    # 32 directions include x+y, so this triangle must not become its bbox.
    A = [[-1,0],[0,-1],[1,1]]
    p = np.array(projected_envelope(A,[0,0,1],axes=(0,1)))
    assert np.max(p.sum(axis=1)) <= 1+1e-7
    np.testing.assert_allclose(p.max(axis=0),[1,1],atol=1e-7)


def test_unbounded_projection_is_not_rendered_as_valid_region():
    with pytest.raises(ValueError,match='Cannot project'):
        projected_envelope([[-1,0]],[0],axes=(0,1))
