import numpy as np
import pytest
from src.rotation_safe_bridge import scaled_radius


def test_bound_covers_simultaneous_translation_and_rotation():
    rng=np.random.default_rng(2);R=1.4;rho=scaled_radius(.18,.02)
    for _ in range(1000):
        u=rng.normal(size=3);u*=rho/np.linalg.norm(u)
        q=u/np.array([1,1,R]);angle=rng.uniform(-np.pi,np.pi);point=R*np.array([np.cos(angle),np.sin(angle)])
        rot=np.array([[np.cos(q[2]),-np.sin(q[2])],[np.sin(q[2]),np.cos(q[2])]])
        assert np.linalg.norm(q[:2]+rot@point-point)<=.16+1e-12


def test_zero_or_invalid_clearance_cannot_create_ball():
    assert scaled_radius(.01,.02)==0
    with pytest.raises(ValueError):scaled_radius(float('nan'),.02)
