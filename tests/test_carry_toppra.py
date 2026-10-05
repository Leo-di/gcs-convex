import numpy as np
import pytest
pytest.importorskip("toppra")
from src.carry_toppra import BezierPath,retime


def test_exact_line_and_known_minimum_time():
    segments=[{'x':[0.,1/3,2/3,1.,0.,1/3,2/3,1.]}]
    path=BezierPath(segments)
    assert np.allclose(path(np.linspace(0,1,10),1),1)
    assert np.allclose(path(np.linspace(0,1,10),2),0)
    result=retime(segments,np.array([.5]),np.array([1.]))
    assert result['solved']
    assert abs(result['raw_duration']-2.5)<1e-4
    assert result['checked_acceleration_ratio']<=1
    assert result['checked_velocity_ratio']<=1
