import numpy as np

from src.compare_carry_planners import retime, LIMITS, PHYSICAL_DT


def test_retiming_preserves_corners_and_respects_common_limits():
    start=np.zeros(10);corner=start.copy();corner[0]=1.
    end=corner.copy();end[3]=.7
    q,metrics=retime([start,(start+corner)/2,corner,end])
    np.testing.assert_allclose(q[0],start)
    np.testing.assert_allclose(q[-1],end)
    assert any(np.allclose(row,corner) for row in q)
    assert np.all((q[:,0]<1.-1e-9) <= (np.abs(q[:,3])<1e-9))
    assert np.max(np.abs(np.diff(q,axis=0))/PHYSICAL_DT/LIMITS)<=1.+1e-8
    assert metrics['knots']==3


def test_retiming_keeps_reversal_not_shortcut():
    a=np.zeros(10);b=a.copy();b[1]=1.
    q,metrics=retime([a,b,a])
    assert any(np.allclose(row,b) for row in q)
    np.testing.assert_allclose(q[-1],a)
    assert metrics['knots']==3
