import numpy as np
from src.base_clearance_domains import separating_plane


def test_clearance_plane_covers_entire_yaw_interval_and_base_footprint():
    half=np.array([.30,.225]);a=np.array([1.34,1.26]);b=np.array([1.36,1.28])
    result=separating_plane([0,0],[1,1],half,-.02,.02,a,b,.015)
    assert result is not None
    normal,rhs,_=result
    for t in np.linspace(0,1,21):
        center=(1-t)*a+t*b
        assert normal@center>=rhs-1e-10
        for yaw in np.linspace(-.02,.02,101):
            rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            corners=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*half
            corners=corners@rotation.T+center
            # Independently compare all robot corners against the obstacle's
            # supporting plane. Convexity then covers both entire boxes.
            obstacle_support=np.where(normal>=0,1.,0.)@normal
            assert min(corners@normal)-obstacle_support>=.015-1e-10


def test_overlapping_base_cannot_receive_a_false_clearance_plane():
    assert separating_plane([0,0],[1,1],[.3,.225],-.02,.02,[.8,.8],[.9,.9],.015) is None


def test_plane_preserves_both_guide_endpoints_not_only_domain_center():
    # Averaging positions on opposite sides would create a spurious connection.
    assert separating_plane([0,0],[1,1],[.3,.225],0,0,[-.4,.5],[1.4,.5],.015) is None
