import numpy as np
from src.geometry_outbound_gcs import make_cover, _se2_yaw_slices, select_navigation_arm


def test_select_navigation_arm_is_clear_at_query_points():
    from src.build_scene import build_scene
    from src.scene_slice_gcs import SliceOracle
    scene = build_scene(visualize=False)
    arm = select_navigation_arm(scene, [[-3.0, 0.0, 0.0], [1.2, -0.1, 0.0]])
    oracle = SliceOracle(scene, arm, scene.gripper_open_positions)
    for q in [[-3.0, 0.0, 0.0], [1.2, -0.1, 0.0]]:
        oracle.yaw = q[2]
        distance, self_distance, _ = oracle.query(np.array(q[:2]))
        assert distance > 0.0
        assert self_distance >= 0.0


def test_select_navigation_arm_falls_back_to_home_when_all_candidates_blocked():
    from src.build_scene import build_scene
    scene = build_scene(visualize=False)
    # Base placed inside the left wall: every candidate posture collides.
    arm = select_navigation_arm(scene, [[-3.95, 2.8, 0.0]])
    np.testing.assert_allclose(arm, scene.arm_home_positions)


def test_current_obstacle_changes_cover_and_collision_point_is_excluded():
    boxes=[{'name':'lane','lower':[-1,-1],'upper':[1,1]}]
    def query(x):return np.linalg.norm(x)-.3,1.,None
    regions,records=make_cover(query,boxes,[-.8,0],[.8,0],np.array([-1.,-1.]),np.array([1.,1.]),spacing=.4)
    assert regions and all(not r.PointInSet(np.array([0.,0.])) for r in regions)
    assert any(r.PointInSet(np.array([-.8,0.])) for r in regions)
    free,unused=make_cover(lambda x:(1.,1.,None),boxes,[-.8,0],[.8,0],np.array([-1.,-1.]),np.array([1.,1.]),spacing=.4)
    assert any(r.PointInSet(np.array([0.,0.])) for r in free)


def test_disks_are_clipped_to_semantic_lane_and_joint_limits():
    boxes=[{'name':'lane','lower':[-1,-.1],'upper':[1,.1]}]
    regions,records=make_cover(lambda x:(2.,1.,None),boxes,[-.4,0],[.4,0],np.array([-.5,-1.]),np.array([.5,1.]),spacing=.2)
    assert regions
    assert all(not r.PointInSet(np.array([0.,.11])) for r in regions)
    assert all(not r.PointInSet(np.array([.51,0.])) for r in regions)


def test_se2_yaw_slices_window_and_endpoints():
    boxes=[{'yaw_bounds':[-0.6,0.6]},{'yaw_bounds':[0.0,2.0]}]
    slices=_se2_yaw_slices(boxes,0.0,0.0,0.6,0.9)
    assert 0.0 in slices
    assert all(-0.9-1e-9<=s<=0.9+1e-9 for s in slices)
    # Start and goal yaw are always present even if not on the step grid.
    assert _se2_yaw_slices(boxes,0.3,0.3,0.6,0.9).count(0.3)==1
    # Window is capped by the narrowest lane on both sides.
    assert min(slices)>= -0.6-1e-9 and max(slices)<=0.9+1e-9


def test_intersection_pruning_matches_independent_drake_feasibility():
    from pydrake.geometry.optimization import Hyperellipsoid,HPolyhedron,Intersection
    from src.geometry_outbound_gcs import clipped_disks_intersect
    rng=np.random.default_rng(62)
    for _ in range(24):
        records=[];sets=[]
        for j in range(2):
            c=rng.uniform(-1,1,2);radius=float(rng.uniform(.2,1));lo=rng.uniform(-1,0,2);hi=lo+rng.uniform(.1,1.5,2)
            records.append({'center':c,'radius':radius,'lower':lo,'upper':hi})
            sets.append(Intersection(Hyperellipsoid.MakeHypersphere(radius,c),HPolyhedron.MakeBox(lo,hi)))
        assert clipped_disks_intersect(*records)==sets[0].IntersectsWith(sets[1])
