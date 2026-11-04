"""Room semantics must not depend on the scene's world-coordinate convention."""
import numpy as np
from src.build_scene import build_scene
from src.scene_spec import load_scene_config
from src.whole_robot_room import WholeRobotRoomOracle


def test_room_predicate_is_equivariant_under_planar_rigid_transform():
    scene=build_scene(config=load_scene_config('configs/scene_doorway_table.yaml'),visualize=False)
    original=WholeRobotRoomOracle.from_xy_bounds(scene,[.1,2.9],[-1.4,1.4],margin=.02)
    rng=np.random.default_rng(91537)
    # Transform both the robot and every region plane, retaining physical margin.
    for _ in range(24):
        angle=rng.uniform(-np.pi,np.pi);c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        shift=np.r_[rng.uniform(-4,4,2),0.]
        normals=original.normals@rotation.T
        transformed=WholeRobotRoomOracle(scene,normals,original.offsets+normals@shift,margin=.02)
        q=np.r_[rng.uniform([-.2,-.5,-.3],[1.2,.5,.3]),rng.uniform(-.8,.8,7)]
        before=original.query(q)
        moved=q.copy();moved[:2]=(rotation@np.r_[q[:2],0]+shift)[:2];moved[2]+=angle
        after=transformed.query(moved)
        for key in ('residual_m','base_residual_m','arm_gripper_residual_m'):
            np.testing.assert_allclose(after[key],before[key],atol=2e-12,rtol=0)
        assert after['satisfied']==before['satisfied']
