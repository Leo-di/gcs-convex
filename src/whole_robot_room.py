"""Room containment of collision volumes, independent of collision filters.

A room is a convex intersection n.p <= b with unit outward normals. An XY
room is an unbounded vertical prism; floor support is deliberately separate.
"""
from __future__ import annotations

import numpy as np
from pydrake.geometry import Box, Capsule, Cylinder, Sphere

from .build_scene import set_robot_configuration


def shape_support(shape, direction):
    """Support relative to the shape origin, for a unit local direction."""
    d = np.asarray(direction, float)
    if isinstance(shape, Box):
        return float(np.abs(d) @ (np.array([shape.width(), shape.depth(), shape.height()]) / 2))
    if isinstance(shape, Sphere):
        return float(shape.radius() * np.linalg.norm(d))
    if isinstance(shape, Cylinder):
        return float(shape.radius() * np.linalg.norm(d[:2]) + shape.length() / 2 * abs(d[2]))
    if isinstance(shape, Capsule):
        return float(shape.radius() * np.linalg.norm(d) + shape.length() / 2 * abs(d[2]))
    raise TypeError(f"Unsupported collision geometry: {type(shape).__name__}")


class WholeRobotRoomOracle:
    def __init__(self, scene, normals, offsets, *, margin=0.0, face_names=None):
        self.scene = scene
        self.normals = np.asarray(normals, float)
        self.offsets = np.asarray(offsets, float)
        if (self.normals.ndim != 2 or self.normals.shape[1] != 3
                or self.offsets.shape != (len(self.normals),) or not len(self.normals)
                or not np.all(np.isfinite(self.normals)) or not np.all(np.isfinite(self.offsets))
                or not np.allclose(np.linalg.norm(self.normals, axis=1), 1.0)
                or not np.isfinite(margin) or margin < 0):
            raise ValueError("Finite unit room normals, matching offsets and nonnegative margin required")
        self.margin = float(margin)
        self.face_names = list(face_names) if face_names is not None else [f"face_{i}" for i in range(len(self.normals))]
        if len(self.face_names) != len(self.normals):
            raise ValueError("Each room face needs a name")
        inspector = scene.scene_graph.model_inspector()
        self.geometry = []
        for instance in (scene.base_instance, scene.arm_instance, scene.gripper_instance):
            for index in scene.plant.GetBodyIndices(instance):
                body = scene.plant.get_body(index)
                for gid in scene.plant.GetCollisionGeometriesForBody(body):
                    shape = inspector.GetShape(gid)
                    shape_support(shape, [1., 0., 0.])  # Fail closed on unsupported shapes.
                    self.geometry.append((body, shape, inspector.GetPoseInFrame(gid), inspector.GetName(gid)))
        if not self.geometry:
            raise ValueError("Robot has no collision geometry")

    @classmethod
    def from_xy_bounds(cls, scene, x_bounds, y_bounds, *, margin=0.0):
        xlo, xhi = x_bounds
        ylo, yhi = y_bounds
        if xlo >= xhi or ylo >= yhi:
            raise ValueError("Room bounds must have positive width")
        return cls(scene, [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]],
                   [-xlo, xhi, -ylo, yhi], margin=margin,
                   face_names=["x_min", "x_max", "y_min", "y_max"])

    def query_current(self):
        """Read the current plant state without changing a dynamic simulation."""
        values = []
        base_values, moving_values = [], []
        for body, shape, X_BG, name in self.geometry:
            pose = self.scene.plant.EvalBodyPoseInWorld(self.scene.plant_context, body) @ X_BG
            rotation = pose.rotation().matrix()
            for normal, offset, face in zip(self.normals, self.offsets, self.face_names):
                support = normal @ pose.translation() + shape_support(shape, rotation.T @ normal)
                values.append({"body": body.name(), "geometry": name, "face": face,
                               "residual_m": float(offset - support - self.margin)})
                (base_values if body.model_instance() == self.scene.base_instance else moving_values).append(values[-1]['residual_m'])
        worst = min(values, key=lambda item: item["residual_m"])
        return {"satisfied": worst["residual_m"] >= -1e-8,
                "residual_m": worst["residual_m"], "worst": worst,
                "base_residual_m": min(base_values),
                "arm_gripper_residual_m": min(moving_values),
                "geometry_count": len(self.geometry), "margin_m": self.margin,
                "scope": "All base/arm/gripper collision volumes; cup excluded; exact support at this configuration"}

    def query(self, state, gripper=None):
        state = np.asarray(state, float)
        if state.shape != (10,) or not np.all(np.isfinite(state)):
            raise ValueError("Expected finite (x,y,yaw,arm[7]) configuration")
        set_robot_configuration(self.scene, state[:3], state[3:],
                                self.scene.gripper_open_positions if gripper is None else gripper)
        return self.query_current()


def arm_point_reach_bound(scene, oracle):
    """Loose uniform distance from any arm joint to any downstream robot point.

    Sum fixed frame/weld translations (overcounting is safe), maximum finger
    displacement and shape reach. Used only with fixed base/yaw. Rotations
    cannot increase any translation norm; |delta p| <= R sum |delta theta|.
    """
    reach = 0.0
    for instance in (scene.arm_instance, scene.gripper_instance):
        for index in scene.plant.GetJointIndices(instance):
            joint = scene.plant.get_joint(index)
            for frame in (joint.frame_on_parent(), joint.frame_on_child()):
                reach += np.linalg.norm(frame.GetFixedPoseInBodyFrame().translation())
            if type(joint).__name__ == "WeldJoint":
                reach += np.linalg.norm(joint.X_FM().translation())
            elif type(joint).__name__ not in ("RevoluteJoint", "PrismaticJoint"):
                raise TypeError("Reach bound requires revolute/prismatic/weld arm joints")
    reach += max(np.abs(scene.gripper_open_positions).sum(), np.abs(scene.gripper_closed_positions).sum())
    radii = []
    for body, shape, pose, _ in oracle.geometry:
        if body.model_instance() == scene.base_instance:
            continue
        # Box enclosing any supported shape; its diagonal bounds the radius.
        extents = np.array([shape_support(shape, axis) for axis in np.eye(3)])
        radii.append(np.linalg.norm(pose.translation()) + np.linalg.norm(extents))
    return float(reach + max(radii))
