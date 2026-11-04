"""Reserved 3D space for people, checked against every robot collision volume.

Boxes express a user restriction, not physical walls. Axis separation is
conservative. Its residual is 1-Lipschitz under rigid translation, and changes
by at most the maximum point displacement under joint/yaw motion.
"""

import numpy as np
from .whole_robot_room import WholeRobotRoomOracle
from .object_exclusion import shape_box_residual


def validate_keep_clear(specs):
    if not isinstance(specs, list):
        raise ValueError("Robot keep-clear zones must be a list")
    ids = set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {
            "id",
            "lower_xyz",
            "upper_xyz",
            "margin_m",
        }:
            raise ValueError("Robot keep-clear needs explicit id, 3D bounds and margin")
        lo, hi = (
            np.asarray(spec["lower_xyz"], float),
            np.asarray(spec["upper_xyz"], float),
        )
        if (
            not isinstance(spec["id"], str)
            or not spec["id"]
            or spec["id"] in ids
            or lo.shape != (3,)
            or hi.shape != (3,)
            or not np.all(np.isfinite([lo, hi]))
            or np.any(lo >= hi)
            or not np.isfinite(spec["margin_m"])
            or spec["margin_m"] < 0
        ):
            raise ValueError(
                "Unique IDs, finite ordered 3D bounds and nonnegative margin required"
            )
        ids.add(spec["id"])


class RobotKeepClearEvaluator:
    def __init__(self, scene, specs=None, *, geometry=None, passage_payload_X_GC=None):
        self.scene = scene
        self.specs = (
            scene.config.get("robot_keep_clear", []) if specs is None else specs
        )
        validate_keep_clear(self.specs)
        from .human_passage import HumanPassageEvaluator

        self.passage = HumanPassageEvaluator(scene, payload_X_GC=passage_payload_X_GC)
        self.geometry = (
            (
                geometry
                if geometry is not None
                else WholeRobotRoomOracle.from_xy_bounds(
                    scene, [-1, 1], [-1, 1]
                ).geometry
            )
            if self.specs
            else []
        )

    def current(self):
        rows, base, arm = [], [], []
        for spec in self.specs:
            for body, shape, offset, name in self.geometry:
                pose = (
                    self.scene.plant.EvalBodyPoseInWorld(self.scene.plant_context, body)
                    @ offset
                )
                residual = shape_box_residual(
                    shape,
                    pose,
                    np.asarray(spec["lower_xyz"]),
                    np.asarray(spec["upper_xyz"]),
                    spec["margin_m"],
                )
                row = dict(
                    zone=spec["id"],
                    body=body.name(),
                    geometry=name,
                    residual_m=residual,
                )
                rows.append(row)
                (
                    base if body.model_instance() == self.scene.base_instance else arm
                ).append(residual)
        worst = min(rows, key=lambda r: r["residual_m"], default=None)
        result = dict(
            satisfied=worst is None or worst["residual_m"] >= -1e-8,
            minimum_residual_m=worst["residual_m"] if worst else None,
            worst=worst,
            base_residual_m=min(base, default=None),
            arm_gripper_residual_m=min(arm, default=None),
            scope="All base/arm/gripper collision volumes; all task phases; payload excluded; conservative axis separation",
        )
        human = self.passage.current()
        if human["enabled"]:

            def minimum(a, b):
                return b if a is None else min(a, b)

            result["human_passage"] = human
            result["satisfied"] = result["satisfied"] and human["satisfied"]
            result["base_residual_m"] = minimum(
                result["base_residual_m"],
                min(human["base_slack_m"], human["objects_slack_m"]),
            )
            result["arm_gripper_residual_m"] = minimum(
                result["arm_gripper_residual_m"],
                min(human["arm_slack_m"], human["payload_slack_m"]),
            )
            if (
                result["minimum_residual_m"] is None
                or human["motion_slack_m"] < result["minimum_residual_m"]
            ):
                result["minimum_residual_m"] = human["motion_slack_m"]
                result["worst"] = dict(
                    body="human_passage",
                    zone=self.passage.spec["id"],
                    geometry="capsule route witness",
                )
            result["scope"] += "; instantaneous human capsule route connectivity"
        return result


class OperationSpatialOracle:
    """Combine room containment and reserved-space exclusion for arm domains.

    Room faces stay unchanged for the separate payload room checks. Domain bounds
    use the minimum of both predicates, with the stationary base treated exactly.
    """

    def __init__(self, room, X_GC=None):
        self.room = room
        self.clear = RobotKeepClearEvaluator(
            room.scene, geometry=room.geometry, passage_payload_X_GC=X_GC
        )
        self.payload_reach_extra = 0.0
        if self.clear.passage.enabled:
            # AABB coordinate extrema change by at most point displacement in each axis.
            # sqrt(3) therefore also bounds motion of conservative AABB corners.
            self.payload_reach_extra = (
                (
                    np.linalg.norm(X_GC.translation())
                    + float(room.scene.config["objects"]["cup"]["height"])
                )
                if X_GC is not None
                else 0.0
            )
        self.aabb_motion_factor = np.sqrt(3) if self.clear.passage.enabled else 1.0

    def __getattr__(self, name):
        return getattr(self.room, name)

    def combine(self, room):
        clear = self.clear.current()
        result = dict(room, room_residual_m=room["residual_m"], keep_clear=clear)
        if clear["minimum_residual_m"] is not None:
            for key in ("base_residual_m", "arm_gripper_residual_m"):
                result[key] = min(room[key], clear[key])
            if clear["minimum_residual_m"] < room["residual_m"]:
                result.update(
                    residual_m=clear["minimum_residual_m"], worst=clear["worst"]
                )
            result["satisfied"] = room["satisfied"] and clear["satisfied"]
            result["scope"] = (
                "Room containment AND whole-robot reserved-space exclusion"
            )
        return result

    def query_current(self):
        return self.combine(self.room.query_current())

    def query(self, state, gripper=None):
        return self.combine(self.room.query(state, gripper))
