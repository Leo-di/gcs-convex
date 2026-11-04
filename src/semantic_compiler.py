"""Compile Inside into FK inequalities, derivatives, and box upper bounds."""

import numpy as np
from pydrake.autodiffutils import AutoDiffXd, InitializeAutoDiff, ExtractGradient
from pydrake.geometry import Box, Sphere, Cylinder, Capsule
from pydrake.multibody.tree import BodyIndex
from pydrake.solvers import MathematicalProgram

from .semantic_dsl import parse
from .iris_region_generator import _make_parameterization, _reduced_position_indices
from .build_scene import set_robot_configuration


def support(shape, d):
    if isinstance(shape, Box):
        return sum(
            abs(d[i]) * v / 2
            for i, v in enumerate([shape.width(), shape.depth(), shape.height()])
        )
    if isinstance(shape, Sphere):
        return shape.radius()
    if isinstance(shape, Capsule):
        return shape.radius() + shape.length() / 2 * abs(d[2])
    if isinstance(shape, Cylinder):
        squared = d[0] * d[0] + d[1] * d[1]
        value = squared.value() if isinstance(squared, AutoDiffXd) else float(squared)
        radial = np.sqrt(squared) if value > 1e-24 else squared * 0
        return shape.radius() * radial + shape.length() / 2 * abs(d[2])
    raise TypeError(
        "Inside supports Box, Sphere, Cylinder and Capsule collision shapes"
    )


def radius(shape):
    if isinstance(shape, Box):
        return float(np.linalg.norm([shape.width(), shape.depth(), shape.height()]) / 2)
    if isinstance(shape, Sphere):
        return shape.radius()
    if isinstance(shape, Cylinder):
        return float(np.hypot(shape.radius(), shape.length() / 2))
    if isinstance(shape, Capsule):
        return shape.radius() + shape.length() / 2
    raise TypeError("Unsupported collision shape")


class CompiledInside:
    def __init__(self, scene, document, base, gripper):
        self.scene = scene
        self.regions, self.predicates = parse(document)
        self.dimension = 7
        base, gripper = np.asarray(base, float), np.asarray(gripper, float)
        if (
            base.shape != (3,)
            or gripper.shape != (2,)
            or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(gripper))
        ):
            raise ValueError("Finite fixed base and gripper positions required")
        self.base = base.copy()
        self.gripper = gripper.copy()
        set_robot_configuration(
            scene, base, scene.config["pick_task"]["navigation_arm_q"], gripper
        )
        full = scene.plant.GetPositions(scene.plant_context).copy()
        self.indices = _reduced_position_indices(scene)[3:]
        self.mapping = _make_parameterization(full, self.indices)
        self.ad = scene.plant.ToAutoDiffXd()
        self.ad_context = self.ad.CreateDefaultContext()
        plant = scene.plant
        ins = scene.scene_graph.model_inspector()
        joints = [plant.get_joint(i) for i in plant.GetJointIndices()]
        self.parent = {int(j.child_body().index()): j for j in joints}
        self.arm_columns = {
            int(j.child_body().index()): int(
                np.where(self.indices == j.position_start())[0][0]
            )
            for j in joints
            if j.num_positions() == 1 and j.position_start() in self.indices
        }
        sets = dict(
            base=[scene.base_instance],
            arm=[scene.arm_instance],
            gripper=[scene.gripper_instance],
            robot=[scene.base_instance, scene.arm_instance, scene.gripper_instance],
        )
        self.items = []
        self.labels = []
        lipschitz = []
        for predicate in self.predicates:
            region = self.regions[predicate.region]
            for instance in sets[predicate.entity]:
                for index in plant.GetBodyIndices(instance):
                    body = plant.get_body(index)
                    for gid in plant.GetCollisionGeometriesForBody(body):
                        shape = ins.GetShape(gid)
                        support(shape, np.array([1.0, 0, 0]))
                        offset = ins.GetPoseInFrame(gid)
                        coefficients = self._motion_coefficients(
                            body, offset, shape, full
                        )
                        self.items.append(
                            (int(index), shape, offset, predicate, region)
                        )
                        for face in range(len(region.offsets)):
                            self.labels.append(
                                dict(
                                    predicate=predicate.id,
                                    body=body.name(),
                                    geometry=ins.GetName(gid),
                                    face=face,
                                )
                            )
                            lipschitz.append(coefficients)
        if not self.items:
            raise ValueError("Predicate entities have no supported collision geometry")
        self.lipschitz = np.asarray(lipschitz)

    def _motion_coefficients(self, body, offset, shape, full):
        result = np.zeros(7)
        reach = float(np.linalg.norm(offset.translation()) + radius(shape))
        node = int(body.index())
        while node in self.parent:
            joint = self.parent[node]
            if type(joint).__name__ == "PlanarJoint":
                break
            reach += float(
                np.linalg.norm(
                    joint.frame_on_child().GetFixedPoseInBodyFrame().translation()
                )
            )
            if node in self.arm_columns:
                result[self.arm_columns[node]] = reach
            if type(joint).__name__ == "WeldJoint":
                reach += float(np.linalg.norm(joint.X_FM().translation()))
            elif type(joint).__name__ == "PrismaticJoint":
                reach += abs(float(full[joint.position_start()]))
            elif type(joint).__name__ != "RevoluteJoint":
                raise TypeError("Unsupported joint in Inside entity chain")
            reach += float(
                np.linalg.norm(
                    joint.frame_on_parent().GetFixedPoseInBodyFrame().translation()
                )
            )
            node = int(joint.parent_body().index())
        return result

    def g(self, q):
        q = np.asarray(q).ravel()
        if q.shape != (7,):
            raise ValueError("Seven arm coordinates required")
        ad = isinstance(q[0], AutoDiffXd)
        if not ad and not np.all(np.isfinite(q)):
            raise ValueError("Finite coordinates required")
        plant, context = (
            (self.ad, self.ad_context)
            if ad
            else (self.scene.plant, self.scene.plant_context)
        )
        plant.SetPositions(context, self.mapping(q))
        poses = {}
        values = []
        for index, shape, offset, predicate, region in self.items:
            if index not in poses:
                poses[index] = plant.EvalBodyPoseInWorld(
                    context, plant.get_body(BodyIndex(index))
                )
            X = poses[index]
            R = X.rotation().matrix() @ offset.rotation().matrix()
            t = X.rotation().matrix() @ offset.translation() + X.translation()
            for n, b in zip(region.normals, region.offsets):
                values.append(n @ t + support(shape, R.T @ n) + predicate.margin_m - b)
        return np.array(values)

    def jacobian(self, q):
        return ExtractGradient(self.g(InitializeAutoDiff(np.asarray(q, float)).ravel()))

    def box_upper_bound(self, lower, upper):
        lo, hi = np.asarray(lower, float), np.asarray(upper, float)
        if (
            lo.shape != (7,)
            or hi.shape != (7,)
            or not np.all(np.isfinite([lo, hi]))
            or np.any(lo > hi)
        ):
            raise ValueError("Finite ordered seven-dimensional bounds required")
        return self.g((lo + hi) / 2) + self.lipschitz @ ((hi - lo) / 2)

    def to_program(self):
        program = MathematicalProgram()
        q = program.NewContinuousVariables(7, "q")
        program.AddConstraint(
            self.g, np.full(len(self.labels), -np.inf), np.zeros(len(self.labels)), q
        )
        return program

    def metadata(self):
        return dict(
            dimension=7,
            sign_convention="g(q) <= 0",
            fixed_base_q=self.base.tolist(),
            fixed_gripper_q=self.gripper.tolist(),
            constraints=self.labels,
            regions={
                name: dict(normals=r.normals.tolist(), offsets_m=r.offsets.tolist())
                for name, r in self.regions.items()
            },
            predicates=[
                dict(id=p.id, entity=p.entity, region=p.region, margin_m=p.margin_m)
                for p in self.predicates
            ],
            formula="g = n @ t_WG(q) + support(shape, R_WG(q).T @ n) + margin - b",
            native_interface="sampled_iris_options.prog_with_additional_constraints",
            per_joint_displacement_bounds=self.lipschitz.tolist(),
            geometry="Exact support functions for registered primitive collision shapes; payload excluded",
        )


def compile_predicates(document, scene, *, base, gripper):
    return CompiledInside(scene, document, base, gripper)
