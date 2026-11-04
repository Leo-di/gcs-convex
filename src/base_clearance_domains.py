"""Deterministic base/environment separation inside a 10D input domain.

Static collision geometry is overapproximated by per-geometry world AABBs.
Base orientation support is bounded over the domain's full yaw interval.
The resulting halfspaces guarantee at least the requested geometric gap
inside this conservative representation, independently of IRIS sampling.
"""

import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from .human_passage import box_corners
from .whole_robot_room import shape_support


def separating_plane(obstacle_lo, obstacle_hi, half_size, yaw_lo, yaw_hi, a, b, margin):
    lo, hi, half, a, b = map(
        lambda x: np.asarray(x, float), (obstacle_lo, obstacle_hi, half_size, a, b)
    )
    yaw = (yaw_lo + yaw_hi) / 2
    radius = float(np.linalg.norm(half))
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    support_xy = np.abs(rotation) @ half + radius * (yaw_hi - yaw_lo) / 2
    middle = (a + b) / 2
    closest = np.clip(middle, lo - support_xy, hi + support_xy)
    v = middle - closest
    normals = [
        np.array([1.0, 0]),
        np.array([-1.0, 0]),
        np.array([0, 1.0]),
        np.array([0, -1.0]),
    ]
    if np.linalg.norm(v) > 1e-12:
        normals.append(v / np.linalg.norm(v))
    candidates = []
    for n in normals:
        support = float(np.abs(rotation.T @ n) @ half + radius * (yaw_hi - yaw_lo) / 2)
        rhs = float(np.where(n >= 0, hi, lo) @ n + support + margin)
        slack = min(float(n @ a - rhs), float(n @ b - rhs))
        candidates.append((slack, n, rhs))
    slack, n, rhs = max(candidates, key=lambda x: x[0])
    if slack < -1e-9:
        return None
    return n, rhs, slack


def environment_boxes(mode):
    scene = mode.scene
    plant = scene.plant
    ins = scene.scene_graph.model_inspector()
    base_ids = plant.GetCollisionGeometriesForBody(scene.base_body)
    boxes = []
    for i in range(plant.num_bodies()):
        from pydrake.multibody.tree import BodyIndex

        body = plant.get_body(BodyIndex(i))
        if mode.checker.IsPartOfRobot(body.index()):
            continue
        X_WB = plant.EvalBodyPoseInWorld(scene.plant_context, body)
        for gid in plant.GetCollisionGeometriesForBody(body):
            if all(ins.CollisionFiltered(base, gid) for base in base_ids):
                continue
            shape = ins.GetShape(gid)
            if hasattr(shape, "GetConvexHull"):
                mesh = shape.GetConvexHull()
                points = np.array([mesh.vertex(k) for k in range(mesh.num_vertices())])
            else:
                radii = np.array([shape_support(shape, e) for e in np.eye(3)])
                points = box_corners(-radii, radii)
            X = X_WB @ ins.GetPoseInFrame(gid)
            points = points @ X.rotation().matrix().T + X.translation()
            boxes.append(
                dict(name=ins.GetName(gid), lower=points.min(0), upper=points.max(0))
            )
    return boxes


def guard(mode, lower, upper, a, b, margin):
    if margin <= 0:
        return np.empty((0, 10)), np.empty(0), dict(enabled=False)
    if not hasattr(mode, "base_environment_boxes"):
        mode.base_environment_boxes = environment_boxes(mode)
    lo, hi = np.asarray(lower), np.asarray(upper)
    half = np.asarray(mode.scene.config["robot"]["size"]) / 2
    z = float(
        mode.scene.plant.EvalBodyPoseInWorld(
            mode.scene.plant_context, mode.scene.base_body
        ).translation()[2]
    )
    A = []
    B = []
    records = []
    for box in mode.base_environment_boxes:
        if (
            z - half[2] >= box["upper"][2] + margin
            or z + half[2] <= box["lower"][2] - margin
        ):
            continue
        plane = separating_plane(
            box["lower"][:2],
            box["upper"][:2],
            half[:2],
            lo[2],
            hi[2],
            np.array(a)[:2],
            np.array(b)[:2],
            margin,
        )
        if plane is None:
            return None
        n, rhs, slack = plane
        row = np.zeros(10)
        row[:2] = -n
        A.append(row)
        B.append(-rhs)
        records.append(
            dict(geometry=box["name"], normal=n.tolist(), rhs=rhs, guide_slack_m=slack)
        )
    return (
        np.asarray(A).reshape(-1, 10),
        np.asarray(B),
        dict(
            enabled=True,
            margin_m=margin,
            planes=records,
            scope="Per-geometry AABB separation plus full-yaw-interval base support bound; conservative static model",
        ),
    )


def intersect_cached(mode, record, margin):
    lo, hi = np.array(record["lower"]), np.array(record["upper"])
    a, b = np.array(record["guide_start"]), np.array(record["guide_end"])
    # Tighten only yaw uncertainty if necessary; endpoints remain available.
    for _ in range(15):
        answer = guard(mode, lo, hi, a, b, margin)
        if answer is not None:
            A, B, proof = answer
            box = HPolyhedron.MakeBox(lo, hi)
            region = HPolyhedron(
                np.vstack([record["A"], box.A(), A]), np.r_[record["b"], box.b(), B]
            )
            if not region.PointInSet(a, 1e-7) or not region.PointInSet(b, 1e-7):
                raise RuntimeError("Base guard removed a guide endpoint")
            updated = dict(
                record,
                A=region.A().tolist(),
                b=region.b().tolist(),
                lower=lo.tolist(),
                upper=hi.tolist(),
                base_clearance_guard=proof,
            )
            return region, updated
        lo[2] = (lo[2] + min(a[2], b[2])) / 2
        hi[2] = (hi[2] + max(a[2], b[2])) / 2
    raise RuntimeError(
        "No conservative base-clearance connection for this guide interval"
    )
