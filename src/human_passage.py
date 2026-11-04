"""Human access as connectivity, with conservative geometry and route witnesses.

The human model is an upright capsule. A cell is admitted only when its entire
square is clear, using the half-diagonal as an extra distance guard. Four-way
grid paths therefore give continuous spatial witnesses in this model. This is
instantaneous access, not a moving pedestrian or a continuous-time certificate.
"""

import heapq
import itertools
import numpy as np
from pydrake.multibody.tree import BodyIndex
from .whole_robot_room import shape_support


def validate_passage(spec):
    if spec is None:
        return
    keys = {
        "id",
        "lower_xy",
        "upper_xy",
        "start_xy",
        "goal_xy",
        "radius_m",
        "height_m",
        "margin_m",
        "resolution_m",
        "enforce",
    }
    if not isinstance(spec, dict) or set(spec) not in (keys, keys | {"connections"}):
        raise ValueError("Explicit human passage model and endpoints required")
    a, b = np.asarray(spec["lower_xy"], float), np.asarray(spec["upper_xy"], float)
    if (
        a.shape != (2,)
        or b.shape != (2,)
        or not np.all(np.isfinite([a, b]))
        or np.any(a >= b)
        or not isinstance(spec["id"], str)
        or not spec["id"]
        or type(spec["enforce"]) is not bool
    ):
        raise ValueError(
            "Finite ordered passage bounds, ID and Boolean enforcement required"
        )
    for k in ["radius_m", "height_m", "margin_m", "resolution_m"]:
        if (
            not np.isfinite(spec[k])
            or spec[k] < 0
            or (k != "margin_m" and spec[k] == 0)
        ):
            raise ValueError("Invalid human passage dimensions")
    if spec["height_m"] <= 2 * spec["radius_m"]:
        raise ValueError("Capsule height must exceed its diameter")
    if np.prod(np.ceil((b - a) / spec["resolution_m"])) > 100000:
        raise ValueError("Passage grid exceeds bounded model size")
    for k in ["start_xy", "goal_xy"]:
        p = np.asarray(spec[k], float)
        if (
            p.shape != (2,)
            or not np.all(np.isfinite(p))
            or np.any(p < a)
            or np.any(p > b)
        ):
            raise ValueError("Passage endpoints must be in bounds")
    if "connections" in spec:
        if not isinstance(spec["connections"], list):
            raise ValueError("Connections must be a list")
        ids = {spec["id"]}
        for connection in spec["connections"]:
            if not isinstance(connection, dict) or set(connection) != {
                "id",
                "start_xy",
                "goal_xy",
            }:
                raise ValueError("Connection requires ID and endpoints")
            if not isinstance(connection["id"], str) or connection["id"] in ids:
                raise ValueError("Unique connection IDs required")
            ids.add(connection["id"])
            child = {k: v for k, v in spec.items() if k != "connections"}
            child.update(connection)
            validate_passage(child)


def box_corners(lo, hi):
    return np.array(list(itertools.product(*zip(lo, hi))))


def body_bounds(scene, body):
    inspector = scene.scene_graph.model_inspector()
    points = []
    for gid in scene.plant.GetCollisionGeometriesForBody(body):
        shape = inspector.GetShape(gid)
        if hasattr(shape, "GetConvexHull"):
            mesh = shape.GetConvexHull()
            v = np.array([mesh.vertex(i) for i in range(mesh.num_vertices())])
        else:
            r = np.array([shape_support(shape, e) for e in np.eye(3)])
            v = box_corners(-r, r)
        X = inspector.GetPoseInFrame(gid)
        points.append(v @ X.rotation().matrix().T + X.translation())
    if not points:
        return None
    points = np.vstack(points)
    return box_corners(points.min(0), points.max(0))


def capsule_box_distances(points, lower, upper, zlo, zhi):
    """Distance from vertical capsule axes to conservative body AABBs."""
    if not len(lower):
        return np.full((len(points), 0), np.inf)
    xy = np.maximum(
        np.maximum(
            lower[None, :, :2] - points[:, None, :],
            points[:, None, :] - upper[None, :, :2],
        ),
        0.0,
    )
    dz = np.maximum(np.maximum(lower[:, 2] - zhi, zlo - upper[:, 2]), 0.0)
    return np.sqrt(np.sum(xy * xy, axis=2) + dz[None, :] ** 2)


def widest_path(values, shape, start, goal):
    """Maximize dynamic clearance subject to the static certified cell mask."""
    width = shape[1]
    cost = np.full(len(values), -np.inf)
    parent = np.full(len(values), -1, int)
    cost[start] = values[start]
    queue = [(-cost[start], start)]
    while queue:
        neg, i = heapq.heappop(queue)
        if -neg < cost[i]:
            continue
        if i == goal:
            break
        y, x = divmod(i, width)
        for j in (
            ([i - 1] if x else [])
            + ([i + 1] if x + 1 < width else [])
            + ([i - width] if y else [])
            + ([i + width] if y + 1 < shape[0] else [])
        ):
            c = min(cost[i], values[j])
            if c > cost[j]:
                cost[j] = c
                parent[j] = i
                heapq.heappush(queue, (-c, j))
    if not np.isfinite(cost[goal]) or cost[goal] < 0:
        return None, float(cost[goal])
    route = [goal]
    while route[-1] != start:
        route.append(int(parent[route[-1]]))
    return np.array(route[::-1]), float(cost[goal])


class HumanPassageEvaluator:
    def __init__(self, scene, *, payload_X_GC=None, force=False, _spec=None):
        self.scene = scene
        self.spec = scene.config.get("human_passage") if _spec is None else _spec
        validate_passage(self.spec)
        self.connections = []
        self.enabled = bool(self.spec and (self.spec["enforce"] or force))
        self.payload_X_GC = payload_X_GC
        self.routes = []
        self.last_key = None
        self.last_result = None
        self.calls = 0
        self.grid_searches = 0
        if not self.enabled:
            return
        spec = self.spec
        self.resolution = spec["resolution_m"]
        self.guard = self.resolution / np.sqrt(2)
        self.required = spec["radius_m"] + spec["margin_m"] + self.guard
        self.zlo = spec["radius_m"]
        self.zhi = spec["height_m"] - spec["radius_m"]
        lo, hi = np.array(spec["lower_xy"]), np.array(spec["upper_xy"])
        self.x = np.arange(lo[0] + self.resolution / 2, hi[0], self.resolution)
        self.y = np.arange(lo[1] + self.resolution / 2, hi[1], self.resolution)
        xx, yy = np.meshgrid(self.x, self.y)
        self.points = np.column_stack([xx.ravel(), yy.ravel()])
        self.shape = xx.shape

        def index(p):
            return int(
                np.argmin(abs(self.y - p[1])) * len(self.x)
                + np.argmin(abs(self.x - p[0]))
            )

        self.start = index(spec["start_xy"])
        self.goal = index(spec["goal_xy"])
        robots = {scene.base_instance, scene.arm_instance, scene.gripper_instance}
        objects = {body.index(): key for key, body in scene.object_bodies.items()}
        active = scene.config.get("active_object_id", "cup")
        self.dynamic = []
        self.static = []
        floor_name = scene.config["world"]["floor"]["name"]
        for i in range(scene.plant.num_bodies()):
            body = scene.plant.get_body(BodyIndex(i))
            if body.name() == floor_name:
                continue
            corners = body_bounds(scene, body)
            if corners is None:
                continue
            if body.model_instance() in robots:
                category = (
                    "base" if body.model_instance() == scene.base_instance else "arm"
                )
            elif body.index() in objects:
                category = (
                    "payload"
                    if objects[body.index()] == active and payload_X_GC is not None
                    else "objects"
                )
            else:
                category = "static"
            item = (body, corners, category)
            (self.static if category == "static" else self.dynamic).append(item)
        low, high = self.bounds(self.static)
        distances = capsule_box_distances(self.points, low, high, self.zlo, self.zhi)
        edge = np.minimum(self.points - lo, hi - self.points).min(axis=1)
        self.static_slack = (
            np.minimum(distances.min(axis=1) if len(low) else np.inf, edge)
            - self.required
        )
        self.static_free = self.static_slack >= 0
        self.categories = [r[2] for r in self.dynamic]
        for connection in spec.get("connections", []):
            child = {k: v for k, v in spec.items() if k != "connections"}
            child.update(connection)
            self.connections.append(
                HumanPassageEvaluator(
                    scene, payload_X_GC=payload_X_GC, force=force, _spec=child
                )
            )

    def bounds(self, items):
        lower = []
        upper = []
        for body, corners, category in items:
            X = self.scene.plant.EvalBodyPoseInWorld(self.scene.plant_context, body)
            if category == "payload":
                X = (
                    self.scene.plant.EvalBodyPoseInWorld(
                        self.scene.plant_context, self.scene.gripper_body
                    )
                    @ self.payload_X_GC
                )
            v = corners @ X.rotation().matrix().T + X.translation()
            lower.append(v.min(0))
            upper.append(v.max(0))
        return np.asarray(lower).reshape(-1, 3), np.asarray(upper).reshape(-1, 3)

    def reset_witness_cache(self):
        """Forget route choices so certification is independent of visit order.

        Geometric/grid data and work counters remain unchanged. Normal online
        checks keep caching; matched region-certificate comparisons opt in.
        """
        self.routes = []
        self.last_key = None
        self.last_result = None
        for child in self.connections:
            child.reset_witness_cache()

    def current(self):
        primary = self._current_single()
        if not self.connections:
            return primary
        results = {
            self.spec["id"]: primary,
            **{h.spec["id"]: h.current() for h in self.connections},
        }
        worst = min(results, key=lambda key: results[key]["motion_slack_m"])
        combined = dict(
            primary,
            satisfied=all(r["satisfied"] for r in results.values()),
            connections=results,
            worst_connection_id=worst,
        )
        for key in [
            "motion_slack_m",
            "base_slack_m",
            "arm_slack_m",
            "objects_slack_m",
            "payload_slack_m",
        ]:
            combined[key] = min(r[key] for r in results.values())
        return combined

    def _current_single(self):
        if not self.enabled:
            return dict(satisfied=True, enabled=False)
        self.calls += 1
        lo, hi = self.bounds(self.dynamic)
        key = lo.tobytes() + hi.tobytes()
        if key == self.last_key:
            return self.last_result
        best = None
        best_value = -np.inf
        best_dist = None
        # A cached spatial witness remains valid under bounded point motion.
        # Only recompute connectivity when no cached witness remains clear.
        for route in self.routes:
            d = (
                capsule_box_distances(self.points[route], lo, hi, self.zlo, self.zhi)
                - self.required
            )
            value = float(d.min())
            if value > best_value:
                best, best_value, best_dist = route, value, d
        # Refresh low-clearance cached routes to obtain a useful domain bound.
        # The returned route is checked independently of this refresh threshold.
        if best_value < 0.12:
            self.grid_searches += 1
            d = (
                capsule_box_distances(self.points, lo, hi, self.zlo, self.zhi)
                - self.required
            )
            values = np.where(self.static_free, d.min(axis=1), -np.inf)
            best, best_value = widest_path(values, self.shape, self.start, self.goal)
            if best is not None:
                self.routes.append(best)
                if len(self.routes) > 8:
                    self.routes.pop(0)
                best_dist = d[best]
        if best is None or best_value < 0:
            result = dict(
                satisfied=False,
                enabled=True,
                motion_slack_m=-1e-6,
                base_slack_m=-1e-6,
                arm_slack_m=-1e-6,
                objects_slack_m=-1e-6,
                payload_slack_m=-1e-6,
                path_xy=[],
                reason="No certified grid route between required endpoints",
                grid_searches=self.grid_searches,
            )
        else:
            values = {}
            for category in ["base", "arm", "objects", "payload"]:
                ids = [i for i, c in enumerate(self.categories) if c == category]
                values[category + "_slack_m"] = (
                    float(best_dist[:, ids].min()) if ids else float("inf")
                )
            result = dict(
                satisfied=True,
                enabled=True,
                motion_slack_m=best_value,
                static_witness_slack_m=float(self.static_slack[best].min()),
                path_xy=self.points[best].tolist(),
                **values,
                grid_searches=self.grid_searches,
                scope="Instantaneous capsule access; body AABBs; full-cell clearance guard; 4-neighbor path; fixed furniture",
            )
        self.last_key = key
        self.last_result = result
        return result
