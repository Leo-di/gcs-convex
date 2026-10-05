"""Build and display the first two-room Drake scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pydrake.geometry import (
    Box,
    CollisionFilterDeclaration,
    Cylinder,
    GeometrySet,
    StartMeshcat,
    AddCompliantHydroelasticProperties, AddRigidHydroelasticProperties,
    ProximityProperties, RoleAssign,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    CoulombFriction, ContactModel, DiscreteContactApproximation,
)
from pydrake.multibody.tree import (
    FixedOffsetFrame, PdControllerGains,
    PlanarJoint,
    SpatialInertia,
    UnitInertia,
    default_model_instance,
)
from pydrake.planning import RobotDiagramBuilder
from pydrake.visualization import AddDefaultVisualization

from .scene_spec import load_scene_config


@dataclass
class SceneHandles:
    diagram: Any
    plant: Any
    scene_graph: Any
    meshcat: Any
    root_context: Any
    plant_context: Any
    base_body: Any
    base_joint: Any
    base_instance: Any
    arm_instance: Any
    arm_end_effector_frame: Any
    arm_home_positions: np.ndarray
    gripper_instance: Any
    gripper_body: Any
    gripper_open_positions: np.ndarray
    gripper_closed_positions: np.ndarray
    cup_body: Any
    cup_initial_pose: RigidTransform
    attached_cup_geometry_id: Any | None
    attached_cup_instance: Any | None
    config: dict[str, Any]


def _spatial_inertia_for_box(size: np.ndarray, mass: float) -> SpatialInertia:
    return SpatialInertia(
        mass=mass,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia.SolidBox(*size),
    )


def _add_fixed_box(
    plant: Any,
    *,
    name: str,
    size: list[float],
    center: list[float],
    color: list[float],
    collision: bool = True,
) -> Any:
    size_array = np.asarray(size, dtype=float)
    body = plant.AddRigidBody(
        name,
        default_model_instance(),
        _spatial_inertia_for_box(size_array, mass=1.0),
    )
    shape = Box(*size_array)

    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        shape,
        f"{name}_visual",
        np.asarray(color, dtype=float),
    )

    if collision:
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(),
            shape,
            f"{name}_collision",
            CoulombFriction(0.9, 0.8),
        )

    plant.WeldFrames(
        plant.world_frame(),
        body.body_frame(),
        RigidTransform(center),
    )
    return body


def _add_detailed_table(
    plant: Any,
    *,
    name: str,
    size: list[float],
    center: list[float],
    color: list[float],
    top_thickness: float,
    leg_size: float,
    edge_color: list[float],
) -> Any:
    size_array = np.asarray(size, dtype=float)
    body = plant.AddRigidBody(
        name,
        default_model_instance(),
        _spatial_inertia_for_box(size_array, mass=15.0),
    )
    width, depth, height = size_array
    X_BT = RigidTransform([0.0, 0.0, height / 2.0 - top_thickness / 2.0])
    tabletop = Box(width, depth, top_thickness)
    plant.RegisterVisualGeometry(
        body,
        X_BT,
        tabletop,
        f"{name}_top_visual",
        np.asarray(color, dtype=float),
    )
    plant.RegisterCollisionGeometry(
        body,
        X_BT,
        tabletop,
        f"{name}_top_collision",
        CoulombFriction(0.9, 0.8),
    )

    leg_height = height - top_thickness
    leg_offset_x = width / 2.0 - leg_size
    leg_offset_y = depth / 2.0 - leg_size
    leg_z = -top_thickness / 2.0
    for index, (x, y) in enumerate(
        [
            (-leg_offset_x, -leg_offset_y),
            (-leg_offset_x, leg_offset_y),
            (leg_offset_x, -leg_offset_y),
            (leg_offset_x, leg_offset_y),
        ]
    ):
        X_BL = RigidTransform([x, y, leg_z])
        leg = Box(leg_size, leg_size, leg_height)
        plant.RegisterVisualGeometry(
            body,
            X_BL,
            leg,
            f"{name}_leg_{index}_visual",
            np.asarray(edge_color, dtype=float),
        )
        plant.RegisterCollisionGeometry(
            body,
            X_BL,
            leg,
            f"{name}_leg_{index}_collision",
            CoulombFriction(0.9, 0.8),
        )

    apron_height = 0.12
    for index, (apron_size, apron_xyz) in enumerate(
        [
            ([width - 0.12, 0.05, apron_height], [0.0, depth / 2.0 - 0.05, height / 2.0 - 0.14]),
            ([width - 0.12, 0.05, apron_height], [0.0, -depth / 2.0 + 0.05, height / 2.0 - 0.14]),
            ([0.05, depth - 0.12, apron_height], [width / 2.0 - 0.05, 0.0, height / 2.0 - 0.14]),
            ([0.05, depth - 0.12, apron_height], [-width / 2.0 + 0.05, 0.0, height / 2.0 - 0.14]),
        ]
    ):
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(apron_xyz),
            Box(*apron_size),
            f"{name}_apron_{index}_visual",
            np.asarray(edge_color, dtype=float),
        )

    plant.WeldFrames(
        plant.world_frame(),
        body.body_frame(),
        RigidTransform(center),
    )
    return body


def _add_floor_grid(plant: Any, floor_spec: dict[str, Any], grid_spec: dict[str, Any]) -> None:
    """Draw a metric floor grid without adding collision geometry."""

    floor_size = np.asarray(floor_spec["size"], dtype=float)
    floor_center = np.asarray(floor_spec["center"], dtype=float)
    spacing = float(grid_spec["spacing"])
    line_width = float(grid_spec["line_width"])
    color = grid_spec["color"]
    z = floor_center[2] + floor_size[2] / 2.0 + 0.001

    x_min = floor_center[0] - floor_size[0] / 2.0
    x_max = floor_center[0] + floor_size[0] / 2.0
    y_min = floor_center[1] - floor_size[1] / 2.0
    y_max = floor_center[1] + floor_size[1] / 2.0
    for index, x in enumerate(np.arange(x_min, x_max + spacing / 2.0, spacing)):
        _add_fixed_box(
            plant,
            name=f"floor_grid_x_{index}",
            size=[line_width, floor_size[1], 0.002],
            center=[float(x), float(floor_center[1]), z],
            color=color,
            collision=False,
        )
    for index, y in enumerate(np.arange(y_min, y_max + spacing / 2.0, spacing)):
        _add_fixed_box(
            plant,
            name=f"floor_grid_y_{index}",
            size=[floor_size[0], line_width, 0.002],
            center=[float(floor_center[0]), float(y), z],
            color=color,
            collision=False,
        )


def _add_open_bin(
    plant: Any,
    *,
    name: str,
    size: list[float],
    center: list[float],
    wall_thickness: float,
    color: list[float],
) -> Any:
    """Add a tray-like open bin with five collision pieces."""

    size_array = np.asarray(size, dtype=float)
    width, depth, height = size_array
    body = plant.AddRigidBody(
        name,
        default_model_instance(),
        _spatial_inertia_for_box(size_array, mass=0.8),
    )
    rgba = np.asarray(color, dtype=float)
    pieces = [
        ("bottom", [width, depth, wall_thickness], [0.0, 0.0, -height / 2.0 + wall_thickness / 2.0]),
        ("left", [wall_thickness, depth, height], [-width / 2.0 + wall_thickness / 2.0, 0.0, 0.0]),
        ("right", [wall_thickness, depth, height], [width / 2.0 - wall_thickness / 2.0, 0.0, 0.0]),
        ("front", [width, wall_thickness, height], [0.0, -depth / 2.0 + wall_thickness / 2.0, 0.0]),
        ("back", [width, wall_thickness, height], [0.0, depth / 2.0 - wall_thickness / 2.0, 0.0]),
    ]
    for part_name, part_size, xyz in pieces:
        shape = Box(*part_size)
        X_BP = RigidTransform(xyz)
        plant.RegisterVisualGeometry(
            body,
            X_BP,
            shape,
            f"{name}_{part_name}_visual",
            rgba,
        )
        plant.RegisterCollisionGeometry(
            body,
            X_BP,
            shape,
            f"{name}_{part_name}_collision",
            CoulombFriction(0.8, 0.7),
        )
    plant.WeldFrames(plant.world_frame(), body.body_frame(), RigidTransform(center))
    return body


def _add_storage_cabinet(
    plant: Any,
    *,
    name: str,
    size: list[float],
    center: list[float],
    color: list[float],
    accent_color: list[float],
) -> Any:
    """Add a collision-safe lab cabinet with visible doors and handles."""

    size_array = np.asarray(size, dtype=float)
    width, depth, height = size_array
    body = plant.AddRigidBody(
        name,
        default_model_instance(),
        _spatial_inertia_for_box(size_array, mass=35.0),
    )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        Box(*size_array),
        f"{name}_case_visual",
        np.asarray(color, dtype=float),
    )
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        Box(*size_array),
        f"{name}_collision",
        CoulombFriction(0.9, 0.8),
    )
    front_x = width / 2.0 + 0.006
    door_width = depth / 2.0 - 0.025
    for index, y in enumerate((-depth / 4.0, depth / 4.0)):
        plant.RegisterVisualGeometry(
            body,
            RigidTransform([front_x, y, 0.0]),
            Box(0.012, door_width, height - 0.05),
            f"{name}_door_{index}_visual",
            np.asarray(color, dtype=float) * np.array([0.88, 0.88, 0.88, 1.0]),
        )
        plant.RegisterVisualGeometry(
            body,
            RigidTransform([front_x + 0.018, y + (-0.06 if index == 0 else 0.06), 0.12]),
            Box(0.035, 0.025, 0.18),
            f"{name}_handle_{index}_visual",
            np.asarray(accent_color, dtype=float),
        )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform([0.0, 0.0, height / 2.0 + 0.02]),
        Box(width + 0.04, depth + 0.04, 0.04),
        f"{name}_top_visual",
        np.asarray(accent_color, dtype=float),
    )
    plant.WeldFrames(plant.world_frame(), body.body_frame(), RigidTransform(center))
    return body


def _add_mobile_cart_fixture(
    plant: Any,
    *,
    name: str,
    size: list[float],
    center: list[float],
    color: list[float],
    shelf_color: list[float],
) -> Any:
    """Add a fixed rolling cart obstacle with shelves, posts, and casters."""

    size_array = np.asarray(size, dtype=float)
    width, depth, height = size_array
    body = plant.AddRigidBody(
        name,
        default_model_instance(),
        _spatial_inertia_for_box(size_array, mass=18.0),
    )
    frame_rgba = np.asarray(color, dtype=float)
    shelf_rgba = np.asarray(shelf_color, dtype=float)
    shelf_thickness = 0.055
    for index, z in enumerate((-height / 2.0 + 0.16, 0.0, height / 2.0 - 0.07)):
        shape = Box(width, depth, shelf_thickness)
        X_BS = RigidTransform([0.0, 0.0, z])
        plant.RegisterVisualGeometry(body, X_BS, shape, f"{name}_shelf_{index}_visual", shelf_rgba)
        plant.RegisterCollisionGeometry(
            body,
            X_BS,
            shape,
            f"{name}_shelf_{index}_collision",
            CoulombFriction(0.8, 0.7),
        )
    post_size = 0.035
    for index, (x, y) in enumerate(
        [
            (-width / 2.0 + 0.04, -depth / 2.0 + 0.04),
            (-width / 2.0 + 0.04, depth / 2.0 - 0.04),
            (width / 2.0 - 0.04, -depth / 2.0 + 0.04),
            (width / 2.0 - 0.04, depth / 2.0 - 0.04),
        ]
    ):
        shape = Box(post_size, post_size, height - 0.12)
        X_BP = RigidTransform([x, y, 0.0])
        plant.RegisterVisualGeometry(body, X_BP, shape, f"{name}_post_{index}_visual", frame_rgba)
        plant.RegisterCollisionGeometry(
            body,
            X_BP,
            shape,
            f"{name}_post_{index}_collision",
            CoulombFriction(0.8, 0.7),
        )
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(RollPitchYaw(np.pi / 2.0, 0.0, 0.0), [x, y, -height / 2.0 + 0.055]),
            Cylinder(0.055, 0.035),
            f"{name}_caster_{index}_visual",
            np.array([0.04, 0.04, 0.05, 1.0]),
        )
    plant.WeldFrames(plant.world_frame(), body.body_frame(), RigidTransform(center))
    return body


def _add_fixed_distractor(plant: Any, spec: dict[str, Any]) -> Any:
    """Add a simple fixed tabletop object used as semantic clutter."""

    shape_name = spec["shape"]
    if shape_name == "box":
        size = np.asarray(spec["size"], dtype=float)
        shape = Box(*size)
        inertia = _spatial_inertia_for_box(size, mass=0.3)
    elif shape_name == "cylinder":
        radius = float(spec["radius"])
        height = float(spec["height"])
        shape = Cylinder(radius, height)
        inertia = SpatialInertia(
            mass=0.3,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia.SolidCylinder(radius, height, np.array([0.0, 0.0, 1.0])),
        )
    else:
        raise ValueError(f"Unsupported distractor shape: {shape_name}")

    body = plant.AddRigidBody(spec["name"], default_model_instance(), inertia)
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        shape,
        f"{spec['name']}_visual",
        np.asarray(spec["color"], dtype=float),
    )
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        shape,
        f"{spec['name']}_collision",
        CoulombFriction(0.8, 0.7),
    )
    plant.WeldFrames(plant.world_frame(), body.body_frame(), RigidTransform(spec["center"]))
    return body


def _add_detailed_cup(
    plant: Any,
    *,
    name: str,
    radius: float,
    height: float,
    center: list[float],
    color: list[float],
    collision: bool = True,
    visual_handle: bool = True,
) -> Any:
    """Add a floating cylindrical mug with a visual handle."""

    model_instance = plant.AddModelInstance(name)
    inertia = SpatialInertia(
        mass=0.2,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia.SolidCylinder(
            radius,
            height,
            np.array([0.0, 0.0, 1.0]),
        ),
    )
    body = plant.AddRigidBody(name, model_instance, inertia)
    cup_shape = Cylinder(radius, height)
    rgba = np.asarray(color, dtype=float)
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        cup_shape,
        f"{name}_body_visual",
        rgba,
    )
    if collision:
        plant.RegisterCollisionGeometry(
            body, RigidTransform(), cup_shape, f"{name}_collision",
            CoulombFriction(0.9, 0.8),
        )

    handle_thickness = 0.014
    handle_reach = 0.055
    for index, (shape, xyz) in enumerate(
        [
            (Box(handle_reach, handle_thickness, handle_thickness), [radius + handle_reach / 2.0, 0.0, 0.045]),
            (Box(handle_reach, handle_thickness, handle_thickness), [radius + handle_reach / 2.0, 0.0, -0.045]),
            (Box(handle_thickness, handle_thickness, 0.10), [radius + handle_reach, 0.0, 0.0]),
        ] if visual_handle else []
    ):
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(xyz),
            shape,
            f"{name}_handle_{index}_visual",
            rgba,
        )

    plant.SetDefaultFloatingBaseBodyPose(body, RigidTransform(center))
    return body


def _add_mobile_base(plant: Any, robot_spec: dict[str, Any]) -> tuple[Any, Any, Any]:
    size = np.asarray(robot_spec["size"], dtype=float)
    model_instance = plant.AddModelInstance("mobile_base_model")
    body = plant.AddRigidBody(
        "mobile_base",
        model_instance,
        _spatial_inertia_for_box(size, mass=20.0),
    )
    collision_shape = Box(*size)

    plant.RegisterVisualGeometry(
        body,
        RigidTransform([0.0, 0.0, -0.055]),
        Box(size[0] * 0.96, size[1] * 0.92, 0.20),
        "mobile_base_lower_visual",
        np.asarray(robot_spec["color"], dtype=float),
    )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform([0.0, 0.0, 0.085]),
        Box(size[0] * 0.68, size[1] * 0.65, 0.12),
        "mobile_base_upper_visual",
        np.array([0.16, 0.34, 0.62, 1.0]),
    )

    wheel_pose_rotation = RollPitchYaw(np.pi / 2.0, 0.0, 0.0)
    for index, (x, y) in enumerate(
        [
            (-0.20, -size[1] / 2.0),
            (-0.20, size[1] / 2.0),
            (0.20, -size[1] / 2.0),
            (0.20, size[1] / 2.0),
        ]
    ):
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(wheel_pose_rotation, [x, y, -0.09]),
            Cylinder(0.09, 0.055),
            f"mobile_base_wheel_{index}_visual",
            np.array([0.035, 0.035, 0.045, 1.0]),
        )

    plant.RegisterVisualGeometry(
        body,
        RigidTransform(
            RollPitchYaw(0.0, np.pi / 2.0, 0.0),
            [size[0] / 2.0, 0.0, 0.04],
        ),
        Cylinder(0.055, 0.045),
        "mobile_base_front_sensor_visual",
        np.array([0.08, 0.08, 0.10, 1.0]),
    )
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        collision_shape,
        "mobile_base_collision",
        CoulombFriction(0.9, 0.8),
    )

    motion_plane = plant.AddFrame(
        FixedOffsetFrame(
            "mobile_base_motion_plane",
            plant.world_frame(),
            RigidTransform([0.0, 0.0, size[2] / 2.0]),
        )
    )
    joint = plant.AddJoint(
        PlanarJoint(
            "mobile_base_joint",
            motion_plane,
            body.body_frame(),
        )
    )
    joint.set_position_limits(
        np.asarray(robot_spec["lower_limits"], dtype=float),
        np.asarray(robot_spec["upper_limits"], dtype=float),
    )
    return body, joint, model_instance


def set_robot_configuration(
    scene: SceneHandles,
    base_q: np.ndarray | list[float],
    arm_q: np.ndarray | list[float] | None = None,
    gripper_q: np.ndarray | list[float] | None = None,
) -> None:
    """Set the planar base, seven arm joints, and two gripper joints."""

    base_q_array = np.asarray(base_q, dtype=float)
    if base_q_array.shape != (3,):
        raise ValueError("Expected base_q=(x, y, yaw).")

    if arm_q is None:
        arm_q_array = scene.arm_home_positions
    else:
        arm_q_array = np.asarray(arm_q, dtype=float)
    if arm_q_array.shape != (7,):
        raise ValueError("Expected seven iiwa joint positions.")

    if gripper_q is None:
        gripper_q_array = scene.gripper_open_positions
    else:
        gripper_q_array = np.asarray(gripper_q, dtype=float)
    if gripper_q_array.shape != (2,):
        raise ValueError("Expected two Schunk gripper positions.")

    scene.base_joint.set_translation(scene.plant_context, base_q_array[:2])
    scene.base_joint.set_rotation(scene.plant_context, float(base_q_array[2]))
    scene.plant.SetPositions(
        scene.plant_context,
        scene.arm_instance,
        arm_q_array,
    )
    scene.plant.SetPositions(
        scene.plant_context,
        scene.gripper_instance,
        gripper_q_array,
    )


def build_scene(
    *,
    attached_cup_X_GC: RigidTransform | None = None,
    visualize: bool = True,
    config: dict | None = None,
    dynamics_time_step: float | None = None,
) -> SceneHandles:
    config = load_scene_config() if config is None else config
    meshcat = StartMeshcat() if visualize else None
    if dynamics_time_step is not None and (dynamics_time_step <= 0 or attached_cup_X_GC is not None):
        raise ValueError("Dynamic execution requires a positive step and a free cup")
    robot_builder = RobotDiagramBuilder(time_step=dynamics_time_step or 0.0)
    builder = robot_builder.builder()
    plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()

    floor_spec = config["world"]["floor"]
    floor_body = _add_fixed_box(plant, **floor_spec)
    _add_floor_grid(plant, floor_spec, config["world"]["floor_grid"])

    wall_color = config["world"]["wall_color"]
    for wall_spec in config["world"]["walls"]:
        _add_fixed_box(plant, color=wall_color, **wall_spec)

    table_body = _add_detailed_table(
        plant,
        name="table",
        **config["objects"]["table"],
    )
    _add_detailed_table(
        plant,
        name="dropoff_table",
        **config["objects"]["dropoff_table"],
    )
    _add_open_bin(
        plant,
        name="dropoff_bin",
        **config["objects"]["dropoff_bin"],
    )
    _add_storage_cabinet(
        plant,
        name="storage_cabinet",
        **config["objects"]["storage_cabinet"],
    )
    _add_mobile_cart_fixture(
        plant,
        name="mobile_cart",
        **config["objects"]["mobile_cart"],
    )
    for distractor_spec in config["objects"].get("distractors", []):
        _add_fixed_distractor(plant, distractor_spec)

    cup_spec = config["objects"]["cup"]
    cup_body = _add_detailed_cup(
        plant,
        name="cup",
        collision=not (attached_cup_X_GC is not None
                       and config["gripper"].get("contact_filter") == "fingertips"),
        **cup_spec,
    )
    cup_initial_pose = RigidTransform(cup_spec["center"])

    for region_name, region_spec in config["semantic_regions"].items():
        x_min, x_max = region_spec["x_bounds"]
        y_min, y_max = region_spec["y_bounds"]
        display_height = float(region_spec["display_height"])
        _add_fixed_box(
            plant,
            name=f"semantic_{region_name}",
            size=[x_max - x_min, y_max - y_min, display_height],
            center=[
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
                display_height / 2.0 + 0.002,
            ],
            color=region_spec["color"],
            collision=False,
        )

    for index, region_spec in enumerate(config["gcs_base_planning"]["regions"]):
        lower = np.asarray(region_spec["lower"], dtype=float)
        upper = np.asarray(region_spec["upper"], dtype=float)
        height = 0.004
        _add_fixed_box(
            plant,
            name=f"gcs_region_{region_spec['name']}",
            size=[upper[0] - lower[0], upper[1] - lower[1], height],
            center=[
                (lower[0] + upper[0]) / 2.0,
                (lower[1] + upper[1]) / 2.0,
                0.021 + index * 0.0006,
            ],
            color=region_spec["color"],
            collision=False,
        )

    for zone_name, zone_spec in config.get("task_zones", {}).items():
        _add_fixed_box(
            plant,
            name=f"task_zone_{zone_name}",
            size=zone_spec["size"],
            center=zone_spec["center"],
            color=zone_spec["color"],
            collision=False,
        )

    base_body, base_joint, base_instance = _add_mobile_base(
        plant,
        config["robot"],
    )

    arm_spec = config["arm"]
    (arm_instance,) = Parser(
        builder,
        model_name_prefix=arm_spec["model_name_prefix"],
    ).AddModels(url=arm_spec["model_url"])
    arm_base_frame = plant.GetFrameByName(
        arm_spec["base_frame"],
        arm_instance,
    )
    plant.WeldFrames(
        base_body.body_frame(),
        arm_base_frame,
        RigidTransform(arm_spec["mount_translation"]),
    )
    arm_end_effector_frame = plant.GetFrameByName(
        arm_spec["end_effector_frame"],
        arm_instance,
    )

    gripper_spec = config["gripper"]
    (gripper_instance,) = Parser(
        builder,
        model_name_prefix=gripper_spec["model_name_prefix"],
    ).AddModels(url=gripper_spec["model_url"])
    gripper_body = plant.GetBodyByName("body", gripper_instance)
    gripper_base_frame = plant.GetFrameByName(
        gripper_spec["base_frame"],
        gripper_instance,
    )
    plant.WeldFrames(
        arm_end_effector_frame,
        gripper_base_frame,
        RigidTransform(
            RollPitchYaw(gripper_spec["mount_rpy"]),
            gripper_spec["mount_translation"],
        ),
    )

    attached_cup_geometry_id = None
    attached_cup_instance = None
    if attached_cup_X_GC is not None:
        cup_spec = config["objects"]["cup"]
        attached_cup_instance = plant.AddModelInstance("attached_cup")
        attached_radius = float(cup_spec["radius"])
        attached_height = float(cup_spec["height"])
        attached_body = plant.AddRigidBody(
            "attached_cup",
            attached_cup_instance,
            SpatialInertia(
                mass=0.2,
                p_PScm_E=np.zeros(3),
                G_SP_E=UnitInertia.SolidCylinder(
                    attached_radius,
                    attached_height,
                    np.array([0.0, 0.0, 1.0]),
                ),
            ),
        )
        attached_cup_geometry_id = plant.RegisterCollisionGeometry(
            attached_body,
            RigidTransform(),
            Cylinder(attached_radius, attached_height),
            "attached_cup_collision",
            CoulombFriction(0.9, 0.8),
        )
        plant.WeldFrames(
            gripper_body.body_frame(),
            attached_body.body_frame(),
            attached_cup_X_GC,
        )

    gripper_geometry_ids = [
        geometry_id
        for body_index in plant.GetBodyIndices(gripper_instance)
        for geometry_id in plant.GetCollisionGeometriesForBody(plant.get_body(body_index))
    ]
    narrow_contact_filter = gripper_spec.get("contact_filter") == "fingertips"
    allowed_cup_contact_ids = gripper_geometry_ids
    if narrow_contact_filter:
        inspector = scene_graph.model_inspector()
        allowed_cup_contact_ids = [
            gid for gid in gripper_geometry_ids
            if any(inspector.GetName(gid).endswith(suffix)
                   for suffix in ("_top", "_bottom", "_left", "_right"))
        ]
        if len(allowed_cup_contact_ids) != 8:
            raise ValueError("Expected eight named WSG fingertip contact spheres")
    # The base is supported by the floor, so this intentional contact should
    # not count as an obstacle collision or dominate the clearance residual.
    filters = CollisionFilterDeclaration().ExcludeBetween(
        GeometrySet(plant.GetCollisionGeometriesForBody(floor_body)),
        GeometrySet(plant.GetCollisionGeometriesForBody(base_body)),
    )
    if dynamics_time_step is None:
        filters.ExcludeBetween(
            GeometrySet(plant.GetCollisionGeometriesForBody(table_body)),
            GeometrySet(plant.GetCollisionGeometriesForBody(cup_body)),
        ).ExcludeBetween(
            GeometrySet(plant.GetCollisionGeometriesForBody(cup_body)),
            GeometrySet(allowed_cup_contact_ids),
        )
    scene_graph.collision_filter_manager().Apply(filters)
    if attached_cup_geometry_id is not None:
        robot_geometry_ids = [
            geometry_id
            for instance in (base_instance, arm_instance, gripper_instance)
            for body_index in plant.GetBodyIndices(instance)
            for geometry_id in plant.GetCollisionGeometriesForBody(
                plant.get_body(body_index)
            )
            if geometry_id != attached_cup_geometry_id
        ]
        free_cup_geometry_ids = plant.GetCollisionGeometriesForBody(cup_body)
        scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration()
            .ExcludeBetween(
                GeometrySet([attached_cup_geometry_id]),
                # Drake's planning checker requires uniform body-pair filters.
                # Finger boxes remain covered by the unfiltered path audit.
                # Crucially, payload versus arm/base is no longer suppressed.
                GeometrySet(gripper_geometry_ids if narrow_contact_filter else robot_geometry_ids),
            )
            .ExcludeBetween(
                GeometrySet(free_cup_geometry_ids),
                GeometrySet(robot_geometry_ids),
            )
        )
    if dynamics_time_step is not None:
        # SAP does not support multi-DOF joint limit constraints. These are
        # workspace planning bounds, not physical end stops on the mobile base.
        base_joint.set_position_limits(np.full(3, -np.inf), np.full(3, np.inf))
        plant.set_contact_model(ContactModel.kPoint)
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
        plant.set_penetration_allowance(0.0001)
        if config.get("execution_dynamics", {}).get("hydroelastic_support", False):
            # A finite contact patch avoids balancing a flat cup bottom on a
            # single point. Unannotated pairs retain point-contact fallback.
            plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
            inspector = scene_graph.model_inspector()
            for name in ("cup", "table", "dropoff_table", "dropoff_bin", "floor"):
                body = plant.GetBodyByName(name)
                for gid in plant.GetCollisionGeometriesForBody(body):
                    props = ProximityProperties(inspector.GetProximityProperties(gid))
                    if name == "cup":
                        AddCompliantHydroelasticProperties(0.01, 1e7, props)
                    else:
                        AddRigidHydroelasticProperties(0.01, props)
                    scene_graph.AssignRole(plant.get_source_id(), gid, props, RoleAssign.kReplace)
        for index in plant.GetJointActuatorIndices():
            actuator = plant.get_joint_actuator(index)
            if actuator.model_instance() == arm_instance:
                actuator.set_controller_gains(PdControllerGains(p=3000.0, d=100.0))
            elif actuator.model_instance() == gripper_instance:
                actuator.set_effort_limit(20.0)
                actuator.set_controller_gains(PdControllerGains(p=5000.0, d=20.0))
    plant.Finalize()

    if visualize:
        AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = robot_builder.Build()
    root_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(root_context)
    scene = SceneHandles(
        diagram=diagram,
        plant=plant,
        scene_graph=scene_graph,
        meshcat=meshcat,
        root_context=root_context,
        plant_context=plant_context,
        base_body=base_body,
        base_joint=base_joint,
        base_instance=base_instance,
        arm_instance=arm_instance,
        arm_end_effector_frame=arm_end_effector_frame,
        arm_home_positions=np.asarray(
            arm_spec["home_positions"],
            dtype=float,
        ),
        gripper_instance=gripper_instance,
        gripper_body=gripper_body,
        gripper_open_positions=np.asarray(
            gripper_spec["open_positions"],
            dtype=float,
        ),
        gripper_closed_positions=np.asarray(
            gripper_spec["closed_positions"],
            dtype=float,
        ),
        cup_body=cup_body,
        cup_initial_pose=cup_initial_pose,
        attached_cup_geometry_id=attached_cup_geometry_id,
        attached_cup_instance=attached_cup_instance,
        config=config,
    )
    set_robot_configuration(scene, config["robot"]["start"])
    plant.SetFreeBodyPose(plant_context, cup_body, cup_initial_pose)
    if visualize:
        meshcat.SetCameraPose([6.0, -8.0, 10.0], [-0.5, 0.0, 0.5])
        diagram.ForcedPublish(root_context)
    return scene


def main() -> None:
    scene = build_scene()
    print(f"Meshcat URL: {scene.meshcat.web_url()}")
    print("Configuration dimension:", scene.plant.num_positions())
    print("Initial configuration:", scene.plant.GetPositions(scene.plant_context))
    input("Press Enter to close the scene... ")


if __name__ == "__main__":
    main()
