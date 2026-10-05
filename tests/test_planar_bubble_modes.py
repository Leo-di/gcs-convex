"""Geometry checks independent of the optimizer's success flag."""
import numpy as np

from src.planar_bubble_demo import (
    BODY_RADII, MARGIN, OBSTACLES, PICKUP, DROPOFF,
    clearance, make_cover, graph_connects, solve,
)


def test_narrow_door_changes_feasibility_with_payload():
    assert clearance(np.array([0., 0.])) > BODY_RADII["empty"] + MARGIN
    assert clearance(np.array([0., 0.])) < BODY_RADII["carrying"]
    assert clearance(np.array([0., 1.65])) > BODY_RADII["carrying"] + MARGIN


def test_disk_swept_body_envelope_stays_clear():
    angles = np.linspace(0, 2*np.pi, 49)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    for mode, body_radius in BODY_RADII.items():
        centers, radii, _ = make_cover(mode)
        for center, radius in zip(centers, radii):
            # Extremal robot surface positions: disk + body Minkowski sum.
            for point in center + (radius + body_radius)*unit:
                assert clearance(point) >= MARGIN - 1e-10


def test_only_empty_mode_connects_through_lower_passage():
    for mode in BODY_RADII:
        centers, _, edges = make_cover(mode)
        adjacency = {i: [] for i in range(len(centers))}
        for i, j in edges:
            if centers[i, 1] < .7 and centers[j, 1] < .7:
                adjacency[i].append(j)
        seen, todo = {0}, [0]  # Seed 0=start; seed 1=pickup.
        while todo:
            for j in adjacency[todo.pop()]:
                if j not in seen:
                    seen.add(j)
                    todo.append(j)
        assert (1 in seen) == (mode == "empty")


def test_closed_wide_door_rejects_task_without_fabricated_path():
    result = solve("blocked")
    assert result["status"] == "geometrically_infeasible"
    assert result["certificate"]
    assert result["graph_reachable"]["empty"]
    assert not result["graph_reachable"]["carrying"]
    assert result["segments"] == [] and result["trace"] == []
    assert result["audit"]["accepted"] is False


def test_wrong_payload_model_is_detected_by_actual_geometry():
    point = np.array([0., 0.])
    assert clearance(point)-BODY_RADII["empty"] >= MARGIN
    assert clearance(point)-BODY_RADII["carrying"] < 0
    obstacles = np.vstack([OBSTACLES, [-.1, .9, .1, 2.4]])
    assert clearance(np.array([0., 1.65]), obstacles) == 0


def test_cover_disconnection_does_not_imply_map_infeasibility():
    cover = make_cover("carrying")
    assert graph_connects(cover, PICKUP, DROPOFF)
    # Same map, deliberately incomplete two-disk cover. Only graph_connects
    # is false; it supplies no geometric impossibility certificate.
    sparse = (np.array([PICKUP, DROPOFF]), np.array([.1, .1]), [])
    assert not graph_connects(sparse, PICKUP, DROPOFF)
