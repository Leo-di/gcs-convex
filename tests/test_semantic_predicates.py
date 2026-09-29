import pytest

from src.scene_spec import Region2D
from src.semantic_predicates import (
    base_inside_region,
    base_inside_region_residual,
)


ROOM = Region2D(
    name="room",
    x_min=0.0,
    x_max=3.0,
    y_min=-1.5,
    y_max=1.5,
)


def test_inside_region() -> None:
    assert base_inside_region([1.0, 0.0, 0.0], ROOM)
    assert not base_inside_region([-0.1, 0.0, 0.0], ROOM)


def test_residual_sign() -> None:
    assert base_inside_region_residual([1.0, 0.0, 0.0], ROOM) > 0.0
    assert base_inside_region_residual([0.0, 0.0, 0.0], ROOM) == 0.0
    assert base_inside_region_residual([-0.1, 0.0, 0.0], ROOM) < 0.0


def test_invalid_configuration_shape() -> None:
    with pytest.raises(ValueError):
        base_inside_region([1.0, 0.0], ROOM)
