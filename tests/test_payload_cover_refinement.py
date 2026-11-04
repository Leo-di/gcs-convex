import json
from types import SimpleNamespace
import numpy as np
import pytest
import src.payload_translation_bridge as bridge


class NarrowPassageOracle:
    """Exact clearance of a straight passage between two parallel boundaries."""
    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.exclusions = SimpleNamespace(specs=[])
        self.corridors = SimpleNamespace(specs=[])
        self.phase = 'test.carry'

    def query(self, xy):
        self.calls += 1
        clearance = .0085-abs(xy[1]+.1432778)
        self.last_residuals = dict(environment=float(clearance), internal=1.)
        return float(clearance), 1., None


def test_disconnected_grid_refines_without_relaxing_clearance(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, 'PayloadTranslationOracle', NarrowPassageOracle)
    spec = dict(lower_xy=[0., -.15], upper_xy=[.1, -.13], spacing_m=.02,
        refinement_spacing_m=.005, margin_m=.005, max_radius_m=.01, min_radius_m=.0025)
    start, goal = [0., -.1432778, 0.], [.1, -.1432778, 0.]
    with pytest.raises(bridge.PayloadBridgeRejected, match='payload_cover_disconnected'):
        bridge.plan_payload_translation(None, start, goal, None, None, None, spec, tmp_path/'blocked.json')
    spec['connectivity_refinement_rounds'] = 1
    path, report = bridge.plan_payload_translation(None, start, goal, None, None, None, spec, tmp_path/'accepted.json')
    assert report['satisfied']
    assert [r['connected'] for r in report['cover_attempts']] == [False, True]
    assert report['cover_attempts'][1]['refinement_spacing_m'] == .0025
    assert report['minimum_clearance_m'] >= .005-1e-7
    assert np.allclose(path[[0,-1]], [start, goal])
    for i in range(2):
        cover = json.loads((tmp_path/f'payload_cover_round_{i}.json').read_text())
        assert cover['margin_m'] == .005 and cover['min_radius_m'] == .0025


def test_refinement_budget_validation():
    from src.stateful_transport_grounding import validate_task
    from src.scene_spec import load_scene_config
    spec = load_scene_config('configs/room_entry_transport.yaml')
    for invalid in (-1, 3, .5, True):
        spec['carry']['connectivity_refinement_rounds'] = invalid
        with pytest.raises(ValueError, match='Connectivity'):
            validate_task(spec)
