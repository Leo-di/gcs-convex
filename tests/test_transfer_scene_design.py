import numpy as np
from src.prepare_transfer_scenes import design


def test_door_changes_modify_actual_collision_walls():
    d=design()
    for name,width in [('door_110',1.1),('door_100',1.0)]:
        walls={w['name']:w for w in d['scenes'][name]['world']['walls']};a=walls['shared_wall_upper'];b=walls['shared_wall_lower']
        assert np.isclose((a['center'][1]-a['size'][1]/2)-(b['center'][1]+b['size'][1]/2),width)


def test_target_moves_with_its_physical_bin_and_table():
    d=design();a=d['scenes']['dropoff_west'];b=d['scenes']['dropoff_east']
    for key in ['dropoff_bin','dropoff_table']:
        assert np.isclose(b['objects'][key]['center'][0]-a['objects'][key]['center'][0],.14)
    assert np.isclose(b['delivery_task']['place_cup_center'][0]-a['delivery_task']['place_cup_center'][0],.14)
    assert np.isclose(b['task_zones']['dropoff']['center'][0]-a['task_zones']['dropoff']['center'][0],.14)


def test_all_scenes_and_contact_denominators_fixed_before_preparation():
    d=design();assert len(d['scenes'])==8 and len(d['cases'])==24
    assert sum(c['contact'] for c in d['cases'])==8
    assert len({c['scene'] for c in d['cases'] if c['contact']})==8
    assert d==design()
