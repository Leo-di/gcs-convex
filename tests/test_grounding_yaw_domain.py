import pytest
from src.joint_grounding_study import validate_bridge_yaws


def test_domain_mismatch_rejected_before_planning():
    config={'robot':{'start':[0,0,0]},'green_room':{'navigation_regions':['lane']},'gcs_base_planning':{'regions':[{'name':'lane','yaw_bounds':[-.01,.01]}]}}
    candidates=[{'base_q':[.65,0,-.2]}]
    with pytest.raises(ValueError,match='outside navigation domain'):
        validate_bridge_yaws(config,candidates,'se2_gcs')
    config['gcs_base_planning']['regions'][0]['yaw_bounds']=[-.4,.4]
    validate_bridge_yaws(config,candidates,'se2_gcs')
    with pytest.raises(ValueError,match='Fixed-yaw'):
        validate_bridge_yaws(config,candidates,'repair_gcs')
