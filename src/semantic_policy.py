"""Ablation gates. All methods retain identical geometric safety checks."""
POLICIES = ('geometry_only', 'goal_only', 'full_motion')


def policy_from_config(config):
    value = config.get('semantic_policy', 'full_motion')
    if value not in POLICIES:
        raise ValueError(f'Unknown semantic policy: {value}')
    return value


def accepts(policy, *, geometry, goals, motion):
    if policy not in POLICIES:
        raise ValueError(f'Unknown semantic policy: {policy}')
    return bool(geometry and (policy == 'geometry_only' or goals)
                and (policy != 'full_motion' or motion))
