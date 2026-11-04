"""Explicit RNG streams for repeatable planning; legacy defaults unchanged."""
import copy


def planning_seed(config, component, default):
    seed=config.get('planning_seeds',{}).get(component,default)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError(f'{component} seed must be an integer in [0,2**31)')
    return seed


def seeded_config(config, trial_seed):
    if type(trial_seed) is not int or not 0 <= trial_seed < 2**31-6000:
        raise ValueError('Trial seed outside supported range')
    result=copy.deepcopy(config)
    result['planning_seeds']={name:trial_seed+1000*i for i,name in enumerate(
        ('ik','pregrasp','iris','region_audit','navigation','gcs_rounding'))}
    result['green_room']['iris']['seed']=result['planning_seeds']['iris']
    result['green_room']['random_seed']=result['planning_seeds']['region_audit']
    return result
