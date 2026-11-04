import numpy as np
import pytest
import json
from scripts.analyze_doorway_generalization import summarize_scene_scores,exact_binomial_interval


def test_scene_is_statistical_unit_not_repeat():
    scores=np.array([[1,1],[1,0],[0,0],[0,1]])
    r=summarize_scene_scores(scores,bootstrap_samples=1000)
    assert r['independent_scenes']==4 and r['total_runs']==8 and r['successful_runs']==4
    assert r['both_repeats_successful_scenes']==1
    assert r['both_repeats_success_interval']==exact_binomial_interval(1,4)
    duplicated=summarize_scene_scores(np.repeat(scores,2,axis=1),bootstrap_samples=1000)
    assert duplicated['scene_cluster_bootstrap_interval']==r['scene_cluster_bootstrap_interval']
    assert duplicated['both_repeats_success_interval']==r['both_repeats_success_interval']


def test_all_success_does_not_imply_population_certainty():
    r=summarize_scene_scores(np.ones((12,2)),bootstrap_samples=100)
    assert r['scene_cluster_bootstrap_interval']==[1.,1.]
    assert 0<r['both_repeats_success_interval'][0]<1
    assert r['bounded_scene_mean_hoeffding_interval'][0]<1
    assert exact_binomial_interval(0,12)[1]>0
    with pytest.raises(ValueError):exact_binomial_interval(1,0)


def test_failed_workers_and_missing_outputs_stay_in_denominator(tmp_path):
    from scripts.analyze_doorway_generalization import analyze
    methods=['geometry_only','base_only','full_motion']
    runs=[dict(scene='scene_00',trial=t,method=m,process_exit='timeout',wall_seconds=285.,success=False,
               search={'success':True,'selected':'partial','attempts':[]} if t==0 else {}) for t in range(2) for m in methods]
    protocol=dict(runs=runs,scenes=[dict(id='scene_00',family='near_lower')],
        spec=dict(planner_seeds=[1,2],analysis=dict(bootstrap_seed=1,bootstrap_samples=100)))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'comparison.json').write_text(json.dumps(runs))
    result=analyze(tmp_path)
    assert result['conditions']==6 and len(result['rows'])==6
    assert not any(row['joint_success'] for row in result['rows'])
    for s in result['statistics']:
        assert s['total_runs']==2 and s['successful_runs']==0
        assert s['reference_certificate_counts']=={'no_completed_execution':2}
