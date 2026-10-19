from src.grounding_strategy_review import strategy_results


def row(name,good,nav,arm,seconds):
    return {'id':name,'verified_task_success':good,'bridge':{'path_length_m':nav},'operation':{'path_length_rad':arm},'elapsed_seconds':seconds}


def test_fixed_budget_strategy_replay_and_cost():
    results=strategy_results([row('a',False,1,1,3),row('b',True,4,4,5),row('c',True,2,4,7)])
    assert [r['selected'] for r in results]==[None,'b','c']
    assert [r['candidate_evaluations'] for r in results]==[1,2,3]
    assert [r['candidate_seconds_sum'] for r in results]==[3,8,15]
    assert results[2]['score']==3


def test_all_failed_candidates_do_not_become_a_success():
    assert all(not r['success'] for r in strategy_results([row('a',False,1,1,2)]))
