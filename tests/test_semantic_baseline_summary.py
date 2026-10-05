from src.summarize_semantic_baselines import summarize


def record(method,cell='positive',cost=10.,status='completed'):
    valid=cost is not None
    return {'method':method,'cell':cell,'allowed':['x110'] if cell!='negative' else [],
            'status':status,'parent_wall_seconds':300 if status.startswith('timeout') else 40,
            'timed_out':status.startswith('timeout'),'enumeration_complete':status=='completed',
            'attempts':[], 'best':{'cost':cost,'discovered_seconds':20} if valid else None,
            'incumbent_history':[{'seconds':20,'cost':cost}] if valid else []}


def test_timeout_with_audited_incumbent_is_success_but_not_complete_enumeration():
    data=summarize([record('B2_enumerate',status='timeout_with_incumbent')])
    row=data['methods']['B2_enumerate']
    assert row['audited_successes']==1 and row['timeouts']==1
    assert row['fully_completed_enumerations']==0
    assert row['by_budget']['10']['successes']==0
    assert row['by_budget']['30']['successes']==1


def test_failures_and_structural_negatives_do_not_disappear_from_denominators():
    records=[record('B1_first_valid',cost=None,status='timeout_without_solution'),
             record('B1_first_valid','negative',None,'unreachable_in_compiled_graph')]
    row=summarize(records)['methods']['B1_first_valid']
    assert row['positive_runs']==1 and row['no_valid_plan']==1
    assert row['median_cost_successes'] is None
    assert row['negative_runs']==1 and row['negative_rejected']==1


def test_quality_comparison_uses_only_complete_paired_successes():
    records=[record('B1_first_valid',cost=12),record('B2_enumerate',cost=10),record('joint_gcs',cost=11),
             record('joint_gcs','unpaired',cost=1)]
    result=summarize(records)
    assert result['paired_all_success_count']==1
    assert result['joint_worse_than_enumerate']==1
    assert abs(result['median_joint_vs_enumerate_relative']-.1)<1e-10


def test_startup_failure_without_worker_metadata_remains_a_positive_failure():
    failed=record('joint_gcs','original__all__seed0',None,'error')
    failed.pop('allowed')
    row=summarize([failed])['methods']['joint_gcs']
    assert row['positive_runs']==1 and row['no_valid_plan']==1
    assert row['negative_runs']==0 and row['errors']==1
