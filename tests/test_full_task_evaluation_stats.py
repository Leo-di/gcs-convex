from src.analyze_full_task_evaluation import exact_discordance_test


def test_equal_outcomes_do_not_prove_a_difference():
    assert exact_discordance_test(0,0)==1.
    assert exact_discordance_test(2,2)==1.


def test_paired_exact_test_counts_disagreements_and_is_symmetric():
    assert exact_discordance_test(5,0)==.0625
    assert exact_discordance_test(0,6)==.03125
    assert exact_discordance_test(7,2)==exact_discordance_test(2,7)
