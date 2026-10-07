from src.publish_contact_dynamics_validation import _page, summarize_suite


def _run(*, satisfied=True, grip=True, slip=0.006):
    return {
        "satisfied": satisfied,
        "lifted_with_bilateral_contact": satisfied,
        "carried_to_room1_with_bilateral_contact": satisfied,
        "released_and_settled_in_bin": satisfied,
        "carry_slip_satisfied": satisfied,
        "actuator_effort_limits_satisfied": True,
        "unexpected_contact_count": 0 if satisfied else 1,
        "cup_attachment": False,
        "cup_pose_resets_after_initialization": 0,
        "maximum_relative_translation_change_during_carry_m": slip,
        "time_step_s": 0.001,
        "duration_s": 10.0,
        "held_bilateral_contact_fraction": 1.0 if satisfied else 0.0,
        "maximum_arm_tracking_error_rad": 0.01,
        "maximum_base_position_error_m": 0.02,
        "grip_enabled": grip,
    }


def test_suite_requires_two_positive_runs_and_failed_open_gripper_control():
    nominal = _run()
    fine = _run(slip=0.0068)
    fine["time_step_s"] = 0.0005
    no_grip = _run(satisfied=False, grip=False, slip=4.2)
    summary = summarize_suite(nominal, fine, no_grip)
    assert summary["suite_satisfied"]
    assert summary["contact_dynamics_validated"]


def test_suite_rejects_time_step_sensitive_slip():
    nominal = _run(slip=0.004)
    fine = _run(slip=0.008)
    no_grip = _run(satisfied=False, grip=False, slip=4.2)
    assert not summarize_suite(nominal, fine, no_grip)["suite_satisfied"]


def test_review_page_exposes_playback_controls_and_trace_phase_jumps():
    nominal = _run()
    nominal["trace"] = [
        {"phase": "close_gripper", "time_s": 2.5},
        {"phase": "whole_body_carry_to_dropoff", "time_s": 4.0},
        {"phase": "release_cup", "time_s": 7.0},
        {"phase": "retreat_arm", "time_s": 8.5},
    ]
    fine = _run(slip=0.0068)
    no_grip = _run(satisfied=False, grip=False, slip=4.2)
    page = _page(summarize_suite(nominal, fine, no_grip))

    assert 'id="play"' in page
    assert 'id="speed"' in page
    assert 'id="timeline"' in page
    assert 'data-seek="2.500"' in page
    assert 'data-seek="4.000"' in page
    assert 'data-seek="7.000"' in page
    assert 'data-seek="8.500"' in page
    assert "a.mixer.timeScale = Number(speed.value)" in page
