from src.impact import build_impact_report


def test_impact_scenarios_are_derived_from_held_out_true_positives():
    report = build_impact_report(
        y_true=[1, 1, 0, 0],
        probabilities=[0.9, 0.6, 0.8, 0.1],
        threshold=0.5,
        recursive_metrics={
            'mae': 5.0,
            'Seasonal Naive (Lag 7)': {'mae': 10.0},
        },
        latency_ms=2.0,
    )
    prediction = report['classifier_test_predictions']
    assert prediction['flagged_count'] == 3
    assert prediction['true_positives'] == 2
    assert prediction['precision'] == 2 / 3
    assert prediction['recall'] == 1.0
    assert len(report['outreach_scenarios']['scenarios']) == 4
    assert report['outreach_scenarios']['scenarios'][0]['expected_prevented_no_shows'] == 0.1
    assert report['staffing']['relative_mae_reduction_percent'] == 50.0
    assert 'scenario' in report['outreach_scenarios']['definition'].lower()
