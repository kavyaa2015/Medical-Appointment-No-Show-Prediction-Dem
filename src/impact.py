"""Descriptive and scenario-based impact calculations for held-out predictions."""
import numpy as np


def build_impact_report(y_true, probabilities, threshold, recursive_metrics, latency_ms):
    """Summarize thresholded test-set flags and clearly labeled scenarios."""
    actual = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predicted = scores >= float(threshold)
    positives = actual == 1
    true_positives = int(np.sum(predicted & positives))
    flagged = int(np.sum(predicted))
    no_shows = int(np.sum(positives))
    precision = true_positives / flagged if flagged else 0.0
    recall = true_positives / no_shows if no_shows else 0.0

    scenarios = []
    for reach in (0.50, 0.80):
        for conversion in (0.10, 0.20):
            prevented = true_positives * reach * conversion
            scenarios.append({
                'reach': reach,
                'conversion': conversion,
                'expected_prevented_no_shows': prevented,
                'prevented_share_of_all_test_no_shows': prevented / no_shows if no_shows else 0.0,
                'label': 'Scenario only; assumes reached true-positive flags convert at the stated rate.',
            })

    recursive_mae = float(recursive_metrics['mae'])
    naive_mae = float(recursive_metrics['Seasonal Naive (Lag 7)']['mae'])
    reduction = (naive_mae - recursive_mae) / naive_mae if naive_mae else 0.0
    return {
        'classifier_test_predictions': {
            'threshold': float(threshold),
            'test_rows': int(len(actual)),
            'all_test_no_shows': no_shows,
            'flagged_count': flagged,
            'flagged_share': flagged / len(actual) if len(actual) else 0.0,
            'true_positives': true_positives,
            'precision': precision,
            'recall': recall,
        },
        'outreach_scenarios': {
            'definition': 'Prevented no-shows = test-set true positives x reach x conversion. These are scenarios, not forecasts.',
            'scenarios': scenarios,
        },
        'staffing': {
            'comparison': 'Selected recursive model relative to seasonal naive (Lag 7)',
            'selected_model_mae': recursive_mae,
            'seasonal_naive_mae': naive_mae,
            'relative_mae_reduction': reduction,
            'relative_mae_reduction_percent': 100.0 * reduction,
            'caveat': 'Absolute forecast error remains large; this relative improvement does not imply accurate daily staffing counts.',
        },
        'deployment_feasibility': {
            'measured_inference_latency_ms': float(latency_ms),
            'paragraph': (
                f"Deployment requires specialty, gender, age, disability, appointment date/time/shift, companion and chronic-condition indicators, SMS status, and weather fields before each visit; place is omitted by the deployed no_place variant. The measured per-record model inference latency was {latency_ms:.2f} ms on this run. "
                "Retrain on a scheduled quarterly cadence or sooner when validated drift monitoring signals degradation; monitor calibration, flagged share, precision/recall, and forecast errors against fresh outcomes. "
                "Restrict health-data access, minimize retained fields, encrypt data in transit and at rest, and follow the clinic's privacy and consent requirements."
            ),
        },
    }
