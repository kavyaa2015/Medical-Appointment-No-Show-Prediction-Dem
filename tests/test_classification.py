"""Unit tests for src/classification.py module."""

import time
import numpy as np
import pandas as pd
import pytest

from src.classification import (
    build_preprocessor,
    predict_no_show,
    load_classifier_artifact,
)
from src.utils import LatencyTimer


def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert 'num' in transformer_names
    assert 'cat' in transformer_names


def test_prediction_output_and_latency():
    model_path = 'models/no_show_model.joblib'
    try:
        artifact = load_classifier_artifact(model_path)
    except FileNotFoundError:
        pytest.skip("Model artifact not yet trained. Run pipeline first.")

    sample = pd.DataFrame([{
        'specialty': 'psychotherapy',
        'gender': 'F',
        'age': 28.0,
        'place': 'ITAJAI',
        'place_clean': 'ITAJAI',
        'disability': 'intellectual',
        'appointment_shift': 'morning',
        'appointment_time': 9,
        'patient_needs_companion': 1,
        'SMS_received': 0,
        'under_12_years_old': 0,
        'over_60_years_old': 0,
        'Hipertension': 0,
        'Diabetes': 0,
        'Alcoholism': 0,
        'Handcap': 0,
        'Scholarship': 0,
        'average_temp_day': 20.0,
        'max_temp_day': 24.0,
        'average_rain_day': 0.0,
        'max_rain_day': 0.0,
        'heat_intensity': 'mild',
        'rain_intensity': 'no_rain',
        'rainy_day_before': 0,
        'storm_day_before': 0,
        'appointment_date_continuous': '2021-02-15'
    }])

    with LatencyTimer() as timer:
        res = predict_no_show(artifact, sample)

    # Check latency requirement: target < 2 seconds
    assert timer.elapsed_sec < 2.0, f"Inference took {timer.elapsed_sec}s, which exceeds 2s limit"

    # Check output format
    assert len(res['probabilities']) == 1
    prob = res['probabilities'][0]
    assert 0.0 <= prob <= 1.0
    assert res['predictions'][0] in [0, 1]
    assert res['risk_categories'][0] in ['Low Risk', 'Medium Risk', 'High Risk']
