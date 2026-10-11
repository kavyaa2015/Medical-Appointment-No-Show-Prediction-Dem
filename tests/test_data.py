"""Unit tests for src/data.py module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data import (
    clean_dataset,
    engineer_features,
    normalize_place,
    prepare_classification_data,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


def test_normalize_place():
    assert normalize_place('ITAJAÍ') == 'ITAJAI'
    assert normalize_place('ITAJA') == 'ITAJAI'
    assert normalize_place('B. CAMBORIU') == 'BALNEARIO CAMBORIU'
    assert normalize_place('CAMBORIU') == 'CAMBORIU'
    assert normalize_place('NAVEGANTES') == 'NAVEGANTES'
    assert normalize_place(None) == 'MISSING'
    assert normalize_place(np.nan) == 'MISSING'
    assert normalize_place('New Michael') == 'OTHER'


def test_clean_dataset_removes_duplicates():
    df_dummy = pd.DataFrame({
        'no_show': ['yes', 'no', 'yes', 'yes'],
        'specialty': ['pedagogo', 'enf', 'pedagogo', 'pedagogo'],
        'disability': ['motor', ' ', None, 'motor'],
        'place': ['ITAJAI', 'CAMBORIU', 'ITAJAI', 'ITAJAI'],
        'appointment_date_continuous': ['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-01']
    })
    cleaned = clean_dataset(df_dummy, drop_duplicates=True)
    assert len(cleaned) == 3
    assert 'target' in cleaned.columns
    assert cleaned['target'].iloc[0] == 1
    assert cleaned['target'].iloc[1] == 0


def test_engineer_features():
    df = pd.DataFrame([{
        'appointment_date': pd.Timestamp('2020-05-15'),
        'appointment_time': 14,
        'age': 25.0,
        'Hipertension': 1,
        'Diabetes': 0,
        'Alcoholism': 0,
        'Handcap': 0,
        'max_temp_day': 28.0,
        'average_temp_day': 22.0,
        'max_rain_day': 5.0,
        'average_rain_day': 1.0,
    }])
    feat = engineer_features(df)
    assert feat['day_of_week'].iloc[0] == 'Friday'
    assert feat['time_period'].iloc[0] == 'early_afternoon'
    assert feat['age_group'].iloc[0] == '19-40'
    assert feat['comorbidity_count'].iloc[0] == 1
    assert feat['has_chronic_condition'].iloc[0] == 1
    assert feat['temp_range'].iloc[0] == 6.0
    assert feat['rain_range'].iloc[0] == 4.0


def test_prepare_classification_data_no_leakage():
    if not Path('data/raw.csv').exists():
        pytest.skip("Private appointment-level dataset is not included in the repository.")
    X_train, X_test, y_train, y_test = prepare_classification_data('data/raw.csv', test_size=0.2, random_state=42)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) <= 109593
    # Verify target is excluded from features
    assert 'target' not in X_train.columns
    assert 'no_show' not in X_train.columns
    # Check stratify preserves balance
    assert abs(y_train.mean() - y_test.mean()) < 0.01
