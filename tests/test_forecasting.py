"""Unit tests for src/forecasting.py module."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    build_daily_time_series,
    engineer_forecasting_features,
    evaluate_forecast,
    load_forecasting_artifact,
    forecast_future_demand,
    FORECAST_FEATURES,
)


def test_build_daily_time_series():
    dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
    df_sample = pd.DataFrame({
        'appointment_date': list(dates) + list(dates[:3]),
        'specialty': ['physiotherapy'] * 13,
        'average_temp_day': [25.0] * 13
    })
    ts = build_daily_time_series(df_sample)
    assert len(ts) == 10  # 10 continuous calendar days
    assert ts['appointment_count'].iloc[0] == 2
    assert ts['appointment_count'].iloc[9] == 1


def test_engineer_forecasting_features_leakage_free():
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    counts = np.arange(10, 40)
    df = pd.DataFrame({'appointment_date': dates, 'appointment_count': counts})
    feat = engineer_forecasting_features(df)

    # lag_1 on day 1 (index 1) must equal day 0 count
    assert feat['lag_1'].iloc[1] == counts[0]
    # lag_7 on day 7 must equal day 0 count
    assert feat['lag_7'].iloc[7] == counts[0]
    # lag 1 on day 0 must be NaN (strictly past)
    assert pd.isna(feat['lag_1'].iloc[0])


def test_forecast_features_exclude_weather():
    assert not (set(FORECAST_FEATURES) & {
        'average_temp_day', 'max_temp_day', 'average_rain_day',
        'max_rain_day', 'storm_day_before'
    })


def test_evaluate_forecast():
    y_true = np.array([10.0, 20.0, 30.0, 0.0])
    y_pred = np.array([10.0, 22.0, 28.0, 2.0])
    metrics = evaluate_forecast(y_true, y_pred)
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'mape' in metrics
    assert 'smape' in metrics
    assert 'r2' in metrics
    assert metrics['mae'] == 1.5


def test_forecast_future_demand_output():
    model_path = 'models/demand_forecast_model.joblib'
    try:
        artifact = load_forecasting_artifact(model_path)
    except FileNotFoundError:
        pytest.skip("Model artifact not yet trained. Run pipeline first.")

    dates = pd.date_range('2021-01-01', periods=60, freq='D')
    hist_df = pd.DataFrame({
        'appointment_date': dates,
        'appointment_count': np.random.default_rng(42).integers(50, 200, size=60)
    })

    future_df = forecast_future_demand(artifact, hist_df, horizon_days=14)
    assert len(future_df) == 14
    assert 'predicted_appointments' in future_df.columns
    assert future_df['appointment_date'].iloc[0] == dates[-1] + pd.Timedelta(days=1)
    assert (future_df['predicted_appointments'] >= 0).all()
