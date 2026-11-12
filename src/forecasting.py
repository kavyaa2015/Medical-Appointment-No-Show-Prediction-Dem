"""
Time Series Demand Forecasting module for Medical Appointment Scheduling.
Builds daily series, chronological splits, feature engineering, model comparison,
and multi-step recursive forecasting across specialties and locations.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def build_daily_time_series(
    df: pd.DataFrame,
    specialty: Optional[str] = None,
    location: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate appointments into daily time series, optionally filtered by specialty or location.
    Ensures continuous date calendar with no gaps.
    """
    filtered_df = df.copy()

    if specialty and specialty != 'All':
        if 'specialty' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['specialty'].astype(str).str.lower() == specialty.lower()]

    if location and location != 'All':
        if 'place_clean' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['place_clean'].astype(str).str.upper() == location.upper()]
        elif 'place' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['place'].astype(str).str.upper().str.contains(location.upper())]

    if 'appointment_date' not in filtered_df.columns:
        if 'appointment_date_continuous' in filtered_df.columns:
            filtered_df['appointment_date'] = pd.to_datetime(filtered_df['appointment_date_continuous'])
        else:
            raise KeyError("Dataset must contain appointment_date or appointment_date_continuous")

    # Aggregate by date
    daily_counts = filtered_df.groupby('appointment_date').size().reset_index(name='appointment_count')

    # Get full continuous calendar from min to max date
    all_dates = pd.date_range(
        start=df['appointment_date'].min(),
        end=df['appointment_date'].max(),
        freq='D'
    )
    calendar_df = pd.DataFrame({'appointment_date': all_dates})
    merged = pd.merge(calendar_df, daily_counts, on='appointment_date', how='left')
    merged['appointment_count'] = merged['appointment_count'].fillna(0).astype(int)

    # Weather aggregates per date from main dataset
    weather_cols = ['average_temp_day', 'max_temp_day', 'average_rain_day', 'max_rain_day', 'storm_day_before']
    avail_weather = [c for c in weather_cols if c in df.columns]
    if avail_weather:
        weather_daily = df.groupby('appointment_date')[avail_weather].mean().reset_index()
        merged = pd.merge(merged, weather_daily, on='appointment_date', how='left')
        for c in avail_weather:
            merged[c] = merged[c].bfill().ffill()

    return merged.sort_values('appointment_date').reset_index(drop=True)


def engineer_forecasting_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive calendar, strictly past lag, and strictly past rolling window features.
    Prevents future data leakage by shifting all series statistics by at least 1 step.
    """
    ts_df = daily_df.copy()
    dates = ts_df['appointment_date']

    # Calendar features
    ts_df['dayofweek'] = dates.dt.dayofweek
    ts_df['day'] = dates.dt.day
    ts_df['month'] = dates.dt.month
    ts_df['quarter'] = dates.dt.quarter
    ts_df['dayofyear'] = dates.dt.dayofyear
    ts_df['is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)

    counts = ts_df['appointment_count']
    log_counts = np.log1p(counts)
    ts_df['log_count'] = log_counts

    # Past lags (shifted >= 1)
    for lag in [1, 2, 3, 7, 14, 21]:
        ts_df[f'lag_{lag}'] = counts.shift(lag)
        ts_df[f'log_lag_{lag}'] = log_counts.shift(lag)

    # Past rolling statistics (shifted 1 step prior to window)
    ts_df['rolling_mean_7'] = counts.shift(1).rolling(7, min_periods=1).mean()
    ts_df['rolling_std_7'] = counts.shift(1).rolling(7, min_periods=1).std().fillna(0)
    ts_df['rolling_mean_14'] = counts.shift(1).rolling(14, min_periods=1).mean()
    ts_df['rolling_mean_28'] = counts.shift(1).rolling(28, min_periods=1).mean()

    ts_df['log_rolling_mean_7'] = log_counts.shift(1).rolling(7, min_periods=1).mean()
    ts_df['log_rolling_std_7'] = log_counts.shift(1).rolling(7, min_periods=1).std().fillna(0)
    ts_df['log_rolling_mean_14'] = log_counts.shift(1).rolling(14, min_periods=1).mean()
    ts_df['log_rolling_mean_28'] = log_counts.shift(1).rolling(28, min_periods=1).mean()

    return ts_df


FORECAST_FEATURES = [
    'dayofweek', 'day', 'month', 'quarter', 'dayofyear', 'is_weekend',
    'log_lag_1', 'log_lag_2', 'log_lag_3', 'log_lag_7', 'log_lag_14', 'log_lag_21',
    'log_rolling_mean_7', 'log_rolling_std_7', 'log_rolling_mean_14',
    'log_rolling_mean_28'
]


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, MAPE (handling small/zero values), sMAPE, and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Safe MAPE: handle zeros / low values using max(y_true, 1.0)
    safe_denominator = np.maximum(y_true, 1.0)
    mape = float(np.mean(np.abs(y_true - y_pred) / safe_denominator) * 100)

    # Symmetric MAPE (robust percentage metric bounded between 0 and 200%)
    denom_smape = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denom_smape) * 100)
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1.0) * 100)
    volume_mask = y_true >= 10
    mape_volume_10 = float(
        np.mean(np.abs(y_true[volume_mask] - y_pred[volume_mask]) / y_true[volume_mask]) * 100
    ) if volume_mask.any() else float('nan')

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'smape': float(smape),
        'wape': wape,
        'mape_volume_at_least_10': mape_volume_10,
        'r2': float(r2)
    }


def train_and_evaluate_forecasting(
    daily_df: pd.DataFrame,
    test_ratio: float = 0.20
) -> Tuple[Dict[str, Any], str, Any, pd.DataFrame]:
    """
    Chronological train/test split.
    Train and compare:
    1. Seasonal Naive / Moving Average Baseline
    2. XGBoost Regressor
    3. SARIMAX Model
    """
    feat_df = engineer_forecasting_features(daily_df)

    # Drop warm-up rows (first 21 days where lag_21 is NaN)
    valid_df = feat_df.dropna(subset=['log_lag_21']).reset_index(drop=True)

    split_idx = int(len(valid_df) * (1.0 - test_ratio))
    train_df = valid_df.iloc[:split_idx].copy()
    test_df = valid_df.iloc[split_idx:].copy()

    y_train = train_df['appointment_count'].values
    y_test = test_df['appointment_count'].values
    y_train_log = train_df['log_count'].values

    results = {}
    models = {}

    # Approach 1: Seasonal Naive Baseline (using lag_7)
    pred_baseline = test_df['lag_7'].values
    results['Seasonal Naive (Lag 7)'] = evaluate_forecast(y_test, pred_baseline)

    # Approach 1b: 7-day Moving Average Baseline
    pred_ma = test_df['rolling_mean_7'].values
    results['7-Day Moving Average'] = evaluate_forecast(y_test, pred_ma)

    # Approach 2: XGBoost Regressor (with log-stabilized target)
    xgb_reg = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    xgb_reg.fit(train_df[FORECAST_FEATURES], y_train_log)
    pred_xgb_log = xgb_reg.predict(test_df[FORECAST_FEATURES])
    pred_xgb = np.clip(np.expm1(pred_xgb_log), a_min=0, a_max=None)
    results['XGBoost Regressor'] = evaluate_forecast(y_test, pred_xgb)
    models['XGBoost Regressor'] = xgb_reg

    # A genuine non-linear ensemble comparator, not another baseline.
    random_forest = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=0.8,
        random_state=42, n_jobs=-1
    )
    random_forest.fit(train_df[FORECAST_FEATURES], y_train_log)
    pred_rf = np.clip(np.expm1(random_forest.predict(test_df[FORECAST_FEATURES])), a_min=0, a_max=None)
    results['Random Forest Regressor'] = evaluate_forecast(y_test, pred_rf)
    models['Random Forest Regressor'] = random_forest

    # Approach 3: SARIMAX Model
    try:
        sarima_order = (1, 1, 1)
        sarima_seasonal = (1, 0, 1, 7)
        sarima_model = SARIMAX(
            y_train,
            order=sarima_order,
            seasonal_order=sarima_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarima_fit = sarima_model.fit(disp=False)
        rolling_fit = sarima_fit
        pred_sarima = []
        for observed in y_test:
            pred_sarima.append(float(max(0.0, rolling_fit.forecast(steps=1)[0])))
            rolling_fit = rolling_fit.append([observed], refit=False)
        pred_sarima = np.asarray(pred_sarima)
        results['SARIMAX'] = evaluate_forecast(y_test, pred_sarima)
        models['SARIMAX'] = rolling_fit
    except Exception as e:
        results['SARIMAX'] = {'error': str(e), 'mae': 999.0, 'rmse': 999.0, 'mape': 999.0, 'smape': 999.0, 'r2': -9.0}

    # Rolling-origin protocol: origins are seven days apart and each predicts
    # recursively for up to fourteen days using only prior observations/predictions.
    recursive_truth, recursive_predictions = {}, {}
    names = ['Seasonal Naive (Lag 7)', '7-Day Moving Average', 'XGBoost Regressor', 'Random Forest Regressor', 'SARIMAX']
    for name in names:
        recursive_truth[name], recursive_predictions[name] = [], []
    for origin in range(split_idx, len(valid_df), 7):
        history_counts = valid_df.iloc[:origin]['appointment_count'].astype(float).tolist()
        recursive_histories = {name: list(history_counts) for name in names}
        try:
            sarimax_origin = SARIMAX(history_counts, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            sarima_horizon = np.clip(sarimax_origin.forecast(steps=14), a_min=0, a_max=None)
        except Exception:
            sarima_horizon = np.asarray([])
        for step in range(14):
            target_idx = origin + step
            if target_idx >= len(valid_df):
                break
            date = valid_df.iloc[target_idx]['appointment_date']
            recursive_truth['Seasonal Naive (Lag 7)'].append(float(valid_df.iloc[target_idx]['appointment_count']))
            naive_value = recursive_histories['Seasonal Naive (Lag 7)'][-7] if len(recursive_histories['Seasonal Naive (Lag 7)']) >= 7 else recursive_histories['Seasonal Naive (Lag 7)'][-1]
            recursive_predictions['Seasonal Naive (Lag 7)'].append(naive_value)
            recursive_histories['Seasonal Naive (Lag 7)'].append(naive_value)
            recursive_truth['7-Day Moving Average'].append(float(valid_df.iloc[target_idx]['appointment_count']))
            ma_value = float(np.mean(recursive_histories['7-Day Moving Average'][-7:]))
            recursive_predictions['7-Day Moving Average'].append(ma_value)
            recursive_histories['7-Day Moving Average'].append(ma_value)
            for name, model in [('XGBoost Regressor', xgb_reg), ('Random Forest Regressor', random_forest)]:
                model_history = recursive_histories[name]
                hist = pd.DataFrame({'appointment_date': pd.date_range(end=valid_df.iloc[origin - 1]['appointment_date'],
                                     periods=len(model_history), freq='D'), 'appointment_count': model_history})
                one = forecast_future_demand({'model': model, 'model_name': name}, hist, horizon_days=1)
                model_value = float(one['predicted_appointments'].iloc[0])
                recursive_truth[name].append(float(valid_df.iloc[target_idx]['appointment_count']))
                recursive_predictions[name].append(model_value)
                recursive_histories[name].append(model_value)
            if 'SARIMAX' in models:
                sarima_value = float(sarima_horizon[step]) if len(sarima_horizon) > step else (history_counts[-7] if len(history_counts) >= 7 else history_counts[-1])
                recursive_truth['SARIMAX'].append(float(valid_df.iloc[target_idx]['appointment_count']))
                recursive_predictions['SARIMAX'].append(sarima_value)
                recursive_histories['SARIMAX'].append(sarima_value)
            history_counts.append(float(valid_df.iloc[target_idx]['appointment_count']))
            # For model recursion, append each model's own prediction is handled by
            # the forecast function's internal running history.
    recursive_metrics = {name: evaluate_forecast(np.asarray(recursive_truth[name]), np.asarray(values))
                         for name, values in recursive_predictions.items() if values}
    one_step_metrics = results.copy()
    results = {'protocols': {'one_step_ahead': one_step_metrics,
                             'recursive_14_day_rolling_origin': recursive_metrics}}

    # Select among all evaluated models by recursive rolling-origin WAPE.
    model_candidates = [name for name in recursive_metrics if 'error' not in one_step_metrics.get(name, {})]
    best_model_name = min(model_candidates, key=lambda name: recursive_metrics[name]['wape'])
    results['best_model'] = best_model_name
    results['selection_metric'] = 'recursive_14_day_rolling_origin.wape'
    best_model = models.get(best_model_name, None)

    # Return test predictions dataframe for plotting
    test_eval_df = test_df[['appointment_date', 'appointment_count']].copy()
    test_eval_df['Seasonal_Naive'] = pred_baseline
    test_eval_df['XGBoost'] = pred_xgb
    test_eval_df['RandomForest'] = pred_rf
    if 'SARIMAX' in models:
        test_eval_df['SARIMAX'] = pred_sarima

    return results, best_model_name, best_model, test_eval_df


def save_forecasting_artifact(
    model: Any,
    model_name: str,
    metadata: Dict[str, Any],
    filepath: str = 'models/demand_forecast_model.joblib'
) -> None:
    """Save forecasting model artifact."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump({
        'model': model,
        'model_name': model_name,
        'metadata': metadata
    }, filepath)


def load_forecasting_artifact(filepath: str = 'models/demand_forecast_model.joblib') -> Dict[str, Any]:
    """Load forecasting model artifact."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Forecasting model not found at {filepath}")
    return joblib.load(filepath)


def forecast_future_demand(
    artifact: Dict[str, Any],
    recent_history: pd.DataFrame,
    horizon_days: int = 14
) -> pd.DataFrame:
    """
    Generate recursive multi-step forecasts for the next N days.
    Uses strictly previous predicted values to compute next lags.
    """
    model = artifact['model']
    model_name = artifact.get('model_name', 'XGBoost Regressor')

    history = recent_history.copy().sort_values('appointment_date').reset_index(drop=True)
    last_date = history['appointment_date'].max()

    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]
    future_records = []

    running_history = history[['appointment_date', 'appointment_count']].copy()

    for target_date in future_dates:
        # Build lag features dynamically from running history
        last_counts = running_history['appointment_count'].values
        last_log_counts = np.log1p(last_counts)

        dayofweek = target_date.dayofweek
        day = target_date.day
        month = target_date.month
        quarter = (target_date.month - 1) // 3 + 1
        dayofyear = target_date.dayofyear
        is_weekend = int(dayofweek in [5, 6])

        # Past lags
        log_lags = {}
        for lag in [1, 2, 3, 7, 14, 21]:
            val = last_log_counts[-lag] if len(last_log_counts) >= lag else last_log_counts[-1]
            log_lags[f'log_lag_{lag}'] = val

        # Rolling means
        log_rolling_7 = np.mean(last_log_counts[-7:]) if len(last_log_counts) >= 7 else np.mean(last_log_counts)
        log_rolling_std_7 = np.std(last_log_counts[-7:]) if len(last_log_counts) >= 7 else 0.0
        log_rolling_14 = np.mean(last_log_counts[-14:]) if len(last_log_counts) >= 14 else np.mean(last_log_counts)
        log_rolling_28 = np.mean(last_log_counts[-28:]) if len(last_log_counts) >= 28 else np.mean(last_log_counts)

        row_feat = {
            'dayofweek': dayofweek,
            'day': day,
            'month': month,
            'quarter': quarter,
            'dayofyear': dayofyear,
            'is_weekend': is_weekend,
            'log_rolling_mean_7': log_rolling_7,
            'log_rolling_std_7': log_rolling_std_7,
            'log_rolling_mean_14': log_rolling_14,
            'log_rolling_mean_28': log_rolling_28,
            **log_lags
        }
        # Artifacts written before new feature releases remain usable.  New
        # artifacts record the complete schema; legacy models receive only
        # the columns on which they were trained.
        expected_features = getattr(model, 'feature_names_in_', None)
        if expected_features is None and hasattr(model, 'get_booster'):
            expected_features = model.get_booster().feature_names
        expected_features = list(expected_features) if expected_features is not None else FORECAST_FEATURES
        X_step = pd.DataFrame([row_feat]).reindex(columns=expected_features, fill_value=0.0)

        if hasattr(model, 'predict'):
            pred_log = model.predict(X_step)[0]
            pred_count = max(0, int(round(np.expm1(pred_log))))
        else:
            # Fallback naive 7-day lag
            pred_count = max(0, int(round(running_history['appointment_count'].iloc[-7])))

        future_records.append({
            'appointment_date': target_date,
            'predicted_appointments': pred_count,
            'day_of_week': target_date.day_name(),
            'is_weekend': is_weekend
        })

        # Append to running history for subsequent multi-step lags
        new_row = pd.DataFrame({'appointment_date': [target_date], 'appointment_count': [pred_count]})
        running_history = pd.concat([running_history, new_row], ignore_index=True)

    return pd.DataFrame(future_records)
