"""
Utility functions for benchmarking latency, reporting metrics,
and generating visual evaluation artifacts.
"""

from typing import Any, Dict, List, Optional
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class LatencyTimer:
    """Context manager for tracking execution latency in milliseconds and seconds."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed_sec = self.end - self.start
        self.elapsed_ms = self.elapsed_sec * 1000.0


def save_metrics_report(metrics: Dict[str, Any], filepath: str) -> None:
    """Save dictionary of metrics to a formatted JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(cm: List[List[int]], labels: List[str], save_path: str) -> None:
    """Plot and save confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(feat_df: pd.DataFrame, top_n: int = 15, save_path: str = 'outputs/figures/feature_importance.png') -> None:
    """Plot top N feature importances."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    top = feat_df.head(top_n).iloc[::-1]
    plt.barh(top['feature'], top['importance'], color='#2b5c8f')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_forecast_comparison(test_df: pd.DataFrame, save_path: str = 'outputs/figures/forecast_test_comparison.png') -> None:
    """Plot actual vs predicted daily appointment counts on test period."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(test_df['appointment_date'], test_df['appointment_count'], label='Actual Appointments', color='#1f77b4', lw=2)
    if 'XGBoost' in test_df.columns:
        plt.plot(test_df['appointment_date'], test_df['XGBoost'], label='XGBoost Forecast', color='#ff7f0e', lw=1.5, ls='--')
    if 'Seasonal_Naive' in test_df.columns:
        plt.plot(test_df['appointment_date'], test_df['Seasonal_Naive'], label='Seasonal Naive (Lag 7)', color='#2ca02c', lw=1, ls=':')
    plt.title('Daily Appointment Demand: Actual vs Forecast (Chronological Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Daily Appointments')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


DATA_DICTIONARY = {
    "specialty": "Medical rehabilitation specialty (psychotherapy, speech therapy, physiotherapy, occupational therapy, etc.)",
    "appointment_time": "Scheduled hour of appointment (7 AM to 6 PM)",
    "gender": "Patient biological gender (M = Male, F = Female, I = Indeterminate)",
    "no_show": "Target variable (yes = Patient missed appointment, no = Patient attended)",
    "disability": "Type of disability (intellectual, motor, Missing)",
    "place": "Municipality / City in the Vale do Itajaí service area",
    "appointment_shift": "Shift of appointment (morning, afternoon)",
    "age": "Patient age in years (ranges from 2 to 110)",
    "under_12_years_old": "Binary flag indicating patient is under 12 years of age",
    "over_60_years_old": "Binary flag indicating patient is over 60 years of age",
    "patient_needs_companion": "Binary flag indicating patient requires a caregiver/companion",
    "average_temp_day": "Daily average temperature (°C)",
    "average_rain_day": "Daily average precipitation (mm)",
    "max_temp_day": "Daily maximum temperature (°C)",
    "max_rain_day": "Daily maximum precipitation (mm)",
    "rainy_day_before": "Binary flag indicating rain on the preceding day",
    "storm_day_before": "Binary flag indicating storm conditions on the preceding day",
    "rain_intensity": "Rain intensity category (no_rain, weak, moderate, heavy)",
    "heat_intensity": "Heat intensity category (cold, mild, warm, heavy_cold, heavy_warm)",
    "appointment_date_continuous": "Calendar date of the appointment (YYYY-MM-DD)",
    "Hipertension": "Binary indicator for diagnosed hypertension",
    "Diabetes": "Binary indicator for diagnosed diabetes",
    "Alcoholism": "Binary indicator for diagnosed alcoholism",
    "Handcap": "Binary indicator for physical handicap",
    "Scholarship": "Binary indicator for welfare support (Bolsa Família)",
    "SMS_received": "Binary indicator for whether an SMS reminder was dispatched"
}
