"""
Data loading, cleaning, validation, and feature engineering module.
Medical Appointment No-Show Prediction & Demand Forecasting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TOP_MUNICIPALITIES = [
    'ITAJAI',
    'BALNEARIO CAMBORIU',
    'CAMBORIU',
    'NAVEGANTES',
    'ITAPEMA',
    'BOMBINHAS',
    'PENHA',
    'PORTO BELO',
    'BALNEARIO PICARRAS',
    'ILHOTA',
    'LUIZ ALVES',
]


def normalize_place(place_val: Optional[str]) -> str:
    """Normalize cities, mapping unrecognized synthetic municipality values to UNKNOWN_SYNTHETIC."""
    if pd.isna(place_val):
        return 'MISSING'
    val = str(place_val).encode('ascii', 'ignore').decode().strip().upper()
    if not val or val == 'NAN':
        return 'MISSING'
    if 'ITAJA' in val:
        return 'ITAJAI'
    if 'B. CAMBORIU' in val or 'BALN' in val and 'CAMBORIU' in val:
        return 'BALNEARIO CAMBORIU'
    if 'CAMBORIU' in val:
        return 'CAMBORIU'
    if 'NAVEGANTES' in val:
        return 'NAVEGANTES'
    if 'ITAPEMA' in val:
        return 'ITAPEMA'
    if 'BOMBINHAS' in val:
        return 'BOMBINHAS'
    if 'PENHA' in val:
        return 'PENHA'
    if 'PORTO BELO' in val:
        return 'PORTO BELO'
    if 'PIARRAS' in val or 'PICARRAS' in val:
        return 'BALNEARIO PICARRAS'
    if 'ILHOTA' in val:
        return 'ILHOTA'
    if 'LUIZ ALVES' in val:
        return 'LUIZ ALVES'
    return 'UNKNOWN_SYNTHETIC'


def load_raw_data(filepath: str = 'data/raw.csv') -> pd.DataFrame:
    """Load raw dataset and perform basic audit check."""
    df = pd.read_csv(filepath)
    return df


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Clean dataset:
    - Remove duplicate rows
    - Standardize text and categorical fields
    - Handle whitespace placeholders
    - Ensure correct target encoding
    """
    df_clean = df.copy()

    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Clean target
    if 'no_show' in df_clean.columns:
        df_clean['target'] = (df_clean['no_show'].astype(str).str.lower().str.strip() == 'yes').astype(int)

    # Standardize specialty
    if 'specialty' in df_clean.columns:
        df_clean['specialty'] = (
            df_clean['specialty']
            .fillna('Missing')
            .astype(str)
            .str.strip()
            .replace({'': 'Missing', 'nan': 'Missing', 'None': 'Missing'})
        )

    # Standardize disability
    if 'disability' in df_clean.columns:
        df_clean['disability'] = (
            df_clean['disability']
            .fillna('Missing')
            .astype(str)
            .str.strip()
            .replace({'': 'Missing', 'nan': 'Missing', 'None': 'Missing'})
        )

    # Standardize place
    if 'place' in df_clean.columns:
        df_clean['place_clean'] = df_clean['place'].apply(normalize_place)
    else:
        df_clean['place_clean'] = 'MISSING'

    # Standardize appointment date
    if 'appointment_date_continuous' in df_clean.columns:
        df_clean['appointment_date'] = pd.to_datetime(df_clean['appointment_date_continuous'])

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain and interaction features without target leakage.
    Works for both training datasets and single-row inference.
    """
    df_feat = df.copy()

    # Temporal features
    if 'appointment_date' in df_feat.columns:
        dt = df_feat['appointment_date']
        df_feat['day_of_week'] = dt.dt.day_name()
        df_feat['day_of_month'] = dt.dt.day
        df_feat['month'] = dt.dt.month
        df_feat['quarter'] = dt.dt.quarter
        df_feat['is_weekend'] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    else:
        df_feat['day_of_week'] = 'Monday'
        df_feat['day_of_month'] = 1
        df_feat['month'] = 1
        df_feat['quarter'] = 1
        df_feat['is_weekend'] = 0

    # Appointment time block
    if 'appointment_time' in df_feat.columns:
        hour = df_feat['appointment_time']
        conditions = [
            (hour < 10),
            (hour >= 10) & (hour < 13),
            (hour >= 13) & (hour < 16),
            (hour >= 16)
        ]
        choices = ['early_morning', 'late_morning', 'early_afternoon', 'late_afternoon']
        df_feat['time_period'] = np.select(conditions, choices, default='early_morning')
    else:
        df_feat['time_period'] = 'early_morning'

    # Age indicators and categories
    if 'age' in df_feat.columns:
        df_feat['age_is_missing'] = df_feat['age'].isna().astype(int)
        # Missing age remains explicit rather than being assigned to a pediatric band.
        bins = [-1, 12, 18, 40, 60, 150]
        labels = ['0-12', '13-18', '19-40', '41-60', '60+']
        df_feat['age_group'] = pd.cut(df_feat['age'], bins=bins, labels=labels).astype(object)
        df_feat.loc[df_feat['age'].isna(), 'age_group'] = 'Unknown'
        df_feat['age_group'] = df_feat['age_group'].astype(str)
    else:
        df_feat['age_is_missing'] = 0
        df_feat['age_group'] = '19-40'

    # Health condition interactions
    health_cols = ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']
    available_health = [c for c in health_cols if c in df_feat.columns]
    if available_health:
        df_feat['comorbidity_count'] = df_feat[available_health].sum(axis=1)
        df_feat['has_chronic_condition'] = (df_feat['comorbidity_count'] > 0).astype(int)
    else:
        df_feat['comorbidity_count'] = 0
        df_feat['has_chronic_condition'] = 0

    # Weather range features
    if 'max_temp_day' in df_feat.columns and 'average_temp_day' in df_feat.columns:
        df_feat['temp_range'] = df_feat['max_temp_day'] - df_feat['average_temp_day']
    else:
        df_feat['temp_range'] = 0.0

    if 'max_rain_day' in df_feat.columns and 'average_rain_day' in df_feat.columns:
        df_feat['rain_range'] = df_feat['max_rain_day'] - df_feat['average_rain_day']
    else:
        df_feat['rain_range'] = 0.0

    return df_feat


# Column definitions for classification pipeline
CATEGORICAL_FEATURES = [
    'specialty',
    'gender',
    'disability',
    'place_clean',
    'appointment_shift',
    'rain_intensity',
    'heat_intensity',
    'day_of_week',
    'time_period',
    'age_group'
]

NUMERICAL_FEATURES = [
    'age',
    'appointment_time',
    'under_12_years_old',
    'over_60_years_old',
    'patient_needs_companion',
    'Hipertension',
    'Diabetes',
    'Alcoholism',
    'Handcap',
    'Scholarship',
    'SMS_received',
    'average_temp_day',
    'average_rain_day',
    'max_temp_day',
    'max_rain_day',
    'rainy_day_before',
    'storm_day_before',
    'day_of_month',
    'month',
    'quarter',
    'is_weekend',
    'age_is_missing',
    'comorbidity_count',
    'has_chronic_condition',
    'temp_range',
    'rain_range'
]


def prepare_classification_data(
    filepath: str = 'data/raw.csv',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load, clean, engineer features, and perform stratified train/test split.
    Prevents data leakage: fit transformers ONLY on training set later.
    """
    raw_df = load_raw_data(filepath)
    clean_df = clean_dataset(raw_df)
    feat_df = engineer_features(clean_df)

    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = feat_df[feature_cols].copy()
    y = feat_df['target'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
