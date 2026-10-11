"""
Binary Classification module for Medical Appointment No-Show Prediction.
Trains, evaluates, and exports models (Logistic Regression, Random Forest, XGBoost).
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    clean_dataset,
    engineer_features,
    prepare_classification_data,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Construct preprocessing ColumnTransformer.
    Numerical: Median Imputer + Standard Scaler
    Categorical: Most Frequent Imputer + OneHotEncoder (handle_unknown='ignore')
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = 'models'
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Train Logistic Regression, Random Forest, and XGBoost.
    Evaluate on test set and return metrics, best model name, and best pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Imbalance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_ratio = neg_count / max(pos_count, 1)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=imbalance_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        preprocessor = build_preprocessor()
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipe.fit(X_train, y_train)
        probs_test = pipe.predict_proba(X_test)[:, 1]
        probs_train = pipe.predict_proba(X_train)[:, 1]

        roc_auc = roc_auc_score(y_test, probs_test)

        # Tune classification threshold on training set to maximize F1
        threshold_candidates = np.linspace(0.20, 0.80, 61)
        train_f1s = [f1_score(y_train, (probs_train >= t).astype(int)) for t in threshold_candidates]
        best_threshold = float(threshold_candidates[np.argmax(train_f1s)])

        # Test set evaluation with tuned threshold
        preds_tuned = (probs_test >= best_threshold).astype(int)
        preds_default = (probs_test >= 0.5).astype(int)

        f1_tuned = f1_score(y_test, preds_tuned)
        prec_tuned = precision_score(y_test, preds_tuned, zero_division=0)
        rec_tuned = recall_score(y_test, preds_tuned, zero_division=0)
        acc_tuned = accuracy_score(y_test, preds_tuned)
        cm_tuned = confusion_matrix(y_test, preds_tuned).tolist()

        f1_def = f1_score(y_test, preds_default)
        prec_def = precision_score(y_test, preds_default, zero_division=0)
        rec_def = recall_score(y_test, preds_default, zero_division=0)

        results[name] = {
            'roc_auc': float(roc_auc),
            'f1_score_tuned': float(f1_tuned),
            'precision_tuned': float(prec_tuned),
            'recall_tuned': float(rec_tuned),
            'accuracy_tuned': float(acc_tuned),
            'confusion_matrix_tuned': cm_tuned,
            'optimal_threshold': best_threshold,
            'f1_score_default': float(f1_def),
            'precision_default': float(prec_def),
            'recall_default': float(rec_def),
        }
        fitted_pipelines[name] = pipe

    # Select best model based on ROC-AUC + tuned F1
    best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'] + results[k]['f1_score_tuned'])
    best_pipeline = fitted_pipelines[best_model_name]

    return results, best_model_name, best_pipeline


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract and sort feature importance or coefficients from fitted pipeline."""
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['ohe']
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = np.ones(len(all_feature_names))

    feat_df = pd.DataFrame({
        'feature': all_feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return feat_df


def save_classifier_artifact(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    filepath: str = 'models/no_show_model.joblib'
) -> None:
    """Save pipeline and metadata together in a joblib package."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        'pipeline': pipeline,
        'metadata': metadata
    }
    joblib.dump(payload, filepath)


def load_classifier_artifact(filepath: str = 'models/no_show_model.joblib') -> Dict[str, Any]:
    """Load classifier artifact."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)


def predict_no_show(
    artifact: Dict[str, Any],
    input_data: pd.DataFrame,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Make single or batch predictions on input data.
    Returns probability, prediction, and risk category.
    """
    pipeline = artifact['pipeline']
    default_thresh = artifact.get('metadata', {}).get('optimal_threshold', 0.5)
    cutoff = threshold if threshold is not None else default_thresh

    # Ensure required columns exist and features are engineered
    cleaned = clean_dataset(input_data, drop_duplicates=False)
    engineered = engineer_features(cleaned)

    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    # Ensure missing columns are present
    for col in feature_cols:
        if col not in engineered.columns:
            engineered[col] = np.nan if col in NUMERICAL_FEATURES else 'Missing'

    X = engineered[feature_cols].copy()
    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= cutoff).astype(int)

    def categorize_risk(p: float) -> str:
        if p < 0.35:
            return 'Low Risk'
        elif p < 0.65:
            return 'Medium Risk'
        return 'High Risk'

    risk_categories = [categorize_risk(p) for p in probs]

    return {
        'probabilities': probs,
        'predictions': preds,
        'risk_categories': risk_categories,
        'threshold_used': cutoff
    }
