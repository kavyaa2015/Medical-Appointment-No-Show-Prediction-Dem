"""
End-to-end training and artifact generation pipeline.
Trains classification and forecasting models, exports artifacts, and generates reports.
"""

import os
import json
import platform
import sklearn, pandas as pd_version, numpy as np_version, xgboost, statsmodels
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import (
    load_raw_data,
    clean_dataset,
    engineer_features,
    prepare_classification_data,
    TOP_MUNICIPALITIES,
)
from src.classification import (
    train_and_compare_models,
    extract_feature_importance,
    save_classifier_artifact,
)
from src.forecasting import (
    build_daily_time_series,
    train_and_evaluate_forecasting,
    save_forecasting_artifact,
)
from src.utils import (
    save_metrics_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_forecast_comparison,
)
from src.insights import generate_insights


def update_readme_from_reports():
    """Fill README data and target summaries exclusively from generated JSON."""
    with open('outputs/insights.json', encoding='utf-8') as handle:
        insights = json.load(handle)
    with open('outputs/classification_report.json', encoding='utf-8') as handle:
        classification = json.load(handle)
    with open('outputs/forecasting_report.json', encoding='utf-8') as handle:
        forecasting = json.load(handle)
    versions = {'numpy': np_version.__version__, 'pandas': pd_version.__version__,
                'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__,
                'statsmodels': statsmodels.__version__}
    classification['python_version'] = forecasting['python_version'] = platform.python_version()
    classification['library_versions'] = forecasting['library_versions'] = versions
    with open('outputs/classification_report.json', 'w', encoding='utf-8') as handle:
        json.dump(classification, handle, indent=2)
    with open('outputs/forecasting_report.json', 'w', encoding='utf-8') as handle:
        json.dump(forecasting, handle, indent=2)
    place_rows = ['| Place type | Appointments | No-show rate |', '|---|---:|---:|']
    for group, values in insights['place_type'].items():
        place_rows.append(f"| {group} | {values['appointments']:,} | {values['no_show_rate']:.1%} |")
    variant = classification['deployed_variant']
    best_name = classification['variants'][variant]['best_model']
    clf = classification['variants'][variant]['models'][best_name]
    forecast_name = forecasting['best_model']
    forecast = forecasting['protocols']['recursive_14_day_rolling_origin'][forecast_name]
    audits = [
        ('Positive-class F1 > 0.70', clf['f1_score_tuned'], clf['f1_score_tuned'] > 0.70),
        ('Forecast R² > 0.65', forecast['r2'], forecast['r2'] > 0.65),
        ('Forecast MAPE < 20%', forecast['mape'], forecast['mape'] < 20),
    ]
    classifier_rows = ['| Variant | Selected model | ROC-AUC | Positive-class F1 |', '|---|---|---:|---:|']
    for name, variant_report in classification['variants'].items():
        selected = variant_report['models'][variant_report['best_model']]
        classifier_rows.append(f"| {name} | {variant_report['best_model']} | {selected['roc_auc']:.4f} | {selected['f1_score_tuned']:.4f} |")
    target_rows = [f"- **{label}:** {'MET' if passed else 'MISSED'} ({value:.4f})." for label, value, passed in audits]
    readme = open('README.md', encoding='utf-8').read()
    replacements = [
        ('<!-- GENERATED_PLACE_STATS_START -->', '<!-- GENERATED_PLACE_STATS_END -->', '\n'.join(place_rows)),
        ('<!-- GENERATED_TARGET_AUDIT_START -->', '<!-- GENERATED_TARGET_AUDIT_END -->',
         f"Deployed classifier: {variant} / {best_name}. Forecast selected by recursive WAPE: {forecast_name}.\n\n" +
         '\n'.join(classifier_rows) + '\n\n' + '\n'.join(target_rows))
    ]
    for start_marker, end_marker, content in replacements:
        start = readme.index(start_marker) + len(start_marker)
        end = readme.index(end_marker, start)
        readme = readme[:start] + '\n' + content + '\n' + readme[end:]
    with open('README.md', 'w', encoding='utf-8') as handle:
        handle.write(readme)


def run():
    print("=" * 60)
    print("STARTING MEDICAL APPOINTMENT ML PIPELINE")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)

    # 1. Load and Audit Data
    print("\n[Phase 1] Auditing and cleaning raw dataset...")
    raw_df = load_raw_data('data/raw.csv')
    print(f"Raw shape: {raw_df.shape}")
    duplicates_count = int(raw_df.duplicated().sum())
    print(f"Exact duplicates found: {duplicates_count}")

    clean_df = clean_dataset(raw_df, drop_duplicates=True)
    print(f"Cleaned shape: {clean_df.shape}")
    feat_df = engineer_features(clean_df)

    # 2. Generate EDA Visualizations
    print("\n[Phase 2] Generating EDA figures...")
    sns.set_theme(style='whitegrid')

    # Figure 1: Target distribution
    plt.figure(figsize=(6, 4))
    counts = feat_df['target'].value_counts()
    pcts = feat_df['target'].value_counts(normalize=True) * 100
    bars = plt.bar(['Attended (0)', 'No-Show (1)'], counts, color=['#2ca02c', '#d62728'], width=0.5)
    for bar, pct in zip(bars, pcts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, f"{pct:.1f}%", ha='center', fontweight='bold')
    plt.title('Overall Patient Attendance vs. No-Show Rate')
    plt.ylabel('Number of Appointments')
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_target_distribution.png', dpi=200)
    plt.close()

    # Figure 2: Specialty No-Show Rate
    plt.figure(figsize=(10, 5))
    spec_rates = feat_df.groupby('specialty')['target'].agg(['mean', 'count']).reset_index()
    spec_rates = spec_rates[spec_rates['count'] > 200].sort_values('mean', ascending=False)
    plt.barh(spec_rates['specialty'], spec_rates['mean'] * 100, color='#1f77b4')
    overall_no_show_rate = float(feat_df['target'].mean() * 100)
    plt.axvline(overall_no_show_rate, color='red', linestyle='--', label=f'Average No-Show ({overall_no_show_rate:.1f}%)')
    plt.title('No-Show Rate (%) by Medical Rehabilitation Specialty')
    plt.xlabel('No-Show Rate (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_specialty_noshow.png', dpi=200)
    plt.close()

    # Figure 3: Weather (Heat Intensity) No-Show Rate
    plt.figure(figsize=(8, 4))
    heat_rates = feat_df.groupby('heat_intensity')['target'].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=False)
    plt.bar(heat_rates['heat_intensity'], heat_rates['mean'] * 100, color='#e377c2')
    plt.axhline(feat_df['target'].mean() * 100, color='red', linestyle='--', label='Average No-Show')
    plt.title('No-Show Rate (%) by Environmental Heat Intensity')
    plt.ylabel('No-Show Rate (%)')
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_weather_noshow.png', dpi=200)
    plt.close()

    # Figure 4: Age Group No-Show Rate
    plt.figure(figsize=(8, 4))
    age_rates = feat_df.groupby('age_group')['target'].agg(['mean', 'count']).reset_index()
    plt.bar(age_rates['age_group'], age_rates['mean'] * 100, color='#ff7f0e')
    plt.title('No-Show Rate (%) across Age Groups')
    plt.ylabel('No-Show Rate (%)')
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_age_noshow.png', dpi=200)
    plt.close()

    # Figure 5: Daily Demand Overview
    daily_all = build_daily_time_series(clean_df)
    plt.figure(figsize=(14, 5))
    plt.plot(daily_all['appointment_date'], daily_all['appointment_count'], color='#3b528b', lw=1.2)
    plt.title(f"Total Daily Appointment Demand ({len(daily_all)} days: {daily_all['appointment_date'].min():%Y-%m-%d} to {daily_all['appointment_date'].max():%Y-%m-%d})")
    plt.xlabel('Date')
    plt.ylabel('Daily Appointments')
    plt.tight_layout()
    plt.savefig('outputs/figures/daily_demand_overview.png', dpi=200)
    plt.close()

    # 3. Train Classification Models
    print("\n[Phase 3] Training and evaluating classification models...")
    X_train, X_test, y_train, y_test = prepare_classification_data('data/raw.csv', test_size=0.2, random_state=42)
    print(f"Classification train samples: {len(X_train)}, test samples: {len(X_test)}")

    clf_variants = {}
    for variant, drop_cols in [('full', []), ('no_place', ['place_clean'])]:
        variant_results, best_clf_name, best_pipeline = train_and_compare_models(
            X_train.drop(columns=drop_cols), y_train, X_test.drop(columns=drop_cols), y_test,
            output_dir='models', variant=variant
        )
        clf_variants[variant] = {'models': variant_results, 'best_model': best_clf_name}
        best_metadata = variant_results[best_clf_name]
        save_classifier_artifact(
            pipeline=best_pipeline,
            metadata={
                'model_name': best_clf_name, 'variant': variant,
                'optimal_threshold': best_metadata['optimal_threshold'],
                'metrics': best_metadata, 'all_models': variant_results,
                'python_version': platform.python_version(),
                'library_versions': {'numpy': np_version.__version__, 'pandas': pd_version.__version__,
                                     'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__}
            },
            filepath=f"models/no_show_model_{variant}.joblib"
        )
        if variant == 'no_place':
            clf_results, best_clf_name, best_metadata = variant_results, best_clf_name, best_metadata
            save_classifier_artifact(best_pipeline, {
                'model_name': best_clf_name, 'variant': variant,
                'optimal_threshold': best_metadata['optimal_threshold'], 'metrics': best_metadata,
                'python_version': platform.python_version(),
                'library_versions': {'numpy': np_version.__version__, 'pandas': pd_version.__version__,
                                     'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__}
            }, 'models/no_show_model.joblib')
    print("\nClassification Comparison Results:")
    for mname, mres in clf_results.items():
        print(f"  {mname:22s} | ROC-AUC: {mres['roc_auc']:.4f} | F1 (Tuned): {mres['f1_score_tuned']:.4f} (thresh={mres['optimal_threshold']:.2f}) | F1 (Def 0.5): {mres['f1_score_default']:.4f}")

    print(f"\nBest Classification Model: {best_clf_name}")
    print("Saved classification artifact: models/no_show_model.joblib")

    # Feature Importance
    feat_imp_df = extract_feature_importance(best_pipeline)
    feat_imp_df.to_csv('outputs/feature_importances.csv', index=False)
    plot_feature_importance(feat_imp_df, top_n=15, save_path='outputs/figures/feature_importance.png')
    plot_confusion_matrix(best_metadata['confusion_matrix_tuned'], ['Show', 'No-Show'], 'outputs/figures/confusion_matrix.png')

    save_metrics_report({'variants': clf_variants, 'deployed_variant': 'no_place',
                         'python_version': platform.python_version(),
                         'library_versions': {'numpy': np_version.__version__, 'pandas': pd_version.__version__,
                                              'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__}},
                        'outputs/classification_report.json')

    # 4. Train Forecasting Models
    print("\n[Phase 4] Training and evaluating time series demand forecasters...")
    ts_results, best_ts_name, best_ts_model, test_eval_df = train_and_evaluate_forecasting(
        daily_all, test_ratio=0.20
    )
    print("\nForecasting Comparison Results (Chronological Split):")
    for protocol_name, model_results in ts_results['protocols'].items():
        print(f"  {protocol_name}")
        for fname, fres in model_results.items():
            print(f"    {fname:25s} | MAE: {fres['mae']:7.2f} | RMSE: {fres['rmse']:7.2f} | R2: {fres['r2']:6.4f} | MAPE: {fres['mape']:6.2f}% | WAPE: {fres['wape']:6.2f}%")

    print(f"\nBest Forecasting Approach: {best_ts_name}")

    save_forecasting_artifact(
        model=best_ts_model,
        model_name=best_ts_name,
        metadata={
            'metrics': ts_results['protocols']['recursive_14_day_rolling_origin'][best_ts_name],
            'all_models': ts_results,
            'python_version': platform.python_version(),
            'library_versions': {'numpy': np_version.__version__, 'pandas': pd_version.__version__,
                                 'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__,
                                 'statsmodels': statsmodels.__version__}
        },
        filepath='models/demand_forecast_model.joblib'
    )
    print("Saved forecasting artifact: models/demand_forecast_model.joblib")

    plot_forecast_comparison(test_eval_df, 'outputs/figures/forecast_test_comparison.png')
    save_metrics_report({**ts_results, 'python_version': platform.python_version(),
                         'library_versions': {'numpy': np_version.__version__, 'pandas': pd_version.__version__,
                                              'scikit_learn': sklearn.__version__, 'xgboost': xgboost.__version__,
                                              'statsmodels': statsmodels.__version__}},
                        'outputs/forecasting_report.json')

    # 5. Derive business insights directly from cleaned appointment rows.
    generate_insights(clean_df, 'outputs')
    update_readme_from_reports()
    print("Business insights exported: outputs/business_insights.md")
    print("\nPIPELINE EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == '__main__':
    run()
