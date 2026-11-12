"""Generate README summaries from the pipeline JSON reports."""
import json
from pathlib import Path


def load_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def classification_table(report):
    rows = [
        '| Variant | Algorithm | ROC-AUC | F1 (tuned) | Precision (tuned) | Recall (tuned) | TN | FP | FN | TP |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for variant, details in report['variants'].items():
        for name, metrics in details['models'].items():
            tn, fp = metrics['confusion_matrix_tuned'][0]
            fn, tp = metrics['confusion_matrix_tuned'][1]
            rows.append(f"| {variant} | {name} | {metrics['roc_auc']:.4f} | {metrics['f1_score_tuned']:.4f} | {metrics['precision_tuned']:.4f} | {metrics['recall_tuned']:.4f} | {tn} | {fp} | {fn} | {tp} |")
    return '\n'.join(rows)


def insights_markdown(insights):
    lines = [
        '### Top observed factors', '',
        'These are associations in the dataset, not causal effects. Sample sizes are included to make small groups visible.', '',
        '| Factor | Group | Appointments | No-show rate |', '|---|---|---:|---:|',
    ]
    factors = insights.get('top_factors', {})
    for factor in ('place_type', 'age_group', 'specialty', 'companion'):
        for label, values in sorted(factors.get(factor, {}).items(), key=lambda item: item[1]['no_show_rate'], reverse=True)[:3]:
            lines.append(f"| {factor.replace('_', ' ')} | {label} | {values['appointments']:,} | {values['no_show_rate']:.1%} |")
    lines += ['', '### Demand by specialty, weekday, and location', '', '**Specialty appointment counts**', '', '| Specialty | Appointments |', '|---|---:|']
    for name, count in sorted(insights.get('demand_by_specialty', {}).items(), key=lambda item: item[1], reverse=True)[:10]:
        lines.append(f'| {name} | {count:,} |')
    lines += ['', '**Weekday appointment counts**', '', '| Weekday | Appointments |', '|---|---:|']
    for name, count in sorted(insights['weekday_volume'].items(), key=lambda item: item[1], reverse=True):
        lines.append(f'| {name} | {count:,} |')
    lines += ['', '**Top 10 recorded locations**', '', '| Location | Appointments | No-show rate |', '|---|---:|---:|']
    locations = sorted(insights.get('location_appointments', {}).items(), key=lambda item: item[1]['appointments'], reverse=True)[:10]
    for name, values in locations:
        lines.append(f"| {name} | {values['appointments']:,} | {values['no_show_rate']:.1%} |")
    lines += ['', '### Recommendations', '']
    lines.extend(f'- {text}' for text in insights.get('recommendations', []))
    lines += ['', 'In this dataset, `storm_day_before` and `rainy_day_before` are near-duplicates and approximately 94% ones. `disability=Missing` is 0% in real municipalities (9 records). Top factors reflect data artifacts as well as observed patterns, not causes.',
               'Among real municipalities, hypertension, diabetes, and disability differ by no more than one percentage point; use standard confirmation and evaluate changes with an A/B test. Caregiver or transport needs remain hypotheses to validate.']
    return '\n'.join(lines)


def update_readme_from_reports():
    classification = load_json('outputs/classification_report.json')
    forecasting = load_json('outputs/forecasting_report.json')
    insights = load_json('outputs/insights.json')
    impact = load_json('outputs/impact.json')
    deployed = classification['deployed_variant']
    selected_classifier = classification['variants'][deployed]['best_model']
    selected_clf_metrics = classification['variants'][deployed]['models'][selected_classifier]
    selected_forecast = forecasting['best_model']
    recursive = forecasting['protocols']['recursive_14_day_rolling_origin'][selected_forecast]
    additional = forecasting.get('additional_metrics', {})

    daily_rows = ['| Protocol | Algorithm | R-squared | MAE | WAPE | MAPE (days >=10 appointments) |', '|---|---|---:|---:|---:|---:|']
    for protocol, label in [('one_step_ahead', 'One-step ahead'), ('recursive_14_day_rolling_origin', 'Recursive 14-day rolling origin')]:
        for name, metrics in forecasting['protocols'][protocol].items():
            restricted = additional.get('days_with_at_least_10_appointments', {}).get(protocol, {}).get(name, {})
            mape10 = restricted.get('mape', metrics.get('mape_volume_at_least_10'))
            daily_rows.append(f"| {label} | {name} | {metrics['r2']:.4f} | {metrics['mae']:.2f} | {metrics['wape']:.2f}% | {mape10:.2f}% |")
    weekly_rows = ['| Algorithm | Weeks | R-squared | WAPE |', '|---|---:|---:|---:|']
    for name, metrics in additional.get('weekly_aggregate_one_step_ahead', {}).items():
        weekly_rows.append(f"| {name} | {metrics['weeks']} | {metrics['r2']:.4f} | {metrics['wape']:.2f}% |")
    restricted_rows = ['| Protocol | Algorithm | Included days/observations | Excluded days/observations | R-squared | WAPE | MAPE |', '|---|---|---:|---:|---:|---:|---:|']
    for protocol, models in additional.get('days_with_at_least_10_appointments', {}).items():
        if protocol == 'exclusion_rule':
            continue
        for name, metrics in models.items():
            included = metrics.get('included_days', metrics.get('included_observations', 0))
            excluded = metrics.get('excluded_days', metrics.get('excluded_observations', 0))
            restricted_rows.append(f"| {protocol.replace('_', ' ')} | {name} | {included} | {excluded} | {metrics['r2']:.4f} | {metrics['wape']:.2f}% | {metrics['mape']:.2f}% |")
    specialty_rows = ['| Specialty | Appointments | Model | Recursive WAPE | Recursive MAE |', '|---|---:|---|---:|---:|']
    for name, detail in forecasting.get('specialty_models', {}).items():
        specialty_rows.append(f"| {name} | {detail['appointment_count']:,} | {detail['best_model']} | {detail['metrics']['wape']:.2f}% | {detail['metrics']['mae']:.2f} |")
    place_rows = ['| Place type | Appointments | No-show rate |', '|---|---:|---:|']
    for name, detail in insights['place_type'].items():
        place_rows.append(f"| {name} | {detail['appointments']:,} | {detail['no_show_rate']:.1%} |")
    scenarios = ['| Reach | Conversion | Prevented no-shows (scenario) | Share of all test no-shows |', '|---:|---:|---:|---:|']
    for item in impact['outreach_scenarios']['scenarios']:
        scenarios.append(f"| {item['reach']:.0%} | {item['conversion']:.0%} | {item['expected_prevented_no_shows']:.1f} | {item['prevented_share_of_all_test_no_shows']:.2%} |")
    test_impact = impact['classifier_test_predictions']

    readme = f'''# Medical Appointment No-Show Prediction and Demand Forecasting

Python and Streamlit tools for retrospective no-show risk analysis and daily demand forecasting at the University of Vale do Itajaí Center of Specialization in Physical and Intellectual Rehabilitation (CER).

## Overview and problem statement

The project evaluates appointment no-show risk and historical daily appointment demand to support operational planning. These are analyses of this dataset; associations do not establish causes or guarantee an intervention effect.

## Dataset audit and data dictionary

The local source contains 109,593 rows; exact duplicate removal leaves 109,557 records. Dates span 2020-01-01 to 2021-05-12. The dataset source described in the project PDF is [Medical Appointments No-Show (Mendeley Data)](https://data.mendeley.com/datasets/wm6w2fvkfj/1). Local place values include synthetic names and may differ from the cited source release.

| Field | Type | Description |
|---|---|---|
| `specialty` | Categorical | Therapy specialty; blank is `Missing`, distinct from `sem especialidade`. |
| `appointment_time` | Integer | Scheduled appointment hour. |
| `gender` | Categorical | Recorded gender. |
| `no_show` | Categorical | Target: `yes` missed, `no` attended. |
| `disability` | Categorical | Recorded disability category or `Missing`. |
| `place` | Categorical | Recorded municipality/place; some local values are synthetic. |
| `appointment_shift` | Categorical | Morning or afternoon. |
| `age` | Numeric | Age in years; missingness is retained. |
| `under_12_years_old`, `over_60_years_old` | Binary | Age group indicators. |
| `patient_needs_companion` | Binary | Recorded companion indicator. |
| `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` | Binary | Source health/disability indicators, spellings preserved. |
| `Scholarship` | Binary | Recorded social assistance indicator. |
| `SMS_received` | Binary | Recorded SMS receipt indicator. |
| `average_temp_day`, `max_temp_day` | Numeric | Weather fields attached to appointment rows. |
| `average_rain_day`, `max_rain_day` | Numeric | Precipitation fields attached to appointment rows. |
| `rain_intensity`, `heat_intensity` | Categorical | Derived weather categories. |
| `rainy_day_before`, `storm_day_before` | Binary | Previous-day weather flags. |
| `appointment_date_continuous` | Date | Appointment date. |

## Folder structure

```text
medical/
|-- data/                 # Local source CSV and shipped aggregate
|-- src/                  # Cleaning, models, impact and reporting
|-- notebooks/            # Analysis notebooks
|-- models/               # Pipeline-generated artifacts
|-- outputs/              # JSON reports, insights and charts
|-- tests/                # Unit and smoke tests
|-- app.py
|-- run_pipeline.py
|-- requirements.txt
`-- README.md
```

## Methodology

Classification retains the established leakage-free feature sets, stratified split, training-only preprocessing, threshold selection, and model-selection rule. Forecasting retains its chronological split, existing calendar/lag/rolling features, and recursive rolling-origin WAPE selection. Weekly and volume-restricted metrics are diagnostics only. Per-specialty models use the same forecasting procedure and selection rule on specialty-level daily series.

## Results

### Classification results

Generated from `outputs/classification_report.json`; confusion counts are TN, FP, FN, TP.

{classification_table(classification)}

XGBoost was selected for both variants under the existing combined ROC-AUC and tuned-F1 rule. For the deployed `no_place` variant, XGBoost also has the highest ROC-AUC and tuned F1 among the compared algorithms; no new search or threshold rule was introduced.

### Daily forecasting results

Overall model selected by recursive WAPE: **{selected_forecast}**.

{chr(10).join(daily_rows)}

### Weekly-aggregate diagnostics

One-step test actuals and predictions are summed by Monday-to-Sunday calendar week. These additional metrics supplement and do not replace daily scores.

{chr(10).join(weekly_rows)}

### Metrics restricted to days with at least 10 appointments

Exclusion rule: include an observation only when actual appointment count is >=10; exclude lower-count observations before calculating all metrics. Recursive values count rolling-origin observations, including repeated dates from overlapping origins.

{chr(10).join(restricted_rows)}

### Per-specialty forecasting models

The three most common named specialties receive models trained on their own specialty-level daily series. Tab 2 uses a dedicated model when that specialty is selected and location is `All`; combined specialty/location filters use the overall model on filtered counts and are labeled approximate.

{chr(10).join(specialty_rows)}

## Business impact scenarios

At the deployed threshold ({test_impact['threshold']:.3f}), test predictions flag {test_impact['flagged_count']:,} of {test_impact['test_rows']:,} appointments ({test_impact['flagged_share']:.1%}); precision is {test_impact['precision']:.1%} and recall is {test_impact['recall']:.1%}.

Prevented no-shows are calculated as test-set true positives x reach x conversion. These are scenarios, not forecasts.

{chr(10).join(scenarios)}

Recursive MAE is reduced by {impact['staffing']['relative_mae_reduction_percent']:.1f}% relative to seasonal naive. {impact['staffing']['caveat']}

### Deployment feasibility

{impact['deployment_feasibility']['paragraph']}

## Insights

{insights_markdown(insights)}

## Dashboard

Start the three-tab app with `streamlit run app.py`. Tab 2 shows forecasts, diagnostics, and dedicated specialty models where available. Tab 3 shows business impact scenarios, descriptive insights, and the data dictionary.

## Setup

```powershell
py -3.14 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Also verified on Python 3.12 (Linux). Metrics can vary by approximately 0.003 across machines.

Run the pipeline with the authorized local `data/raw.csv`:

```bash
python run_pipeline.py
```

Run tests with:

```bash
python -m pytest -v
```

## Screenshots

![No-show predictor](docs/screenshots/tab1_noshow.png)

![Demand forecaster](docs/screenshots/tab2_forecast.png)

![Insights](docs/screenshots/tab3_insights.png)

## Limitations

The PDF classification and forecast targets were missed. Synthetic place names weaken municipality signals, weather varies by appointment row, and near-empty periods destabilize percentage metrics. Associations are not causal claims, and forecast absolute error remains large. Outreach effects require prospective testing.

<!-- GENERATED_PLACE_STATS_START -->
{chr(10).join(place_rows)}
<!-- GENERATED_PLACE_STATS_END -->

<!-- GENERATED_TARGET_AUDIT_START -->
- Deployed classifier: {deployed} / {selected_classifier}; ROC-AUC {selected_clf_metrics['roc_auc']:.4f}; positive-class F1 {selected_clf_metrics['f1_score_tuned']:.4f}.
- Recursive forecast model: {selected_forecast}; R-squared {recursive['r2']:.4f}; MAPE {recursive['mape']:.2f}%.
<!-- GENERATED_TARGET_AUDIT_END -->
'''
    Path('README.md').write_text(readme, encoding='utf-8')
