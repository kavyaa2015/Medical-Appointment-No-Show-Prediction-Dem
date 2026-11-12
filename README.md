# Medical Appointment No-Show Prediction and Demand Forecasting

A Python and Streamlit project for appointment no-show risk estimation and daily demand forecasting using a rehabilitation appointment dataset.

## Overview and problem statement

The project evaluates two operational tasks: identify appointment records associated with higher observed no-show risk and forecast daily appointment volume. These are retrospective analyses of this dataset; associations do not establish causes or guarantee that interventions will change attendance.

## Data audit and dictionary

The source contains 109,593 appointment rows; exact duplicate removal leaves 109,557 records. Dates span 2020-01-01 through 2021-05-12. The dataset includes synthetic municipality names and row-level weather values. See the limitations and interpretation notes below.

### Data dictionary

| Column | Type | Description |
|---|---|---|
| `specialty` | Categorical | Appointment specialty; blank values are retained as `Missing`, distinct from `sem especialidade`. |
| `appointment_time` | Integer | Scheduled hour. |
| `gender` | Categorical | Recorded gender category. |
| `no_show` | Categorical | Target label: `yes` means missed; `no` means attended. |
| `disability` | Categorical | Recorded disability type or `Missing`. |
| `place` | Categorical | Recorded municipality/place; some values are synthetic. |
| `appointment_shift` | Categorical | Morning or afternoon. |
| `age` | Numeric | Age in years; missing ages remain represented as unknown. |
| `under_12_years_old`, `over_60_years_old` | Binary | Age group indicators. |
| `patient_needs_companion` | Binary | Recorded companion indicator. |
| `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` | Binary | Recorded health/disability flags (source spellings preserved). |
| `Scholarship` | Binary | Recorded social assistance flag. |
| `SMS_received` | Binary | Whether an SMS was recorded as received. |
| `average_temp_day`, `max_temp_day` | Numeric | Weather temperature values attached to appointment rows. |
| `average_rain_day`, `max_rain_day` | Numeric | Weather precipitation values attached to appointment rows. |
| `rain_intensity`, `heat_intensity` | Categorical | Derived weather categories. |
| `rainy_day_before`, `storm_day_before` | Binary | Previous-day weather flags. |
| `appointment_date_continuous` | Date | Appointment date. |

## Folder structure

```text
medical/
|-- data/                 # Local source CSV and shipped daily aggregate
|-- src/                  # Cleaning, classification, forecasting, insights
|-- notebooks/            # EDA and analysis notebooks (no artifact writes)
|-- models/               # Artifacts produced by run_pipeline.py
|-- outputs/              # Generated reports, insights, charts
|-- tests/                # Unit and artifact contract tests
|-- app.py                # Streamlit dashboard
|-- run_pipeline.py       # Reproducible report and artifact generation
|-- requirements.txt
`-- README.md
```

## Methodology

Classification uses stratified train/test splitting, training-only preprocessing, three model families, and a threshold selected on a validation split within training data. The deployed artifact is the `no_place` variant. Forecasting compares seasonal naive, moving average, XGBoost, Random Forest, and SARIMAX on chronological evaluation protocols. The recursive model selection criterion is WAPE. Model training and selection logic are implemented in `src/` and run by `run_pipeline.py`.

## Results

Values below are generated from `outputs/classification_report.json` and `outputs/forecasting_report.json`.

### Classification

| Variant | Selected model | ROC-AUC | Positive-class F1 |
|---|---|---:|---:|
| full | XGBoost | 0.7860 | 0.6399 |
| no_place | XGBoost | 0.7629 | 0.6116 |

Deployed model: **no_place / XGBoost**. The PDF positive-class F1 target (>0.70) was missed; the deployed value is 0.6116.

### Forecasting

Selected by recursive rolling-origin WAPE: **Random Forest Regressor**.

| Protocol | Model | R-squared | MAE | WAPE | MAPE (days >=10 appts) |
|---|---|---:|---:|---:|---:|
| One-step ahead | Seasonal Naive (Lag 7) | -0.4906 | 180.08 | 93.32% | 89.33% |
| One-step ahead | 7-Day Moving Average | 0.1237 | 165.44 | 85.73% | 182.93% |
| One-step ahead | XGBoost Regressor | 0.1909 | 134.66 | 69.78% | 70.80% |
| One-step ahead | Random Forest Regressor | 0.2832 | 115.64 | 59.92% | 56.19% |
| One-step ahead | SARIMAX | 0.1034 | 161.14 | 83.50% | 130.31% |
| Recursive 14-day rolling origin | Seasonal Naive (Lag 7) | -0.5776 | 190.11 | 101.73% | 96.49% |
| Recursive 14-day rolling origin | 7-Day Moving Average | -0.6600 | 224.91 | 120.35% | 206.52% |
| Recursive 14-day rolling origin | XGBoost Regressor | -0.1026 | 173.90 | 93.05% | 86.95% |
| Recursive 14-day rolling origin | Random Forest Regressor | 0.0097 | 156.94 | 83.98% | 75.06% |
| Recursive 14-day rolling origin | SARIMAX | -0.2687 | 221.23 | 118.38% | 185.34% |

The PDF forecast targets of R-squared >0.65 and MAPE <20% were missed. Recursive selected-model R-squared is 0.0097; MAPE on days with at least ten appointments is 75.06%. Overall recursive MAPE is 1411.39%. Overall MAPE is highly affected by zero and near-zero counts.

<!-- GENERATED_PLACE_STATS_START -->
| Place type | Appointments | No-show rate |
|---|---:|---:|
| missing | 11,509 | 34.8% |
| real_municipality | 44,363 | 9.1% |
| synthetic_or_other | 53,685 | 49.9% |
<!-- GENERATED_PLACE_STATS_END -->

<!-- GENERATED_TARGET_AUDIT_START -->
Deployed classifier: no_place / XGBoost. Forecast selected by recursive WAPE: Random Forest Regressor.

| Variant | Selected model | ROC-AUC | Positive-class F1 |
|---|---|---:|---:|
| full | XGBoost | 0.7860 | 0.6399 |
| no_place | XGBoost | 0.7629 | 0.6116 |

- **Positive-class F1 > 0.70:** MISSED (0.6116).
- **Forecast R² > 0.65:** MISSED (0.0097).
- **Forecast MAPE < 20%:** MISSED (1411.3918).
<!-- GENERATED_TARGET_AUDIT_END -->

## Insights

Generated place-type rates are shown above. Synthetic place names and row-level weather values confound broad comparisons; near-empty periods occur in January 2020, November 2020, and May 2021. In this dataset, `storm_day_before` and `rainy_day_before` are approximately 94% ones and near-duplicates. `disability=Missing` is 0% in real municipalities (9 records; rounded to 0.0%), while much more common among synthetic/other locations. Top predictive factors therefore reflect data artifacts and correlations, not causes. After stratifying by place type, hypertension, diabetes, and disability differ by at most one percentage point among real municipalities; use standard confirmation and evaluate any outreach change with an A/B test. Caregiver or transport needs are hypotheses to validate. SMS receipt rates are nearly identical in this dataset; an A/B test is needed before attributing an effect.

Exact generated summaries are available in `outputs/insights.json` and `outputs/business_insights.md`.

## Dashboard

Run `streamlit run app.py` to open the dashboard. It provides no-show scoring, demand forecasting, and operational insights/data dictionary views. Forecasts and risk scores are decision-support estimates based on the supplied data and model artifacts.

## Setup

Use Python 3.14.7 on Windows for the pinned environment; also verified on Python 3.12 (Linux). Metrics can vary by approximately 0.003 across machines.

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

Provide the authorized `data/raw.csv` locally, then generate models and reports with:

```bash
python run_pipeline.py
```

`data/raw.csv` is gitignored. The included `data/daily_series.csv` supports aggregate forecasting without raw appointment rows.

## Tests

Run the test suite with:

```bash
python -m pytest -v
```

Tests cover data processing, classification and forecasting utilities, notebook artifact-write restrictions, and deployed artifact/report consistency. Tests requiring a trained artifact or raw source data may skip when those inputs are absent.

## Limitations

The PDF targets were missed in this dataset; synthetic place names weaken location signals, row-level weather values add noisy/confounded predictors, and near-empty periods destabilize forecast percentage metrics. The results should not be read causally, and metrics can vary by approximately 0.003 across machines. More reliable geographic, weather, and operational data plus prospective evaluation are needed before operational policy changes.
  ## Screenshots

  ![No-show predictor](docs/screenshots/tab1_noshow.png)
  ![Demand forecaster](docs/screenshots/tab2_forecast.png)
  ![Insights](docs/screenshots/tab3_insights.png)