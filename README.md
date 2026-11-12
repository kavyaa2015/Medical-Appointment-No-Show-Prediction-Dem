# Medical Appointment No-Show Prediction and Demand Forecasting

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

| Variant | Algorithm | ROC-AUC | F1 (tuned) | Precision (tuned) | Recall (tuned) | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | Logistic Regression | 0.7622 | 0.6258 | 0.5028 | 0.8285 | 9240 | 5706 | 1195 | 5771 |
| full | Random Forest | 0.7804 | 0.6350 | 0.5073 | 0.8487 | 9204 | 5742 | 1054 | 5912 |
| full | XGBoost | 0.7860 | 0.6399 | 0.5152 | 0.8444 | 9410 | 5536 | 1084 | 5882 |
| no_place | Logistic Regression | 0.6624 | 0.5139 | 0.3844 | 0.7753 | 6295 | 8651 | 1565 | 5401 |
| no_place | Random Forest | 0.7506 | 0.5950 | 0.4865 | 0.7659 | 9314 | 5632 | 1631 | 5335 |
| no_place | XGBoost | 0.7629 | 0.6116 | 0.5054 | 0.7745 | 9666 | 5280 | 1571 | 5395 |

XGBoost was selected for both variants under the existing combined ROC-AUC and tuned-F1 rule. For the deployed `no_place` variant, XGBoost also has the highest ROC-AUC and tuned F1 among the compared algorithms; no new search or threshold rule was introduced.

### Daily forecasting results

Overall model selected by recursive WAPE: **Random Forest Regressor**.

| Protocol | Algorithm | R-squared | MAE | WAPE | MAPE (days >=10 appointments) |
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

### Weekly-aggregate diagnostics

One-step test actuals and predictions are summed by Monday-to-Sunday calendar week. These additional metrics supplement and do not replace daily scores.

| Algorithm | Weeks | R-squared | WAPE |
|---|---:|---:|---:|
| Seasonal Naive (Lag 7) | 15 | -0.4562 | 81.10% |
| 7-Day Moving Average | 15 | 0.6496 | 42.57% |
| XGBoost Regressor | 15 | 0.3065 | 57.51% |
| Random Forest Regressor | 15 | 0.3919 | 48.04% |
| SARIMAX | 15 | 0.6443 | 47.97% |

### Metrics restricted to days with at least 10 appointments

Exclusion rule: include an observation only when actual appointment count is >=10; exclude lower-count observations before calculating all metrics. Recursive values count rolling-origin observations, including repeated dates from overlapping origins.

| Protocol | Algorithm | Included days/observations | Excluded days/observations | R-squared | WAPE | MAPE |
|---|---|---:|---:|---:|---:|---:|
| one step ahead | Seasonal Naive (Lag 7) | 56 | 40 | -0.1670 | 64.37% | 89.33% |
| one step ahead | 7-Day Moving Average | 56 | 40 | 0.1777 | 61.68% | 182.93% |
| one step ahead | XGBoost Regressor | 56 | 40 | -0.1702 | 61.22% | 70.80% |
| one step ahead | Random Forest Regressor | 56 | 40 | -0.0393 | 54.38% | 56.19% |
| one step ahead | SARIMAX | 56 | 40 | 0.1726 | 55.95% | 130.31% |
| recursive 14 day rolling origin | Seasonal Naive (Lag 7) | 105 | 80 | -0.1935 | 66.04% | 96.49% |
| recursive 14 day rolling origin | 7-Day Moving Average | 105 | 80 | -0.1867 | 73.71% | 206.52% |
| recursive 14 day rolling origin | XGBoost Regressor | 105 | 80 | -0.5295 | 76.23% | 86.95% |
| recursive 14 day rolling origin | Random Forest Regressor | 105 | 80 | -0.4106 | 71.83% | 75.06% |
| recursive 14 day rolling origin | SARIMAX | 105 | 80 | -0.0405 | 66.77% | 185.34% |

### Per-specialty forecasting models

The three most common named specialties receive models trained on their own specialty-level daily series. Tab 2 uses a dedicated model when that specialty is selected and location is `All`; combined specialty/location filters use the overall model on filtered counts and are labeled approximate.

| Specialty | Appointments | Model | Recursive WAPE | Recursive MAE |
|---|---:|---|---:|---:|
| psychotherapy | 28,642 | XGBoost Regressor | 76.49% | 37.42 |
| speech therapy | 22,321 | Random Forest Regressor | 87.57% | 33.26 |
| physiotherapy | 21,001 | Random Forest Regressor | 84.02% | 30.49 |

## Business impact scenarios

At the deployed threshold (0.440), test predictions flag 10,675 of 21,912 appointments (48.7%); precision is 50.5% and recall is 77.4%.

Prevented no-shows are calculated as test-set true positives x reach x conversion. These are scenarios, not forecasts.

| Reach | Conversion | Prevented no-shows (scenario) | Share of all test no-shows |
|---:|---:|---:|---:|
| 50% | 10% | 269.8 | 3.87% |
| 50% | 20% | 539.5 | 7.74% |
| 80% | 10% | 431.6 | 6.20% |
| 80% | 20% | 863.2 | 12.39% |

Recursive MAE is reduced by 17.4% relative to seasonal naive. Absolute forecast error remains large; this relative improvement does not imply accurate daily staffing counts.

### Deployment feasibility

Deployment requires specialty, gender, age, disability, appointment date/time/shift, companion and chronic-condition indicators, SMS status, and weather fields before each visit; place is omitted by the deployed no_place variant. The measured per-record model inference latency was 15.16 ms on this run. Retrain on a scheduled quarterly cadence or sooner when validated drift monitoring signals degradation; monitor calibration, flagged share, precision/recall, and forecast errors against fresh outcomes. Restrict health-data access, minimize retained fields, encrypt data in transit and at rest, and follow the clinic's privacy and consent requirements.

## Insights

### Top observed factors

These are associations in the dataset, not causal effects. Sample sizes are included to make small groups visible.

| Factor | Group | Appointments | No-show rate |
|---|---|---:|---:|
| place type | synthetic_or_other | 53,685 | 49.9% |
| place type | missing | 11,509 | 34.8% |
| place type | real_municipality | 44,363 | 9.1% |
| age group | 19-40 | 10,567 | 45.5% |
| age group | 13-18 | 20,790 | 35.1% |
| age group | Unknown | 22,929 | 33.3% |
| specialty | sem especialidade | 324 | 52.8% |
| specialty | Missing | 20,100 | 40.7% |
| specialty | physiotherapy | 21,001 | 34.4% |
| companion | 0 | 52,591 | 34.8% |
| companion | 1 | 56,966 | 29.0% |

### Demand by specialty, weekday, and location

**Specialty appointment counts**

| Specialty | Appointments |
|---|---:|
| psychotherapy | 28,642 |
| speech therapy | 22,321 |
| physiotherapy | 21,001 |
| Missing | 20,100 |
| occupational therapy | 11,318 |
| pedagogo | 3,535 |
| enf | 1,681 |
| assist | 635 |
| sem especialidade | 324 |

**Weekday appointment counts**

| Weekday | Appointments |
|---|---:|
| Thursday | 18,369 |
| Friday | 17,024 |
| Saturday | 16,968 |
| Monday | 16,236 |
| Tuesday | 15,536 |
| Wednesday | 14,330 |
| Sunday | 11,094 |

**Top 10 recorded locations**

| Location | Appointments | No-show rate |
|---|---:|---:|
| UNKNOWN_SYNTHETIC | 53,685 | 49.9% |
| ITAJAI | 20,512 | 8.9% |
| MISSING | 11,509 | 34.8% |
| BALNEARIO CAMBORIU | 6,018 | 9.2% |
| CAMBORIU | 5,523 | 7.4% |
| NAVEGANTES | 3,901 | 10.5% |
| ITAPEMA | 2,665 | 9.7% |
| BOMBINHAS | 1,707 | 11.7% |
| PENHA | 1,038 | 6.2% |
| PORTO BELO | 950 | 11.7% |

### Recommendations

- Use standard confirmation and evaluate outreach changes with an A/B test.
- Treat health-segment, weather, and location associations as descriptive, not causal.
- Validate appointment volume and staffing plans against current clinic operations.

In this dataset, `storm_day_before` and `rainy_day_before` are near-duplicates and approximately 94% ones. `disability=Missing` is 0% in real municipalities (9 records). Top factors reflect data artifacts as well as observed patterns, not causes.
Among real municipalities, hypertension, diabetes, and disability differ by no more than one percentage point; use standard confirmation and evaluate changes with an A/B test. Caregiver or transport needs remain hypotheses to validate.

## Dashboard

Start the three-tab app with `streamlit run app.py`. Tab 2 shows forecasts, diagnostics, and dedicated specialty models where available. Tab 3 shows business impact scenarios, descriptive insights, and the data dictionary.

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
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
| Place type | Appointments | No-show rate |
|---|---:|---:|
| missing | 11,509 | 34.8% |
| real_municipality | 44,363 | 9.1% |
| synthetic_or_other | 53,685 | 49.9% |
<!-- GENERATED_PLACE_STATS_END -->

<!-- GENERATED_TARGET_AUDIT_START -->
- Deployed classifier: no_place / XGBoost; ROC-AUC 0.7629; positive-class F1 0.6116.
- Recursive forecast model: Random Forest Regressor; R-squared 0.0097; MAPE 1411.39%.
<!-- GENERATED_TARGET_AUDIT_END -->
