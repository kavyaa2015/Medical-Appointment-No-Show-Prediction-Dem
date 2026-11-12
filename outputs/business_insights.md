# Business insights (computed from cleaned appointment data)

**SMS:** no association observed; recommend an A/B test before relying on SMS.

## No-show by place type

| Segment | Appointments | No-show rate |
|---|---:|---:|
| missing | 11509 | 34.8% |
| real_municipality | 44363 | 9.1% |
| synthetic_or_other | 53685 | 49.9% |

## Age group (includes Unknown)

| Segment | Appointments | No-show rate |
|---|---:|---:|
| 0-12 | 44417 | 27.7% |
| 13-18 | 20790 | 35.1% |
| 19-40 | 10567 | 45.5% |
| 41-60 | 4633 | 30.6% |
| 60+ | 6221 | 22.4% |
| Unknown | 22929 | 33.3% |

## Specialty

| Segment | Appointments | No-show rate |
|---|---:|---:|
| Missing | 20100 | 40.7% |
| assist | 635 | 26.9% |
| enf | 1681 | 19.8% |
| occupational therapy | 11318 | 23.9% |
| pedagogo | 3535 | 16.0% |
| physiotherapy | 21001 | 34.4% |
| psychotherapy | 28642 | 32.3% |
| sem especialidade | 324 | 52.8% |
| speech therapy | 22321 | 27.9% |

## Companion

| Segment | Appointments | No-show rate |
|---|---:|---:|
| 0 | 52591 | 34.8% |
| 1 | 56966 | 29.0% |

## Shift

| Segment | Appointments | No-show rate |
|---|---:|---:|
| afternoon | 59321 | 30.7% |
| morning | 50236 | 33.1% |

## Heat intensity by place type

| Place type | Heat category | Appointments | No-show rate |
|---|---|---:|---:|
| missing | cold | 3129 | 35.7% |
| missing | heavy_cold | 1474 | 39.0% |
| missing | heavy_warm | 119 | 51.3% |
| missing | mild | 4673 | 30.8% |
| missing | warm | 2114 | 38.7% |
| real_municipality | cold | 12836 | 9.7% |
| real_municipality | heavy_cold | 722 | 11.9% |
| real_municipality | heavy_warm | 509 | 11.4% |
| real_municipality | mild | 21217 | 7.9% |
| real_municipality | warm | 9079 | 10.7% |
| synthetic_or_other | cold | 14221 | 50.7% |
| synthetic_or_other | heavy_cold | 6114 | 61.3% |
| synthetic_or_other | heavy_warm | 1102 | 50.1% |
| synthetic_or_other | mild | 21003 | 43.4% |
| synthetic_or_other | warm | 11245 | 54.7% |

## SMS reminder status

no association observed; recommend an A/B test before relying on SMS

## Specialty data quality

`sem especialidade`: 324 appointments, 52.8% no-show. Blank specialty (`Missing`): 20100 appointments, 40.7% no-show.

## Weekday appointment volume

| Weekday | Appointments |
|---|---:|
| Friday | 17024 |
| Monday | 16236 |
| Saturday | 16968 |
| Sunday | 11094 |
| Thursday | 18369 |
| Tuesday | 15536 |
| Wednesday | 14330 |

## Health segments and reminder recommendations

Within real municipalities, hypertension, diabetes, and disability segment no-show rates differ by no more than 1 percentage point. Use standard confirmation and test any changed outreach with an A/B test. Caregiver or transport needs remain hypotheses, not findings from these data.

### Hipertension
- missing: 0 34.9% (n=11370), 1 29.5% (n=139)
- real_municipality: 0 9.0% (n=41261), 1 10.1% (n=3102)
- synthetic_or_other: 0 50.3% (n=50560), 1 42.5% (n=3125)
- Reminder: Among real municipalities, no-show rates differ by at most 1 percentage point across this segment; use standard confirmation and evaluate changes with an A/B test.

### Diabetes
- missing: 0 34.8% (n=11426), 1 37.3% (n=83)
- real_municipality: 0 9.1% (n=43116), 1 9.5% (n=1247)
- synthetic_or_other: 0 50.0% (n=52390), 1 44.4% (n=1295)
- Reminder: Among real municipalities, no-show rates differ by at most 1 percentage point across this segment; use standard confirmation and evaluate changes with an A/B test.

### disability
- missing: Missing 24.5% (n=6331), intellectual 49.1% (n=3427), motor 44.3% (n=1751)
- real_municipality: Missing 44.4% (n=9), intellectual 8.9% (n=30839), motor 9.6% (n=13515)
- synthetic_or_other: Missing 60.2% (n=10650), intellectual 48.3% (n=28581), motor 45.4% (n=14454)
- Reminder: Among real municipalities, no-show rates differ by at most 1 percentage point across this segment; use standard confirmation and evaluate changes with an A/B test. Caregiver or transport needs are hypotheses to validate, not conclusions from these data.
