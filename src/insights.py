"""Data-derived business insights and segment summaries."""
import json
from pathlib import Path
import pandas as pd


def _summary(df, by):
    return {
        str(key): {'appointments': int(group['target'].size),
                   'no_show_rate': float(group['target'].mean())}
        for key, group in df.groupby(by, dropna=False, observed=False)
    }


def generate_insights(cleaned_df, output_dir='outputs'):
    df = cleaned_df.copy()
    if 'target' not in df:
        df['target'] = (df['no_show'].astype(str).str.lower() == 'yes').astype(int)
    if 'age_group' not in df:
        from src.data import engineer_features
        df = engineer_features(df)
    known_places = set(['ITAJAI', 'BALNEARIO CAMBORIU', 'CAMBORIU', 'NAVEGANTES', 'ITAPEMA',
                        'BOMBINHAS', 'PENHA', 'PORTO BELO', 'BALNEARIO PICARRAS', 'ILHOTA', 'LUIZ ALVES'])
    df['place_type'] = df['place_clean'].map(lambda value: 'missing' if value == 'MISSING' else
                                              'real_municipality' if value in known_places else 'synthetic_or_other')
    insights = {
        'place_type': _summary(df, 'place_type'),
        'heat_intensity_by_place_type': {
            str(place): _summary(group, 'heat_intensity')
            for place, group in df.groupby('place_type', observed=False)
        },
        'sms_received': _summary(df, 'SMS_received'),
        'sms_conclusion': 'no association observed; recommend an A/B test before relying on SMS',
        'age_group': _summary(df, 'age_group'),
        'specialty': _summary(df, 'specialty'),
        'companion': _summary(df, 'patient_needs_companion'),
        'shift': _summary(df, 'appointment_shift'),
        'weekday_volume': {str(k): int(v) for k, v in df.groupby(df['appointment_date'].dt.day_name()).size().items()}
    }
    health_segments = {}
    for col in ['Hipertension', 'Diabetes', 'disability']:
        if col in df:
            health_segments[col] = {
                str(place): _summary(group, col)
                for place, group in df.groupby('place_type', observed=False)
            }
    insights['health_segments'] = health_segments
    insights['segment_reminder_recommendations'] = {
        'Hipertension': 'Use standard confirmation; monitor segment attendance by municipality.',
        'Diabetes': 'Use standard confirmation; monitor segment attendance by municipality.',
        'disability': 'Confirm caregiver and accessible transport needs before the visit.'
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, 'insights.json').write_text(json.dumps(insights, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Business insights (computed from cleaned appointment data)', '',
             '**SMS:** no association observed; recommend an A/B test before relying on SMS.', '']
    for title, key in [('No-show by place type', 'place_type'), ('Age group (includes Unknown)', 'age_group'),
                       ('Specialty', 'specialty'), ('Companion', 'companion'), ('Shift', 'shift')]:
        lines += [f'## {title}', '', '| Segment | Appointments | No-show rate |', '|---|---:|---:|']
        for label, values in insights[key].items():
            lines.append(f"| {label} | {values['appointments']} | {values['no_show_rate']:.1%} |")
        lines.append('')
    lines += ['## Heat intensity by place type', '', '| Place type | Heat category | Appointments | No-show rate |', '|---|---|---:|---:|']
    for place, categories in insights['heat_intensity_by_place_type'].items():
        for category, values in categories.items():
            lines.append(f"| {place} | {category} | {values['appointments']} | {values['no_show_rate']:.1%} |")
    lines += ['', '## SMS reminder status', '', insights['sms_conclusion'], '']
    if 'sem especialidade' in insights['specialty'] and 'Missing' in insights['specialty']:
        named, blank = insights['specialty']['sem especialidade'], insights['specialty']['Missing']
        lines += ['## Specialty data quality', '',
                  f"`sem especialidade`: {named['appointments']} appointments, {named['no_show_rate']:.1%} no-show. Blank specialty (`Missing`): {blank['appointments']} appointments, {blank['no_show_rate']:.1%} no-show.", '']
    lines += ['## Weekday appointment volume', '', '| Weekday | Appointments |', '|---|---:|']
    for day, count in insights['weekday_volume'].items():
        lines.append(f'| {day} | {count} |')
    lines += ['', '## Health segments and reminder recommendations', '']
    for condition, by_place in insights['health_segments'].items():
        lines.append(f'### {condition}')
        for place, values in by_place.items():
            lines.append(f"- {place}: " + ', '.join(f"{label} {value['no_show_rate']:.1%} (n={value['appointments']})" for label, value in values.items()))
        lines.append(f"- Reminder: {insights['segment_reminder_recommendations'][condition]}")
        lines.append('')
    Path(output_dir, 'business_insights.md').write_text('\n'.join(lines), encoding='utf-8')
    return insights
