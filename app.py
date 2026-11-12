"""
Medical Appointment No-Show Prediction & Demand Forecasting Application.
Interactive Streamlit Dashboard for Healthcare Operations & Resource Management.
"""

import time
import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import (
    load_raw_data,
    clean_dataset,
    engineer_features,
    TOP_MUNICIPALITIES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from src.classification import (
    load_classifier_artifact,
    predict_no_show,
    extract_feature_importance,
)
from src.forecasting import (
    load_forecasting_artifact,
    build_daily_time_series,
    forecast_future_demand,
)
from src.utils import LatencyTimer, DATA_DICTIONARY


st.set_page_config(
    page_title="CER Medical Operations & Scheduling AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #d9534f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f0ad4e;
        font-weight: bold;
    }
    .risk-low {
        color: #5cb85c;
        font-weight: bold;
    }
    .latency-tag {
        font-size: 0.85rem;
        background-color: #e9ecef;
        padding: 3px 8px;
        border-radius: 4px;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classification_artifact():
    model_path = 'models/no_show_model.joblib'
    if not os.path.exists(model_path):
        return None
    return load_classifier_artifact(model_path)


@st.cache_resource
def get_forecasting_artifact():
    model_path = 'models/demand_forecast_model.joblib'
    if not os.path.exists(model_path):
        return None
    return load_forecasting_artifact(model_path)


@st.cache_data
def get_dataset():
    if not os.path.exists('data/raw.csv'):
        return None
    raw = load_raw_data('data/raw.csv')
    cleaned = clean_dataset(raw)
    return cleaned


@st.cache_data
def get_daily_series():
    """Load the privacy-preserving aggregate shipped with the repository."""
    path = 'data/daily_series.csv'
    if not os.path.exists(path):
        return None
    series = pd.read_csv(path, parse_dates=['appointment_date'])
    return series.sort_values('appointment_date').reset_index(drop=True)


def metric_value(metrics, key, format_spec):
    """Display only a metric saved by the training run; never invent a fallback."""
    value = metrics.get(key)
    return format(value, format_spec) if value is not None else "Unavailable"


# App Header
st.title("🏥 CER Rehabilitation Center Operations Management")
st.markdown(
    "**Clinical Intelligence System** for Vale do Itajaí Rehabilitation Center (CER): "
    "Dual-engine Machine Learning for **Patient No-Show Prediction** and **Daily Demand Forecasting**."
)

tab_pred, tab_fore, tab_insights = st.tabs([
    "🎯 Tab 1: No-Show Predictor",
    "📈 Tab 2: Demand Forecaster",
    "📊 Tab 3: Operational Insights & Data Dictionary"
])

# =========================================================================
# TAB 1: NO-SHOW PREDICTOR
# =========================================================================
with tab_pred:
    st.header("Patient No-Show Risk Assessment")
    st.markdown(
        "Evaluate individual patient appointment risk before scheduling. "
        "Allows proactive patient engagement, customized SMS reminders, and selective overbooking buffer."
    )

    clf_artifact = get_classification_artifact()

    if clf_artifact is None:
        st.warning("⚠️ Classification model artifact (`models/no_show_model.joblib`) not found. Please run the training pipeline first.")
    else:
        meta = clf_artifact.get('metadata', {})
        best_name = meta.get('model_name', 'Trained Classifier')
        optimal_t = meta.get('optimal_threshold', 0.50)
        metrics = meta.get('metrics', {})

        col_top1, col_top2, col_top3, col_top4 = st.columns(4)
        with col_top1:
            st.metric("Active Model", best_name)
        with col_top2:
            st.metric("Test ROC-AUC", metric_value(metrics, 'roc_auc', '.4f'))
        with col_top3:
            st.metric("Validation-selected F1", metric_value(metrics, 'f1_score_tuned', '.4f'))
        with col_top4:
            st.metric("Default Threshold", f"{optimal_t:.2f}")

        st.markdown("---")

        with st.form("noshow_input_form"):
            st.subheader("1. Patient & Appointment Information")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                specialty = st.selectbox(
                    "Specialty",
                    options=[
                        'psychotherapy', 'speech therapy', 'physiotherapy',
                        'occupational therapy', 'pedagogo', 'enf', 'assist', 'Missing'
                    ],
                    index=0
                )
                gender = st.selectbox("Gender", options=['M', 'F', 'I'], index=0)
                age_unknown = st.checkbox("Age unknown", value=False)
                age = st.number_input("Patient Age", min_value=1, max_value=115, value=25, disabled=age_unknown)

            with c2:
                disability = st.selectbox("Disability Type", options=['intellectual', 'motor', 'Missing'], index=0)
                shift = st.selectbox("Appointment Shift", options=['morning', 'afternoon'], index=0)

            with c3:
                hour = st.slider("Appointment Hour", min_value=7, max_value=18, value=10)
                needs_companion = st.selectbox("Needs Companion / Caregiver", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
                sms_received = st.selectbox("SMS Reminder Sent", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", index=0)

            with c4:
                appointment_date = st.date_input("Appointment date")
                threshold_cutoff = st.slider("Risk Cutoff Threshold", min_value=0.20, max_value=0.80, value=float(optimal_t), step=0.01)
                under_12 = 0 if age_unknown else int(age < 12)
                over_60 = 0 if age_unknown else int(age >= 60)

            st.subheader("2. Health Conditions & Social Assistance")
            h1, h2, h3, h4, h5 = st.columns(5)
            with h1:
                hipertension = st.checkbox("Hypertension", value=False)
            with h2:
                diabetes = st.checkbox("Diabetes", value=False)
            with h3:
                alcoholism = st.checkbox("Alcoholism", value=False)
            with h4:
                handcap = st.checkbox("Handicap", value=False)
            with h5:
                scholarship = st.checkbox("Bolsa Família Support", value=False)

            st.subheader("3. Environmental & Weather Forecast for Appointment Day")
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                avg_temp = st.slider("Average Temp (°C)", min_value=10.0, max_value=40.0, value=22.0, step=0.5)
                max_temp = st.slider("Max Temp (°C)", min_value=10.0, max_value=45.0, value=26.0, step=0.5)
            with w2:
                avg_rain = st.number_input("Average Rain (mm)", min_value=0.0, max_value=50.0, value=0.5, step=0.1)
                max_rain = st.number_input("Max Rain (mm)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
            with w3:
                heat_intensity = st.selectbox("Heat Intensity", options=['cold', 'mild', 'warm', 'heavy_cold', 'heavy_warm'], index=1)
                rain_intensity = st.selectbox("Rain Intensity", options=['no_rain', 'weak', 'moderate', 'heavy'], index=0)
            with w4:
                rain_day_before = st.checkbox("Rain Day Before", value=True)
                storm_day_before = st.checkbox("Storm Day Before", value=True)

            submit_btn = st.form_submit_button("⚡ Predict No-Show Risk", width="stretch")

        if submit_btn:
            if max_temp < avg_temp:
                st.error("Maximum temperature must be greater than or equal to average temperature.")
                st.stop()
            with st.spinner("Analyzing appointment parameters and scoring risk..."):
                input_df = pd.DataFrame([{
                    'specialty': specialty,
                    'gender': gender,
                    'age': np.nan if age_unknown else float(age),
                    'disability': disability,
                    'appointment_shift': shift,
                    'appointment_time': int(hour),
                    'patient_needs_companion': int(needs_companion),
                    'SMS_received': int(sms_received),
                    'under_12_years_old': int(under_12),
                    'over_60_years_old': int(over_60),
                    'Hipertension': int(hipertension),
                    'Diabetes': int(diabetes),
                    'Alcoholism': int(alcoholism),
                    'Handcap': int(handcap),
                    'Scholarship': int(scholarship),
                    'average_temp_day': float(avg_temp),
                    'max_temp_day': float(max_temp),
                    'average_rain_day': float(avg_rain),
                    'max_rain_day': float(max_rain),
                    'heat_intensity': heat_intensity,
                    'rain_intensity': rain_intensity,
                    'rainy_day_before': int(rain_day_before),
                    'storm_day_before': int(storm_day_before),
                    'appointment_date_continuous': pd.Timestamp(appointment_date).strftime('%Y-%m-%d')
                }])

                try:
                    with LatencyTimer() as timer:
                        pred_res = predict_no_show(clf_artifact, input_df, threshold=threshold_cutoff)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.stop()

                latency_ms = timer.elapsed_ms
                latency_sec = timer.elapsed_sec
                prob = float(pred_res['probabilities'][0])
                pred_label = int(pred_res['predictions'][0])
                risk_cat = pred_res['risk_categories'][0]

            st.markdown("### Prediction Results")
            r_col1, r_col2, r_col3 = st.columns(3)

            with r_col1:
                st.metric("Predicted No-Show Probability", f"{prob * 100:.1f}%")
            with r_col2:
                if risk_cat == 'High Risk':
                    st.error(f"🔴 Category: {risk_cat}")
                elif risk_cat == 'Medium Risk':
                    st.warning(f"🟡 Category: {risk_cat}")
                else:
                    st.success(f"🟢 Category: {risk_cat}")
            with r_col3:
                badge = f"⏱️ Latency: {latency_ms:.1f} ms"
                if latency_sec < 2.0:
                    badge += " (Target < 2.0s Met ✅)"
                st.info(badge)

            # Actionable Intervention Recommendation
            st.markdown("#### Clinical Staff Action Guidance")
            if risk_cat == 'High Risk':
                st.error(
                    "**Priority Patient Outreach Required:**\n"
                    "- 📞 Direct telephone call 24 hours prior to verify transportation and caregiver readiness.\n"
                    "- 💬 Automated WhatsApp reminder with clinic directions and weather travel advisory.\n"
                    "- 📅 Consider overbooking backup slot or offering telehealth/home-based alternative."
                )
            elif risk_cat == 'Medium Risk':
                st.warning(
                    "**Standard Preventive Engagement:**\n"
                    "- 📱 Ensure SMS confirmation reminder is dispatched 48 hours prior.\n"
                    "- 🔄 Reconfirm slot if patient has previous unattended history."
                )
            else:
                st.success(
                    "**Normal Attendance Likely:**\n"
                    "- Standard appointment confirmation protocol. No specialized staff intervention needed."
                )

            # Feature Importance preview
            st.markdown("---")
            st.markdown("#### Top Model Factors Influencing Classification")
            feat_imp_file = 'outputs/feature_importances.csv'
            if os.path.exists(feat_imp_file):
                top_imp = pd.read_csv(feat_imp_file).head(10)
                fig_imp = px.bar(
                    top_imp,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Global Factor Importance (Fitted Model)',
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=350)
                st.plotly_chart(fig_imp, width="stretch")


# =========================================================================
# TAB 2: DEMAND FORECASTER
# =========================================================================
with tab_fore:
    st.header("Daily Appointment Demand Forecaster")
    st.caption("Specialty and city forecasts are approximate (overall-series model applied to a filtered series).")
    st.markdown(
        "Forecast daily appointment volumes overall and across specialized rehabilitation services. "
        "Captures 7-day cyclicality, seasonal shifts, and helps clinic administrators optimize staffing schedules."
    )

    fore_artifact = get_forecasting_artifact()
    df_data = get_dataset()
    shipped_daily_series = get_daily_series()

    if fore_artifact is None or shipped_daily_series is None:
        st.warning("⚠️ Forecasting artifact or `data/daily_series.csv` is unavailable. Restore the repository files or run the training pipeline.")
    else:
        meta_ts = fore_artifact.get('metadata', {})
        best_fore_name = fore_artifact.get('model_name', 'Forecasting Model')
        ts_metrics = meta_ts.get('metrics', {})

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Forecasting Engine", best_fore_name)
        with m2:
            st.metric("Test MAE", metric_value(ts_metrics, 'mae', '.1f'))
        with m3:
            st.metric("Test RMSE", metric_value(ts_metrics, 'rmse', '.1f'))
        with m4:
            st.metric("Test R² Score", metric_value(ts_metrics, 'r2', '.4f'))
        with m5:
            st.metric("Test sMAPE", f"{metric_value(ts_metrics, 'smape', '.1f')}%" if ts_metrics.get('smape') is not None else "Unavailable")

        st.markdown("---")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            spec_options = ['All'] + sorted(list(df_data['specialty'].dropna().unique())) if df_data is not None else ['All']
            selected_specialty = st.selectbox("Filter Specialty", options=spec_options, index=0, disabled=df_data is None)
        with fc2:
            loc_options = ['All'] + TOP_MUNICIPALITIES if df_data is not None else ['All']
            selected_location = st.selectbox("Filter Location / Municipality", options=loc_options, index=0, disabled=df_data is None)
        with fc3:
            horizon = st.slider("Forecast Horizon (Days Ahead)", min_value=7, max_value=30, value=14, step=7)

        run_forecast_btn = st.button("🚀 Generate Demand Forecast", width="stretch")
        if run_forecast_btn:
            st.session_state['forecast_requested'] = True
        if df_data is None:
            st.caption("Overall forecasting uses the included daily aggregate. Add local `data/raw.csv` and rerun the pipeline to enable specialty and location filters.")

        # Build daily series
        daily_series = (
            build_daily_time_series(df_data, specialty=selected_specialty, location=selected_location)
            if df_data is not None else shipped_daily_series
        )

        if not st.session_state.get('forecast_requested', False):
            st.info('Select Generate Demand Forecast to run this forecast.')
        else:
            try:
                with st.spinner("Generating recursive demand forecast..."):
                    with LatencyTimer() as f_timer:
                        future_forecast = forecast_future_demand(fore_artifact, daily_series, horizon_days=horizon)
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                st.stop()

            f_latency_ms = f_timer.elapsed_ms

            st.markdown(f"<span class='latency-tag'>Forecast generated in {f_latency_ms:.1f} ms</span>", unsafe_allow_html=True)

            # Plot Historical + Forecast
            history_window = daily_series.tail(60).copy()

            fig_ts = go.Figure()
            # Historical trace
            fig_ts.add_trace(go.Scatter(
                x=history_window['appointment_date'],
                y=history_window['appointment_count'],
                mode='lines',
                name='Historical Demand (Recent 60 Days)',
                line=dict(color='#1f77b4', width=2)
            ))
            # Forecast trace
            fig_ts.add_trace(go.Scatter(
                x=future_forecast['appointment_date'],
                y=future_forecast['predicted_appointments'],
                mode='lines+markers',
                name=f'Recursive Forecast ({horizon} Days)',
                line=dict(color='#ff7f0e', width=2.5, dash='dash'),
                marker=dict(size=6, color='#d62728')
            ))

            fig_ts.update_layout(
                title=f"Daily Appointment Demand: Historical Baseline & {horizon}-Day Projected Forecast ({selected_specialty} | {selected_location})",
                xaxis_title="Date",
                yaxis_title="Daily Appointments",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=450
            )
            st.plotly_chart(fig_ts, width="stretch")

            # Forecast Summary KPIs
            tot_proj = int(future_forecast['predicted_appointments'].sum())
            avg_proj = float(future_forecast['predicted_appointments'].mean())
            peak_row = future_forecast.loc[future_forecast['predicted_appointments'].idxmax()]
            weekend_proj = int(future_forecast[future_forecast['is_weekend'] == 1]['predicted_appointments'].sum())

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Projected Appointments", f"{tot_proj:,}")
            with k2:
                st.metric("Avg Daily Projected Load", f"{avg_proj:.1f}")
            with k3:
                st.metric("Projected Peak Day", f"{peak_row['appointment_date'].strftime('%b %d')} ({peak_row['predicted_appointments']})")
            with k4:
                st.metric("Projected Weekend Volume", f"{weekend_proj:,}")

            # Staffing Allocation Recommendations
            st.subheader("Intelligent Staffing & Scheduling Recommendations")
            # Estimate specialist hours (assuming ~45 min per appointment = 0.75 hours)
            est_hours = tot_proj * 0.75
            est_fte = est_hours / (horizon * 7.0)  # 7 productive hours per therapist per day

            st.info(
                f"**Resource Allocation Projection:**\n"
                f"- Projected therapeutic load over {horizon} days requires approximately **{est_hours:,.0f} direct therapy hours**.\n"
                f"- Recommended active specialist staffing: **~{max(1, int(round(est_fte)))} Full-Time Therapists** on duty during peak weekday shifts.\n"
                f"- Peak anticipated volume falls on **{peak_row['day_of_week']} ({peak_row['appointment_date'].strftime('%Y-%m-%d')})**; ensure full shift coverage and minimal planned leave on this date."
            )

            with st.expander("View Daily Forecast Table"):
                st.dataframe(future_forecast[['appointment_date', 'day_of_week', 'predicted_appointments', 'is_weekend']], width="stretch")


    # =========================================================================
    # TAB 3: OPERATIONAL INSIGHTS & DATA DICTIONARY
# =========================================================================
with tab_insights:
    st.header("Comprehensive Operational Insights & Architecture")

    try:
        with open('outputs/insights.json', encoding='utf-8') as insights_file:
            insights = json.load(insights_file)
        st.info(insights['sms_conclusion'])
        for key in ['place_type', 'heat_intensity_by_place_type', 'age_group', 'specialty', 'companion', 'shift', 'weekday_volume', 'health_segments']:
            st.markdown(f"#### {key.replace('_', ' ').title()}")
            st.json(insights[key])
        st.markdown('#### Segment reminder recommendations')
        st.json(insights['segment_reminder_recommendations'])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        st.error(f'Generated insights are unavailable: {exc}')
    st.markdown("---")
    st.subheader("Dataset Dictionary")
    dict_df = pd.DataFrame([
        {"Feature": k, "Description": v} for k, v in DATA_DICTIONARY.items()
    ])
    st.dataframe(dict_df, width="stretch", height=400)
