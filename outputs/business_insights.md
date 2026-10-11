# Business Insights & Operational Recommendations

## Executive Summary
This document summarizes analytical findings from the Center of Specialization in Physical and Intellectual Rehabilitation (CER) dataset (109,557 unique appointments across 498 continuous calendar days from Jan 2020 to May 2021).

---

## 1. Key Drivers of Patient No-Show Behavior
1. **Severe Weather Sensitivity**:
   - During **heavy cold** conditions, the patient no-show rate rises to **52.9%** (vs. baseline average 31.8%).
   - Heavy warm conditions also elevate no-shows to **38.8%**.
   - Rehabilitation patients (many with motor or intellectual disabilities) encounter severe transport hurdles during extreme weather.
2. **Missing Clinical Specialty or Registration (`sem especialidade`)**:
   - Appointments without an assigned specialty have a **40.6% to 52.8%** no-show rate.
   - Administrative ambiguity correlates heavily with appointment abandonment.
3. **Age & Demographics**:
   - Young adults (19–40 years) exhibit the highest no-show rate at **45.5%**, followed by adolescents (13–18 years) at **35.1%**.
   - Elderly patients (60+) demonstrate significantly higher commitment, with only **22.4%** no-show rate.
   - Pediatric patients accompanied by caregivers have lower no-shows (**27.7%**).
4. **Caregiver Presence**:
   - Patients requiring a companion have a **29.0%** no-show rate compared to **34.8%** for unassisted patients. Caregiver engagement acts as a stabilizing attendance factor.
5. **Shift Timing**:
   - Morning appointments experience a **33.0%** no-show rate compared to **30.7%** in the afternoon.

---

## 2. Demand Patterns Across Specialties & Geography
1. **Specialty Demand Concentration**:
   - Psychotherapy (28,645 appointments), Speech Therapy (22,322 appointments), and Physiotherapy (21,004 appointments) comprise over **80%** of specialized capacity.
   - Pedagogy and nursing consultations experience the highest punctuality/attendance rates (>80%).
2. **Geographic Concentration**:
   - Itajaí represents the single largest municipal patient origin (over 20,500 appointments), followed by Balneário Camboriú and Camboriú.
   - Outlying municipalities exhibit higher transit-related cancellation risks during inclement weather.

---

## 3. Operational Staffing Implications
1. **Dynamic Staff Scheduling**:
   - Daily appointment demand displays pronounced 7-day cyclicality with mid-week peaks and reduced Sunday operating volumes.
   - Demand forecasting allows clinical directors to align specialist schedules 7 to 14 days in advance, mitigating specialist idle hours.
2. **Buffer Overbooking on High-Risk Slots**:
   - Instead of uniform booking, apply targeted double-booking ONLY on slots flagged as **High Risk** (probability > 0.65).
   - *Estimated Hypothetical Impact*: Assuming an outreach and buffer scheduling intervention reaches 25% of high-risk patients and reallocates 15% of open slots, clinic capacity utilization could recover approximately 4.5% to 6.2% of currently lost hours. (Note: This is an operational projection subject to clinical trial validation).

---

## 4. Patient Engagement & Intervention Strategies
1. **Weather-Triggered Reminders**:
   - Deploy automated SMS reminders and transport advisories 24 hours prior when rain/cold weather thresholds are forecasted.
2. **Targeted Young-Adult Engagement**:
   - Patients aged 19–40 should receive multi-channel confirmations (SMS + WhatsApp/call) 48 hours and 2 hours prior to scheduled time.
3. **Caregiver Inclusivity**:
   - For patients marked as requiring a companion, include the caregiver's contact in reminder pipelines.
