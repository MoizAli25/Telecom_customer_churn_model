import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="ChurnIQ — Predict",
    page_icon="🔍",
    layout="wide"
)

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Load Models ──
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return {
        "model"      : joblib.load(f"{base}/models/final_model.pkl"),
        "scaler"     : joblib.load(f"{base}/models/scaler.pkl"),
        "scale_cols" : joblib.load(f"{base}/models/scale_cols.pkl"),
        "features"   : joblib.load(f"{base}/models/feature_names.pkl"),
        "threshold"  : joblib.load(f"{base}/models/optimal_threshold.pkl"),
        "kmeans"     : joblib.load(f"{base}/models/kmeans_model.pkl"),
        "cluster_scaler": joblib.load(f"{base}/models/cluster_scaler.pkl"),
        "cluster_feats" : joblib.load(f"{base}/models/cluster_features.pkl"),
        "label_map"  : joblib.load(f"{base}/models/segment_label_map.pkl"),
    }

m = load_models()

# ── Helpers ──
def risk_tier(prob):
    if prob >= 0.75:   return "Very High", "#FF4B6E", "rgba(255,75,110,0.10)"
    elif prob >= 0.51: return "High",      "#FF8C42", "rgba(255,140,66,0.10)"
    elif prob >= 0.30: return "Medium",    "#FFD166", "rgba(255,209,102,0.10)"
    else:              return "Low",       "#06D6A0", "rgba(6,214,160,0.10)"

def get_recommendations(segment, prob, tenure, monthly, longterm,
                         fiber, security, support, auto_pay, addons):
    recs = []
    if segment == "Critical Churn-Risk":
        if tenure <= 12:
            recs.append(("🚨", "Immediate onboarding call", "Assign a dedicated support agent within 24 hours"))
            recs.append(("💰", "Early discount offer", "20% off for first 3 months to reduce early churn"))
        else:
            recs.append(("📞", "Personalized retention call", "Acknowledge loyalty and offer exclusive deal"))
            recs.append(("💰", "Loyalty discount", "15% off for 6 months for long-standing customers"))
        if not longterm:
            recs.append(("📋", "Contract upgrade incentive", "Offer 2 free months on annual plan commitment"))
        if fiber and monthly >= 80:
            recs.append(("🌐", "Fiber service review", "Speed upgrade or bill reduction to justify cost"))
        if not security or not support:
            recs.append(("🔒", "Security & Support bundle", "OnlineSecurity + TechSupport at 50% off for 3 months"))
        if not auto_pay:
            recs.append(("💳", "Auto-pay enrollment", "$5/month discount for switching to automatic payment"))

    elif segment == "Vulnerable High-Risk":
        if tenure <= 12:
            recs.append(("📧", "Welcome engagement", "Send personalized value summary and onboarding email"))
            recs.append(("🎁", "First add-on free trial", "Complimentary add-on service for 2 months"))
        else:
            recs.append(("📋", "Mid-cycle check-in", "Proactive satisfaction survey with exclusive offer"))
        if not longterm:
            recs.append(("📋", "Annual contract promotion", "Highlight monthly savings vs current plan"))
        if addons <= 1:
            recs.append(("🔒", "Add-on recommendation", "Demonstrate OnlineSecurity or TechSupport value"))
        if monthly <= 50:
            recs.append(("📺", "Plan upsell opportunity", "Enhanced plan with streaming bundle at reduced rate"))

    elif segment == "Stable Mid-Risk":
        recs.append(("🏆", "Quarterly loyalty reward", "Bonus data or service credit for continued loyalty"))
        if addons <= 2:
            recs.append(("➕", "Cross-sell add-ons", "DeviceProtection or StreamingTV at discounted rate"))
        if fiber:
            recs.append(("🌐", "Proactive quality check", "Fiber performance review to prevent dissatisfaction"))
        if not auto_pay:
            recs.append(("💳", "Auto-pay promotion", "Enroll for convenience discount"))

    elif segment == "Loyal Low-Risk":
        recs.append(("🎖️", "Annual loyalty reward", "Thank-you gift card or complimentary free month"))
        if addons == 0:
            recs.append(("📺", "Premium bundle offer", "StreamingTV + Movies at 30% off"))
        recs.append(("👥", "Referral program", "Reward customer for bringing new subscribers"))

    return recs

def build_feature_row(inputs):
    """Build a single-row DataFrame matching model feature names."""
    feat = m["features"]
    row  = {f: 0 for f in feat}

    # Direct mappings
    row["tenure"]          = inputs["tenure"]
    row["MonthlyCharges"]  = inputs["monthly"]
    row["MultipleLines"]   = 1 if inputs["multi_lines"] == "Yes" else 0
    row["OnlineSecurity"]  = 1 if inputs["security"] == "Yes" else 0
    row["TechSupport"]     = 1 if inputs["tech_support"] == "Yes" else 0
    row["StreamingTV"]     = 1 if inputs["streaming_tv"] == "Yes" else 0
    row["PaperlessBilling"]= 1 if inputs["paperless"] == "Yes" else 0
    row["Dependents"]      = 1 if inputs["dependents"] == "Yes" else 0
    row["HasStreaming"]     = 1 if (inputs["streaming_tv"]=="Yes" or inputs["streaming_movies"]=="Yes") else 0
    row["IsAutoPayment"]   = 1 if inputs["payment"] in ["Credit card (automatic)", "Bank transfer (automatic)"] else 0
    row["IsLongTermContract"] = 1 if inputs["contract"] in ["One year", "Two year"] else 0

    # NumAddons
    addon_fields = [inputs["security"], inputs["backup"], inputs["device_prot"],
                    inputs["tech_support"], inputs["streaming_tv"], inputs["streaming_movies"]]
    row["NumAddons"] = sum(1 for v in addon_fields if v == "Yes")

    # TenureGroup
    t = inputs["tenure"]
    row["TenureGroup"] = 0 if t<=12 else (1 if t<=24 else (2 if t<=48 else 3))

    # ChargesPerTenure
    row["ChargesPerTenure"] = inputs["monthly"] / (inputs["tenure"] + 1)

    # OHE — Contract
    if inputs["contract"] == "Month-to-month":
        row["Contract_Month-to-month"] = 1
    elif inputs["contract"] == "One year":
        row["Contract_One year"] = 1
    elif inputs["contract"] == "Two year":
        row["Contract_Two year"] = 1

    # OHE — InternetService
    if inputs["internet"] == "DSL":
        row["InternetService_DSL"] = 1
    elif inputs["internet"] == "Fiber optic":
        row["InternetService_Fiber optic"] = 1
    elif inputs["internet"] == "No":
        row["InternetService_No"] = 1

    # OHE — PaymentMethod
    pm_map = {
        "Electronic check"         : "PaymentMethod_Electronic check",
        "Credit card (automatic)"  : "PaymentMethod_Credit card (automatic)",
    }
    if inputs["payment"] in pm_map:
        row[pm_map[inputs["payment"]]] = 1

    df_row = pd.DataFrame([row])

    # Scale
    cols_to_scale = [c for c in m["scale_cols"] if c in df_row.columns]
    df_row[cols_to_scale] = m["scaler"].transform(df_row[cols_to_scale])

    return df_row

def predict_segment(inputs):
    cluster_feat = m["cluster_feats"]
    row = {f: 0 for f in cluster_feat}
    row["tenure"]          = inputs["tenure"]
    row["MonthlyCharges"]  = inputs["monthly"]
    row["ChargesPerTenure"]= inputs["monthly"] / (inputs["tenure"] + 1)
    row["IsLongTermContract"] = 1 if inputs["contract"] in ["One year", "Two year"] else 0
    row["Contract_Month-to-month"] = 1 if inputs["contract"] == "Month-to-month" else 0
    row["Contract_Two year"]       = 1 if inputs["contract"] == "Two year" else 0
    row["InternetService_Fiber optic"] = 1 if inputs["internet"] == "Fiber optic" else 0
    row["InternetService_DSL"]         = 1 if inputs["internet"] == "DSL" else 0
    row["TechSupport"]    = 1 if inputs["tech_support"] == "Yes" else 0
    row["OnlineSecurity"] = 1 if inputs["security"] == "Yes" else 0
    row["PaperlessBilling"]= 1 if inputs["paperless"] == "Yes" else 0
    row["HasStreaming"]    = 1 if (inputs["streaming_tv"]=="Yes" or inputs["streaming_movies"]=="Yes") else 0
    row["IsAutoPayment"]   = 1 if inputs["payment"] in ["Credit card (automatic)", "Bank transfer (automatic)"] else 0
    t = inputs["tenure"]
    row["TenureGroup"]     = 0 if t<=12 else (1 if t<=24 else (2 if t<=48 else 3))
    row["Dependents"]      = 1 if inputs["dependents"] == "Yes" else 0
    addon_fields = [inputs["security"], inputs["backup"], inputs["device_prot"],
                    inputs["tech_support"], inputs["streaming_tv"], inputs["streaming_movies"]]
    row["NumAddons"] = sum(1 for v in addon_fields if v == "Yes")

    df_row  = pd.DataFrame([row])[cluster_feat]
    scaled  = m["cluster_scaler"].transform(df_row)
    cluster = m["kmeans"].predict(scaled)[0]
    return m["label_map"][cluster]

# ── PAGE HEADER ──
st.markdown("""
<div style="margin-bottom:2rem">
    <div style="color:#4F7FFF;font-size:0.75rem;font-weight:600;
                letter-spacing:0.1em;text-transform:uppercase;
                margin-bottom:8px">Single Customer Analysis</div>
    <h1 style="font-size:1.8rem;font-weight:700;color:#707070;
               margin:0;letter-spacing:-0.01em">Customer Churn Predictor</h1>
    <p style="color:#8B97B8;font-size:0.9rem;margin-top:6px">
        Enter customer details to generate churn probability, risk tier, and retention actions.
    </p>
</div>
""", unsafe_allow_html=True)

# ── INPUT FORM ──
with st.form("predict_form"):

    st.markdown("""
    <div style="color:#8B97B8;font-size:0.72rem;text-transform:uppercase;
                letter-spacing:0.08em;font-weight:600;margin-bottom:1rem">
        Account Information
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        tenure  = st.number_input("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
    with c2:
        contract   = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet   = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with c3:
        payment    = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless  = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown("<hr style='border-color:#2A3350;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8B97B8;font-size:0.72rem;text-transform:uppercase;
                letter-spacing:0.08em;font-weight:600;margin-bottom:1rem">
        Services & Demographics
    </div>""", unsafe_allow_html=True)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        security    = st.selectbox("Online Security",   ["No", "Yes"])
        backup      = st.selectbox("Online Backup",     ["No", "Yes"])
    with c5:
        device_prot = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support= st.selectbox("Tech Support",      ["No", "Yes"])
    with c6:
        streaming_tv    = st.selectbox("Streaming TV",     ["No", "Yes"])
        streaming_movies= st.selectbox("Streaming Movies", ["No", "Yes"])
    with c7:
        multi_lines = st.selectbox("Multiple Lines",    ["No", "Yes"])
        dependents  = st.selectbox("Dependents",        ["No", "Yes"])

    submitted = st.form_submit_button("⚡  Analyze Customer", use_container_width=True)

# ── RESULTS ──
if submitted:
    inputs = dict(
        tenure=tenure, monthly=monthly, contract=contract,
        internet=internet, payment=payment, paperless=paperless,
        security=security, backup=backup, device_prot=device_prot,
        tech_support=tech_support, streaming_tv=streaming_tv,
        streaming_movies=streaming_movies, multi_lines=multi_lines,
        dependents=dependents
    )

    with st.spinner("Running inference..."):
        df_input  = build_feature_row(inputs)
        prob      = float(m["model"].predict_proba(df_input)[:, 1][0])
        tier, color, bg = risk_tier(prob)
        segment   = predict_segment(inputs)
        recs      = get_recommendations(
            segment, prob, tenure, monthly,
            inputs["contract"] in ["One year","Two year"],
            inputs["internet"] == "Fiber optic",
            inputs["security"] == "Yes",
            inputs["tech_support"] == "Yes",
            inputs["payment"] in ["Credit card (automatic)","Bank transfer (automatic)"],
            sum(1 for v in [security,backup,device_prot,tech_support,
                            streaming_tv,streaming_movies] if v=="Yes")
        )

    st.markdown("<div style='margin:1.5rem 0 1rem'>", unsafe_allow_html=True)

    # ── Result Banner ──
    verdict = "⚠️ Likely to Churn" if prob >= m["threshold"] else "✅ Likely to Stay"
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {color};
                border-radius:12px;padding:1.5rem 2rem;
                display:flex;justify-content:space-between;
                align-items:center;margin-bottom:1.5rem">
        <div>
            <div style="font-size:1.4rem;font-weight:700;
                        color:{color}">{verdict}</div>
            <div style="color:#8B97B8;font-size:0.85rem;margin-top:4px">
                Segment: <span style="color:#707070;font-weight:500">{segment}</span>
            </div>
        </div>
        <div style="text-align:right">
            <div style="font-size:2.8rem;font-weight:700;
                        color:{color};line-height:1">{prob*100:.1f}%</div>
            <div style="color:#8B97B8;font-size:0.8rem">churn probability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Row ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div style="background:#1C2333;border:1px solid #2A3350;
                    border-radius:10px;padding:1rem;text-align:center">
            <div style="color:#8B97B8;font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:6px">Risk Tier</div>
            <span style="background:{bg};color:{color};border:1px solid {color};
                         border-radius:20px;padding:4px 14px;
                         font-size:0.85rem;font-weight:600">{tier}</span>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="background:#1C2333;border:1px solid #2A3350;
                    border-radius:10px;padding:1rem;text-align:center">
            <div style="color:#8B97B8;font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:6px">Segment</div>
            <div style="color:#707070;font-size:0.85rem;
                        font-weight:600">{segment.split()[0]}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div style="background:#1C2333;border:1px solid #2A3350;
                    border-radius:10px;padding:1rem;text-align:center">
            <div style="color:#8B97B8;font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:6px">Tenure</div>
            <div style="color:#707070;font-size:0.85rem;
                        font-weight:600">{tenure} months</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div style="background:#1C2333;border:1px solid #2A3350;
                    border-radius:10px;padding:1rem;text-align:center">
            <div style="color:#8B97B8;font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:6px">Monthly</div>
            <div style="color:#707070;font-size:0.85rem;
                        font-weight:600">${monthly:.0f}</div>
        </div>""", unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown("""
    <div style="margin:1.5rem 0 1rem">
        <div style="color:#707070;font-size:1rem;font-weight:600;
                    margin-bottom:4px">Retention Recommendations</div>
        <div style="color:#8B97B8;font-size:0.82rem">
            Prioritized actions based on segment and churn drivers
        </div>
    </div>
    """, unsafe_allow_html=True)

    if recs:
        cols = st.columns(min(len(recs), 3))
        for i, (icon, title, desc) in enumerate(recs):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:#1C2333;border:1px solid #2A3350;
                            border-radius:10px;padding:1rem;
                            margin-bottom:0.8rem;height:110px">
                    <div style="font-size:1.2rem;margin-bottom:6px">{icon}</div>
                    <div style="color:#707070;font-size:0.82rem;
                                font-weight:600;margin-bottom:4px">{title}</div>
                    <div style="color:#8B97B8;font-size:0.76rem;
                                line-height:1.4">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1C2333;border:1px solid #2A3350;
                    border-radius:10px;padding:1rem;color:#8B97B8">
            No urgent actions — customer is low risk.
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)