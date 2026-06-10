import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="ChurnIQ — Batch Analysis",
    page_icon="📊",
    layout="wide"
)

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return {
        "model"         : joblib.load(f"{base}/models/final_model.pkl"),
        "scaler"        : joblib.load(f"{base}/models/scaler.pkl"),
        "scale_cols"    : joblib.load(f"{base}/models/scale_cols.pkl"),
        "features"      : joblib.load(f"{base}/models/feature_names.pkl"),
        "threshold"     : joblib.load(f"{base}/models/optimal_threshold.pkl"),
        "kmeans"        : joblib.load(f"{base}/models/kmeans_model.pkl"),
        "cluster_scaler": joblib.load(f"{base}/models/cluster_scaler.pkl"),
        "cluster_feats" : joblib.load(f"{base}/models/cluster_features.pkl"),
        "label_map"     : joblib.load(f"{base}/models/segment_label_map.pkl"),
    }

m = load_models()

# ── Helpers ──
def risk_tier(prob):
    if prob >= 0.75:   return "Very High"
    elif prob >= 0.51: return "High"
    elif prob >= 0.30: return "Medium"
    else:              return "Low"

def risk_color(tier):
    return {"Very High":"#FF4B6E","High":"#FF8C42",
            "Medium":"#FFD166","Low":"#06D6A0"}.get(tier,"#8B97B8")

def engineer_features(df_raw):
    """Engineer features from raw cleaned CSV to match model input."""
    df = df_raw.copy()

    # TotalCharges fix if present
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'].astype(str).str.strip().replace('', np.nan),
            errors='coerce'
        )
        df.dropna(subset=['TotalCharges'], inplace=True)

    # Drop non-feature cols
    for col in ['customerID', 'Churn']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # SeniorCitizen
    if df['SeniorCitizen'].dtype in [int, float]:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({1:'Yes',0:'No'})

    # Derived
    df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['TenureGroup'] = pd.cut(
        df['tenure'], bins=[0,12,24,48,72], labels=[0,1,2,3]
    ).astype(int)

    addon_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                  'TechSupport','StreamingTV','StreamingMovies']
    df['NumAddons'] = df[addon_cols].apply(
        lambda r: sum(v=='Yes' for v in r), axis=1
    )
    df['HasStreaming'] = (
        (df['StreamingTV']=='Yes')|(df['StreamingMovies']=='Yes')
    ).astype(int)
    df['IsLongTermContract'] = df['Contract'].isin(
        ['One year','Two year']
    ).astype(int)
    df['IsAutoPayment'] = df['PaymentMethod'].isin(
        ['Credit card (automatic)','Bank transfer (automatic)']
    ).astype(int)

    # Binary encoding
    binary_cols = ['gender','SeniorCitizen','Partner','Dependents',
                   'PhoneService','PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1,'No':0,'Male':1,'Female':0})

    service_map = {'Yes':1,'No':0,'No phone service':0,'No internet service':0}
    for col in ['MultipleLines','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']:
        if col in df.columns:
            df[col] = df[col].map(service_map)

    # OHE
    ohe_cols = ['InternetService','Contract','PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Drop TotalCharges if present
    if 'TotalCharges' in df.columns:
        df.drop(columns=['TotalCharges'], inplace=True)

    return df

def align_features(df_eng, feature_names):
    """Align engineered df to exact model feature list."""
    for col in feature_names:
        if col not in df_eng.columns:
            df_eng[col] = 0
    return df_eng[feature_names]

def run_batch(df_raw):
    df_eng  = engineer_features(df_raw)
    df_feat = align_features(df_eng.copy(), m["features"])

    # Scale
    cols_to_scale = [c for c in m["scale_cols"] if c in df_feat.columns]
    df_feat[cols_to_scale] = m["scaler"].transform(df_feat[cols_to_scale])

    # Predict
    probs      = m["model"].predict_proba(df_feat)[:, 1]
    preds      = (probs >= m["threshold"]).astype(int)
    risk_tiers = [risk_tier(p) for p in probs]

    # Segment
    cluster_feat = m["cluster_feats"]
    df_clust = df_eng.copy()
    for col in cluster_feat:
        if col not in df_clust.columns:
            df_clust[col] = 0
    df_clust = df_clust[cluster_feat]
    scaled_c = m["cluster_scaler"].transform(df_clust)
    clusters = m["kmeans"].predict(scaled_c)
    segments = [m["label_map"][c] for c in clusters]

    # Build result
    result = df_raw.copy()
    if 'customerID' in result.columns:
        id_col = result['customerID']
    else:
        id_col = pd.Series([f"CUST-{i+1:04d}" for i in range(len(result))])

    out = pd.DataFrame({
        'CustomerID'        : id_col.values,
        'Churn_Probability' : np.round(probs * 100, 1),
        'Churn_Predicted'   : ['Yes' if p else 'No' for p in preds],
        'Risk_Tier'         : risk_tiers,
        'Segment'           : segments,
        'Tenure_Months'     : df_raw['tenure'].values if 'tenure' in df_raw.columns else 0,
        'Monthly_Charges'   : df_raw['MonthlyCharges'].values if 'MonthlyCharges' in df_raw.columns else 0,
    })
    return out.sort_values('Churn_Probability', ascending=False).reset_index(drop=True)

def priority_color(tier):
    return {"Very High":"#FF4B6E","High":"#FF8C42",
            "Medium":"#FFD166","Low":"#06D6A0"}.get(tier,"#8B97B8")

# ── PAGE HEADER ──
st.markdown("""
<div style="margin-bottom:2rem">
    <div style="color:#4F7FFF;font-size:0.75rem;font-weight:600;
                letter-spacing:0.1em;text-transform:uppercase;
                margin-bottom:8px;margin-top:50px">Batch Processing</div>
    <h1 style="font-size:1.8rem;font-weight:700;color:#F0F4FF;
               margin:0;letter-spacing:-0.01em">Batch Customer Analysis</h1>
    <p style="color:#8B97B8;font-size:0.9rem;margin-top:6px">
        Upload a CSV file to score multiple customers simultaneously.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ──
st.markdown("""
<div style="background:#1C2333;border:1px solid #2A3350;
            border-radius:12px;padding:1.5rem;margin-bottom:1.5rem">
    <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                margin-bottom:0.5rem">Upload Customer Data</div>
    <div style="color:#8B97B8;font-size:0.82rem;margin-bottom:1rem">
        Upload a CSV with the same columns as the IBM Telco dataset.
        Required columns: tenure, MonthlyCharges, Contract, InternetService, PaymentMethod, and service columns.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose CSV file",
    type=["csv"],
    help="Upload telecom customer data CSV"
)

# ── Demo Mode ──
use_demo = st.checkbox(
    "Use sample dataset (demo mode)",
    value=False,
    help="Run batch analysis on the built-in dataset"
)

if use_demo and not uploaded:
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    demo_path = f"{base}/data/cleaned_churn.csv"
    if os.path.exists(demo_path):
        df_raw = pd.read_csv(demo_path).head(200)
        # Reverse-map Churn for demo
        if 'Churn' in df_raw.columns and df_raw['Churn'].dtype in [int, float]:
            df_raw['Churn'] = df_raw['Churn'].map({1:'Yes',0:'No'})
        # Reverse SeniorCitizen
        if 'SeniorCitizen' in df_raw.columns:
            df_raw['SeniorCitizen'] = df_raw['SeniorCitizen'].map(
                {'Yes':'Yes','No':'No',1:'Yes',0:'No'}
            )
        st.success(f"Demo mode: loaded {len(df_raw)} customers")
    else:
        st.error("Demo dataset not found. Please upload a CSV.")
        df_raw = None
elif uploaded:
    df_raw = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df_raw)} customers from {uploaded.name}")
else:
    df_raw = None

if df_raw is not None:
    with st.spinner("Scoring customers..."):
        try:
            results = run_batch(df_raw)
        except Exception as e:
            st.error(f"Processing error: {e}")
            st.stop()

    # ── Summary KPIs ──
    st.markdown("<div style='margin:1.5rem 0 1rem'>", unsafe_allow_html=True)

    total     = len(results)
    churners  = (results['Churn_Predicted']=='Yes').sum()
    churn_rate= churners / total * 100
    avg_prob  = results['Churn_Probability'].mean()
    critical  = (results['Risk_Tier']=='Very High').sum()

    k1,k2,k3,k4 = st.columns(4)

    def kpi(label, value, color="#4F7FFF"):
        return f"""
        <div style="background:#1C2333;border:1px solid #2A3350;
                    border-left:3px solid {color};border-radius:12px;
                    padding:1.2rem 1.5rem">
            <div style="color:#8B97B8;font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.08em;font-weight:600;margin-bottom:6px">{label}</div>
            <div style="color:#F0F4FF;font-size:1.8rem;
                        font-weight:700;line-height:1">{value}</div>
        </div>"""

    with k1: st.markdown(kpi("Total Customers", f"{total:,}"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("Predicted Churners", f"{churners:,}", "#FF4B6E"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("Churn Rate", f"{churn_rate:.1f}%", "#FF8C42"), unsafe_allow_html=True)
    with k4: st.markdown(kpi("Very High Risk", f"{critical:,}", "#FFD166"), unsafe_allow_html=True)

    st.markdown("<div style='margin:1.5rem 0'>", unsafe_allow_html=True)

    # ── Segment & Risk Distribution ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                    margin-bottom:1rem">Risk Tier Breakdown</div>
        """, unsafe_allow_html=True)

        tier_counts = results['Risk_Tier'].value_counts()
        tier_order  = ["Very High","High","Medium","Low"]
        tier_colors = ["#FF4B6E","#FF8C42","#FFD166","#06D6A0"]

        for tier, color in zip(tier_order, tier_colors):
            count = tier_counts.get(tier, 0)
            pct   = count / total * 100
            st.markdown(f"""
            <div style="margin-bottom:0.7rem">
                <div style="display:flex;justify-content:space-between;
                            margin-bottom:4px">
                    <span style="color:#F0F4FF;font-size:0.82rem;
                                 font-weight:500">{tier}</span>
                    <span style="color:{color};font-size:0.82rem;
                                 font-weight:600">{count} ({pct:.0f}%)</span>
                </div>
                <div style="background:#2A3350;border-radius:4px;height:6px">
                    <div style="background:{color};width:{pct}%;
                                height:6px;border-radius:4px;
                                transition:width 0.3s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                    margin-bottom:1rem">Segment Distribution</div>
        """, unsafe_allow_html=True)

        seg_counts = results['Segment'].value_counts()
        seg_colors = {
            'Critical Churn-Risk' : '#FF4B6E',
            'Vulnerable High-Risk': '#FF8C42',
            'Stable Mid-Risk'     : '#FFD166',
            'Loyal Low-Risk'      : '#06D6A0',
        }
        for seg, color in seg_colors.items():
            count = seg_counts.get(seg, 0)
            pct   = count / total * 100
            st.markdown(f"""
            <div style="margin-bottom:0.7rem">
                <div style="display:flex;justify-content:space-between;
                            margin-bottom:4px">
                    <span style="color:#F0F4FF;font-size:0.82rem;
                                 font-weight:500">{seg}</span>
                    <span style="color:{color};font-size:0.82rem;
                                 font-weight:600">{count} ({pct:.0f}%)</span>
                </div>
                <div style="background:#2A3350;border-radius:4px;height:6px">
                    <div style="background:{color};width:{pct}%;
                                height:6px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1.5rem 0'>", unsafe_allow_html=True)

    # ── Results Table ──
    st.markdown("""
    <div style="color:#F0F4FF;font-size:0.9rem;font-weight:600;
                margin-bottom:0.5rem">Customer Risk Table</div>
    <div style="color:#8B97B8;font-size:0.8rem;margin-bottom:1rem">
        Sorted by churn probability — highest risk first.
    </div>
    """, unsafe_allow_html=True)

    # Filter controls
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_tier = st.multiselect(
            "Filter by Risk Tier",
            ["Very High","High","Medium","Low"],
            default=["Very High","High"]
        )
    with fc2:
        filter_seg = st.multiselect(
            "Filter by Segment",
            list(seg_colors.keys()),
            default=[]
        )
    with fc3:
        filter_pred = st.selectbox(
            "Churn Prediction",
            ["All","Yes - Churners Only","No - Retained Only"]
        )

    filtered = results.copy()
    if filter_tier:
        filtered = filtered[filtered['Risk_Tier'].isin(filter_tier)]
    if filter_seg:
        filtered = filtered[filtered['Segment'].isin(filter_seg)]
    if filter_pred == "Yes - Churners Only":
        filtered = filtered[filtered['Churn_Predicted']=='Yes']
    elif filter_pred == "No - Retained Only":
        filtered = filtered[filtered['Churn_Predicted']=='No']

    st.markdown(f"""
    <div style="color:#8B97B8;font-size:0.78rem;margin-bottom:0.5rem">
        Showing {len(filtered):,} of {total:,} customers
    </div>
    """, unsafe_allow_html=True)

    # Style dataframe
    def style_risk(val):
        colors = {"Very High":"#FF4B6E","High":"#FF8C42",
                  "Medium":"#FFD166","Low":"#06D6A0"}
        c = colors.get(val,"")
        return f"color:{c};font-weight:600" if c else ""

    def style_churn(val):
        return "color:#FF4B6E;font-weight:600" if val=="Yes" else "color:#06D6A0"

    display_df = filtered.copy()
    display_df['Churn_Probability'] = display_df['Churn_Probability'].apply(
        lambda x: f"{x:.1f}%"
    )

    styled = (
        display_df.style
        .applymap(style_risk,   subset=['Risk_Tier'])
        .applymap(style_churn,  subset=['Churn_Predicted'])
        .set_properties(**{
            'background-color': '#1C2333',
            'color'           : '#F0F4FF',
            'border-color'    : '#2A3350',
            'font-size'       : '13px',
        })
        .hide(axis='index')
    )
    st.dataframe(styled, use_container_width=True, height=400)

    # ── Download ──
    st.markdown("<div style='margin:1rem 0'>", unsafe_allow_html=True)
    csv_out = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇  Download Results CSV",
        data=csv_out,
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=False
    )

else:
    # ── Empty State ──
    st.markdown("""
    <div style="background:#1C2333;border:1px dashed #2A3350;
                border-radius:12px;padding:3rem;text-align:center;
                margin-top:1rem">
        <div style="font-size:2.5rem;margin-bottom:1rem">📂</div>
        <div style="color:#F0F4FF;font-size:1rem;
                    font-weight:600;margin-bottom:8px">No data loaded</div>
        <div style="color:#8B97B8;font-size:0.85rem;max-width:400px;margin:0 auto">
            Upload a CSV file above or enable demo mode to analyze
            customer churn risk across your entire base.
        </div>
    </div>
    """, unsafe_allow_html=True)
